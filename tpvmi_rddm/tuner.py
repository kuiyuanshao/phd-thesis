import optuna
import copy
import numpy as np
import yaml
import torch
import pandas as pd

from tpvmi_rddm.tpvmi_rddm import TPVMI_RDDM
from tpvmi_rddm.utils import process_data, BiasCalc


class RDDMTuner:
    def __init__(self, model, base_config, data_info, param_grid, file_path, n_trials=50, n_folds=5):
        self.base_config = copy.deepcopy(base_config)
        self.data_info = data_info
        self.param_grid = param_grid
        self.file_path = file_path
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.study = None

        self.biascalc = BiasCalc(model)
        print("[RDDMTuner] Initializing: Loading and Processing Dataset...")
        (proc_data, proc_mask, p1_idx, p2_idx, _, schema, stats, _) = process_data(file_path, data_info)

        p2_mask = proc_mask[:, p2_idx.astype(int)]
        self.valid_rows_idx = np.where(p2_mask.min(axis=1) == 1.0)[0]

        print(
            f"[RDDMTuner] Tuning on Fully Observed Subsample. Size: {len(self.valid_rows_idx)} / {proc_data.shape[0]}")
        p1_sub = proc_data[self.valid_rows_idx][:, p1_idx.astype(int)]
        p2_sub = proc_data[self.valid_rows_idx][:, p2_idx.astype(int)]

        all_idx = set(range(proc_data.shape[1]))
        reserved = set(p1_idx.astype(int)) | set(p2_idx.astype(int))
        aux_idx = np.array(sorted(list(all_idx - reserved)), dtype=int)

        if len(aux_idx) > 0:
            aux_sub = proc_data[self.valid_rows_idx][:, aux_idx]
        else:
            aux_sub = np.zeros((len(self.valid_rows_idx), 0))

        self.tensor_data = {
            'p1': torch.from_numpy(p1_sub),
            'p2': torch.from_numpy(p2_sub),
            'aux': torch.from_numpy(aux_sub),
            'schema': schema,
            'stats': stats
        }

    def _update(self, d, target_key, target_val):
        for k, v in d.items():
            if k == target_key:
                d[k] = target_val
                return True
            elif isinstance(v, dict):
                if self._update(v, target_key, target_val):
                    return True
        return False


    def _get_trial_config(self, trial):
        config = copy.deepcopy(self.base_config)
        for param_name, params in self.param_grid.items():
            param_type = params[0]
            if param_type == "cat":
                choices = params[1]
                chosen_value = trial.suggest_categorical(param_name, choices)
            elif param_type == "int":
                low, high = params[1], params[2]
                chosen_value = trial.suggest_int(param_name, low, high)
            elif param_type == "float":
                low, high = params[1], params[2]
                chosen_value = trial.suggest_float(param_name, low, high)
            elif param_type == "log_float":
                low, high = params[1], params[2]
                chosen_value = trial.suggest_float(param_name, low, high, log=True)
            else:
                raise ValueError(f"Unknown parameter type: {param_type} for {param_name}")
            found = self._update(config, param_name, chosen_value)
            if not found:
                print(f"[Warning] Parameter '{param_name}' not found in base_config. It was skipped.")

            return config

    def objective(self, trial):
        trial_config = self._get_trial_config(trial)
        N = self.tensor_data['p1'].shape[0]
        indices = np.arange(N)
        np.random.shuffle(indices)
        fold_size = N // self.n_folds
        all_folds_imputed = []

        for k in range(self.n_folds):
            val_start = k * fold_size
            val_end = (k + 1) * fold_size

            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            train_data_subset = {
                'p1': self.tensor_data['p1'][train_idx],
                'p2': self.tensor_data['p2'][train_idx],
                'aux': self.tensor_data['aux'][train_idx],
                'schema': self.tensor_data['schema'],
                'stats': self.tensor_data['stats']
            }

            val_p1 = self.tensor_data['p1'][val_idx].float()
            val_aux = self.tensor_data['aux'][val_idx].float()

            model = TPVMI_RDDM(trial_config, self.data_info)
            model.fit(provided_data=train_data_subset)
            val_p1 = val_p1.to(model.device)
            val_aux = val_aux.to(model.device)
            fold_imputed_m_list = model.impute(m=trial_config["else"]["m"],
                                               p1=val_p1, aux=val_aux,
                                               batch_size=2048, fill=False)

            all_folds_imputed.append(fold_imputed_m_list)

        final_m_imputed_dfs = []
        m_count = len(all_folds_imputed[0])

        for i in range(m_count):
            folds_for_round_i = [fold_list[i] for fold_list in all_folds_imputed]
            full_df_i = pd.concat(folds_for_round_i, axis=0, ignore_index=True)
            final_m_imputed_dfs.append(full_df_i)

        current_bias = self.biascalc.evaluate_imputations(final_m_imputed_dfs)

        return current_bias

    def tune(self, save_best_config=True, config_path="best_config.yaml",
             save_results_log=True, results_path="tuning_results.csv"):
        """
        Executes the tuning process.

        Args:
            save_best_config (bool): Whether to save the YAML of the best run.
            config_path (str): Filename for the best YAML.
            save_results_log (bool): Whether to save a CSV of all trials and their bias.
            results_path (str): Filename for the CSV log.
        """
        self.study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        print(
            f"\n[RDDMTuner] Starting Optimization (Metric: Mean Bias Ratio): {self.n_trials} trials, {self.n_folds}-Fold CV")

        self.study.optimize(self.objective, n_trials=self.n_trials)

        print("\n[RDDMTuner] Optimization Finished.")

        # --- NEW LOGIC: EXPORT ALL RESULTS ---
        if save_results_log:
            try:
                # trials_dataframe() returns columns: number, value, datetime_start, params_*, etc.
                df_results = self.study.trials_dataframe()

                # Sort by performance (ascending bias)
                if 'value' in df_results.columns:
                    df_results = df_results.sort_values(by='value', ascending=True)

                df_results.to_csv(results_path, index=False)
                print(f"[RDDMTuner] Full tuning log saved to: {results_path}")
            except Exception as e:
                print(f"[RDDMTuner] Warning: Failed to save results log. Error: {e}")
        # -------------------------------------

        # Check for valid trials before accessing best_value
        complete_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        if len(complete_trials) == 0:
            print("[RDDMTuner] All trials failed. No best config available.")
            return self.base_config

        print(f"Best Value (Mean Bias Ratio): {self.study.best_value:.6f}")

        best_config = copy.deepcopy(self.base_config)
        for param_name, chosen_value in self.study.best_params.items():
            self._update(best_config, param_name, chosen_value)

        if save_best_config:
            with open(config_path, "w") as f:
                yaml.dump(best_config, f)
            print(f"[RDDMTuner] Best config saved to {config_path}")

        return best_config