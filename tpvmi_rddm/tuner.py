import optuna
import copy
import numpy as np
import yaml
import torch
import pandas as pd

from tpvmi_rddm.tpvmi_rddm import TPVMI_RDDM
from tpvmi_rddm.utils import process_data, BiasCalc


class RDDMTuner:
    def __init__(self, model, base_config, data_info, param_grid, file_path, n_trials=50, n_folds=5, weights=1):
        self.base_config = copy.deepcopy(base_config)
        self.data_info = data_info
        self.param_grid = param_grid
        self.file_path = file_path
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.study = None

        self.biascalc = BiasCalc(model, weights)
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

        N = self.tensor_data['p1'].shape[0]
        self.cv_indices = np.arange(N)
        rng = np.random.default_rng(42)
        rng.shuffle(self.cv_indices)

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

        # Iterate through ALL parameters
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

        return config  # <--- FIX: This must be OUTSIDE the for loop

    def objective(self, trial):
        trial_config = self._get_trial_config(trial)
        N = len(self.cv_indices)
        fold_size = N // self.n_folds
        all_folds_imputed = []

        for k in range(self.n_folds):
            val_start = k * fold_size
            val_end = (k + 1) * fold_size

            val_idx = self.cv_indices[val_start:val_end]
            train_idx = np.concatenate([self.cv_indices[:val_start], self.cv_indices[val_end:]])

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
        self.study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        print(
            f"\n[RDDMTuner] Starting Optimization (Metric: Mean Bias Ratio): {self.n_trials} trials, {self.n_folds}-Fold CV")

        self.study.optimize(self.objective, n_trials=self.n_trials)

        print("\n[RDDMTuner] Optimization Finished.")

        if save_results_log:
            try:
                df_results = self.study.trials_dataframe()
                if 'value' in df_results.columns:
                    df_results = df_results.sort_values(by='value', ascending=True)

                df_results.to_csv(results_path, index=False)
                print(f"[RDDMTuner] Full tuning log saved to: {results_path}")
            except Exception as e:
                print(f"[RDDMTuner] Warning: Failed to save results log. Error: {e}")
        # -------------------------------------

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

class BiasCalc:
    def __init__(self, template_model, weights):
        self.template = template_model
        self.type = self._identify_type(template_model)
        self.reference_coeffs = self._extract_coeffs(self.template)

        # Ensure weights align with reference coefficients
        if isinstance(weights, list):
            # Auto-pad or trim if lengths differ (Safety mechanism)
            if len(weights) != len(self.reference_coeffs):
                print(
                    f"[BiasCalc] Warning: Weight count ({len(weights)}) != Coeff count ({len(self.reference_coeffs)}). Padding with 1.0.")
                diff = len(self.reference_coeffs) - len(weights)
                weights = weights + [1.0] * diff
            self.weights = pd.Series(weights, index=self.reference_coeffs.index)
        else:
            self.weights = weights

        self.config = self._extract_config(self.template, self.type)

        # NEW: Automatically figure out which columns need to be Bool/Category
        # based on the parameter names (e.g., "SEX[T.True]" -> SEX must be bool)
        self.required_dtypes = self._infer_dtypes_from_params(self.reference_coeffs)

    def evaluate_imputations(self, imputed_dfs_list):
        estimates = []
        for df in imputed_dfs_list:
            if df is None or df.empty:
                continue

            # NEW: Transform the variables to match model format
            # This converts imputed floats (0.0, 1.0) back to Bools/Categories
            df_transformed = self._enforce_dtypes(df)

            params = self.refit_on_new_data(df_transformed)
            if params is not None:
                estimates.append(params)

        if not estimates:
            return float('inf')  # Return high penalty if all refits failed

        estimates_df = pd.DataFrame(estimates)

        # 1. Direct Subtraction (Pandas automatically matches column names)
        diff_matrix = estimates_df.sub(self.reference_coeffs, axis=1)

        # 2. Soft Relative Error Calculation
        avg_beta_mag = self.reference_coeffs.abs().mean()
        denominator = self.reference_coeffs.abs() + avg_beta_mag
        soft_rel_error_matrix = diff_matrix.abs().div(denominator, axis=1)

        # 3. Apply Weights & Score
        # (Weights are already aligned Series from __init__)
        weighted_error_matrix = soft_rel_error_matrix.mul(self.weights, axis=1)

        # Double mean: Average over columns (features), then average over rows (imputations)
        final_score = weighted_error_matrix.mean().mean()

        return final_score

    def _infer_dtypes_from_params(self, coeffs):
        """
        Reverse-engineers required column types from coefficient names.
        """
        required_casts = {}
        for p in coeffs.index:
            # Case 1: Boolean (e.g., "SEX[T.True]")
            if "[T.True]" in p:
                col_name = p.split("[")[0]
                required_casts[col_name] = 'bool'
            # Case 2: Categorical (e.g., "RACE[T.Black]")
            elif "[T." in p:
                col_name = p.split("[")[0]
                required_casts[col_name] = 'category'
        return required_casts

    def _enforce_dtypes(self, df):
        """
        Applies the inferred types to the dataframe.
        """
        df = df.copy()
        for col, dtype in self.required_dtypes.items():
            if col in df.columns:
                try:
                    if dtype == 'bool':
                        # Handle 0.0/1.0 floats converting to Bool safe
                        df[col] = df[col].astype(float).astype(bool)
                    else:
                        df[col] = df[col].astype(dtype)
                except Exception:
                    # If casting fails (e.g., messy data), leave it alone to avoid crash
                    pass
        return df

    def _identify_type(self, model):
        if hasattr(model, 'model') and isinstance(model.model, sm_glm.GLM):
            return "statsmodels_glm"
        elif isinstance(model, CoxPHFitter):
            return "lifelines_cox"
        else:
            # Fallback for wrapped statsmodels
            try:
                if isinstance(model.model, sm.genmod.generalized_linear_model.GLM):
                    return "statsmodels_glm"
            except:
                pass
            raise ValueError(f"Unsupported model type: {type(model)}")

    def _extract_coeffs(self, model):
        if self.type == "statsmodels_glm":
            return model.params
        elif self.type == "lifelines_cox":
            return model.params_
        return None

    def _extract_config(self, model, model_type):
        config = {}
        if model_type == "lifelines_cox":
            config['weights_col'] = getattr(model, 'weights_col', None)
            config['cluster_col'] = getattr(model, 'cluster_col', None)
            config['strata'] = getattr(model, 'strata', None)
            config['robust'] = getattr(model, 'robust', False)
            config['formula'] = getattr(model, 'formula', None)
            config['duration_col'] = model.duration_col
            config['event_col'] = model.event_col

        elif model_type == "statsmodels_glm":
            config['family'] = model.model.family
            config['freq_weights'] = getattr(model.model, 'freq_weights', None)
            config['var_weights'] = getattr(model.model, 'var_weights', None)
            config['cov_type'] = getattr(model, 'cov_type', 'nonrobust')
            config['cov_kwds'] = getattr(model, 'cov_kwds', {})

            if hasattr(model.model.data, 'formula'):
                config['formula'] = model.model.data.formula
            elif hasattr(model.model, 'formula'):
                config['formula'] = model.model.formula
            else:
                config['formula'] = None
        return config

    def refit_on_new_data(self, new_df):
        if self.type == "statsmodels_glm":
            if self.config['formula'] is None:
                return None
            try:
                new_mod = smf.glm(
                    formula=self.config['formula'],
                    data=new_df,
                    family=self.config['family'],
                    freq_weights=self.config['freq_weights'],
                    var_weights=self.config['var_weights']
                )
                result = new_mod.fit(cov_type=self.config['cov_type'], cov_kwds=self.config['cov_kwds'])
                return result.params
            except Exception:
                return None

        elif self.type == "lifelines_cox":
            new_cph = copy.deepcopy(self.template)
            try:
                new_cph.fit(
                    new_df,
                    duration_col=self.config['duration_col'],
                    event_col=self.config['event_col'],
                    formula=self.config['formula'],
                    weights_col=self.config['weights_col'],
                    cluster_col=self.config['cluster_col'],
                    strata=self.config['strata'],
                    robust=self.config['robust']
                )
                return new_cph.params_
            except Exception:
                return None