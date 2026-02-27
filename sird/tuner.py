import optuna
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import KFold, train_test_split
import gc
import torch
import os
import contextlib
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metric import Regularization, Loss
from sird import SIRD


class BivariateTuner:
    def __init__(self, data, base_config, param_grid, data_info, reg_config, n_splits=1):
        self.base_config = base_config
        self.param_grid = param_grid
        self.data_info = data_info
        self.phase2_vars = data_info.get('phase2_vars', [])
        self.n_splits = n_splits

        self.reg_adapter = Regularization(reg_config)
        self.loss_calc = Loss(data_info)
        cleaned_data = data.replace('nan', np.nan).replace('NaN', np.nan).replace('', np.nan)
        self.data = cleaned_data.dropna().reset_index(drop=True)
        if self.n_splits > 1:
            self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            self.splits = list(self.kf.split(self.data))
            self.fold_histories = {i: [] for i in range(self.n_splits - 1)}
        else:
            self.kf = None
            indices = np.arange(len(self.data))
            self.train_idx, self.val_idx = train_test_split(indices, test_size=0.20, random_state=42)
            self.splits = [(self.train_idx, self.val_idx)]
            self.fold_histories = {}
        self.trial_history = []

    def _get_trial_config(self, trial):
        config = copy.deepcopy(self.base_config)
        flat_params = {}

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

            flat_params[param_name] = chosen_value

            found = self._update(config, param_name, chosen_value)
            if not found:
                print(f"[Warning] Parameter '{param_name}' not found in base_config. It was skipped.")

        return config, flat_params

    def _update(self, d, target_key, target_val):
        for k, v in d.items():
            if k == target_key:
                d[k] = target_val
                return True
            elif isinstance(v, dict):
                if self._update(v, target_key, target_val):
                    return True
        return False

    def objective(self, trial):
        trial_config, flat_params = self._get_trial_config(trial)
        reg_score = self.reg_adapter.compute_score(flat_params)

        fold_total_losses = []
        fold_mse_losses = []
        fold_ce_losses = []
        fold_ed_losses = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.splits):
            masked_data = self.data.copy()
            masked_data.loc[val_idx, self.phase2_vars] = np.nan

            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                model = SIRD(trial_config, self.data_info)
                model.fit(provided_data=masked_data)
                imputed_data = model.impute()

            del model
            gc.collect()
            torch.cuda.empty_cache()

            true_val_data = self.data.iloc[val_idx]
            m_total, m_mse, m_ce, m_ed = [], [], [], []

            for fake_df in imputed_data:
                fake_val_data = fake_df.iloc[val_idx]
                loss_results = self.loss_calc.calculate_loss(true_val_data, fake_val_data)
                m_total.append(loss_results['total_loss'])
                m_mse.append(loss_results.get('weighted_mse', 0))
                m_ce.append(loss_results.get('weighted_ce', 0))
                m_ed.append(loss_results.get('weighted_ed', 0))

            fold_total_losses.append(m_total)
            fold_mse_losses.append(m_mse)
            fold_ce_losses.append(m_ce)
            fold_ed_losses.append(m_ed)

            current_fold_loss = float(np.mean(fold_total_losses))
            if self.n_splits > 1 and fold_idx < (self.n_splits - 1):
                self.fold_histories[fold_idx].append(current_fold_loss)
                if len(self.fold_histories[fold_idx]) >= max(5, int(self.n_trials * 0.15)):
                    threshold = np.nanpercentile(self.fold_histories[fold_idx], 75)
                    if current_fold_loss > threshold:
                        logger = optuna.logging.get_logger("optuna")
                        logger.info(
                            f"Trial {trial.number} pruned at fold {fold_idx}. "
                            f"Loss {current_fold_loss:.4f} is in the worst 25% (Threshold: {threshold:.4f})."
                        )
                        raise optuna.exceptions.TrialPruned()

        arr_total = np.array(fold_total_losses)
        arr_mse = np.array(fold_mse_losses)
        arr_ce = np.array(fold_ce_losses)
        arr_ed = np.array(fold_ed_losses)

        avg_data_loss = float(np.mean(arr_total))
        avg_mse = float(np.mean(arr_mse))
        avg_ce = float(np.mean(arr_ce))
        avg_ed = float(np.mean(arr_ed))

        m_specific_totals = np.round(np.mean(arr_total, axis=0), 4).tolist()
        m_specific_eds = np.round(np.mean(arr_ed, axis=0), 4).tolist()

        history_record = {
            'trial_number': trial.number,
            'avg_total_loss': avg_data_loss,
            'reg_score': reg_score,
            'avg_mse': avg_mse,
            'avg_ce': avg_ce,
            'avg_ed': avg_ed,
            'm_replicate_total_losses': str(m_specific_totals),
            'm_replicate_eds': str(m_specific_eds),
            **flat_params
        }
        self.trial_history.append(history_record)

        logger = optuna.logging.get_logger("optuna")
        logger.info(
            f"Trial {trial.number} Summary -> "
            f"Loss: {avg_data_loss:.4f} (MSE: {avg_mse:.4f}, CE: {avg_ce:.4f}, ED: {avg_ed:.4f}) | "
            f"Reg Score: {reg_score:.4f}"
        )

        return avg_data_loss, reg_score

    def tune(self, n_trials=50, output_csv='optuna_tuning_results.csv'):
        print(f"Starting Bivariate Tuning for {n_trials} trials...")
        self.n_trials = n_trials
        study = optuna.create_study(directions=['minimize', 'maximize'])
        study.optimize(self.objective, n_trials=n_trials)
        history_df = pd.DataFrame(self.trial_history)
        history_df.to_csv(output_csv, index=False)
        print(f"Tuning complete. Results saved to {output_csv}")

        try:
            print("Calculating parameter importance for Data Loss...")
            importance_dict = optuna.importance.get_param_importances(
                study, target=lambda t: t.values[0]
            )
            importance_csv = output_csv.replace('.csv', '_importance.csv')
            pd.Series(importance_dict, name='importance').to_csv(importance_csv, index_label='parameter')
            print(f"Parameter importance successfully saved to {importance_csv}")
        except Exception as e:
            print(f"[Warning] Could not calculate parameter importance. Error: {e}")