import optuna
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import KFold, train_test_split
from .metric import Regularization, Loss
from .sicg import SICG
import gc
import torch
import os
import contextlib

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
        else:
            self.kf = None
            indices = np.arange(len(self.data))
            self.train_idx, self.val_idx = train_test_split(indices, test_size=0.20, random_state=42)
            self.splits = [(self.train_idx, self.val_idx)]
        self.trial_history = []

    def enforce_wgan_rules(self, config, p):
        # 1. Enforce pack limit (modifies 'p' so the correct batch_size is logged in history)
        p["batch_size"] -= p["batch_size"] % p["pack"]
        config["train"]["batch_size"] = p["batch_size"]

        # 2. Apply Generator scaling overrides
        config["model"]["generator"]["hidden_dim"] = p["hidden_dim"] * p["scale_hidden_dim"]
        config["model"]["generator"]["layers"] = p["layers"] * p["scale_layer"]

        # 3. Apply Learning Rate rules (maps the flat 'lr' to the specific YAML targets)
        lr_g = p["lr"] / p["scale_lr"]
        for optimizer in ["Adam", "SGD"]:
            config["train"][optimizer]["lr_d"] = p["lr"]
            config["train"][optimizer]["lr_g"] = lr_g

        return config

    def _smart_update(self, config_node, p):
        for key, value in config_node.items():
            if isinstance(value, dict):
                self._smart_update(value, p)
            elif key in p:
                config_node[key] = p[key]

    def _get_trial_config(self, trial):
        config = copy.deepcopy(self.base_config)
        flat_params = {}

        for key, params in self.param_grid.items():
            p_type = params[0]
            if p_type == "cat":
                flat_params[key] = trial.suggest_categorical(key, params[1])
            elif p_type == "int":
                flat_params[key] = trial.suggest_int(key, params[1], params[2])
            elif p_type == "float":
                flat_params[key] = trial.suggest_float(key, params[1], params[2])
            elif p_type == "log_float":
                flat_params[key] = trial.suggest_float(key, params[1], params[2], log=True)

        p = flat_params

        self._smart_update(config, p)
        config = self.enforce_wgan_rules(config, p)

        return config, flat_params

    def objective(self, trial):
        trial_config, flat_params = self._get_trial_config(trial)
        reg_score = self.reg_adapter.compute_score(flat_params)

        fold_total_losses = []
        fold_mse_losses = []
        fold_ce_losses = []
        fold_ed_losses = []

        for train_idx, val_idx in self.splits:
            masked_data = self.data.copy()
            masked_data.loc[val_idx, self.phase2_vars] = np.nan

            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                model = SICG(trial_config, self.data_info)
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
        study = optuna.create_study(directions=['minimize', 'maximize'])
        study.optimize(self.objective, n_trials=n_trials)
        history_df = pd.DataFrame(self.trial_history)
        history_df.to_csv(output_csv, index=False)
        print(f"Tuning complete. Results saved to {output_csv}")

        return study, history_df