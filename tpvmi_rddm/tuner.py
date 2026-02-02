import optuna
import copy
import numpy as np
import yaml

from tpvmi_rddm.tpvmi_rddm import TPVMI_RDDM
from tpvmi_rddm.utils import process_data


class RDDMTuner:
    def __init__(self, base_config, data_info, param_grid, file_path, n_trials=50, n_folds=5):
        self.base_config = copy.deepcopy(base_config)
        self.data_info = data_info
        self.param_grid = param_grid
        self.file_path = file_path
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.study = None

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
        for param_name, choices in self.param_grid.items():
            chosen_value = trial.suggest_categorical(param_name, choices)
            found = self._update(config, param_name, chosen_value)
            if not found:
                print(f"[Warning] Parameter '{param_name}' not found in base_config. It was skipped.")

        return config

    def objective(self, trial):
        """
        The main objective function optimized by Optuna.
        Runs 5-Fold CV and returns the average validation loss.
        """
        # 1. Generate Config for this specific Trial
        trial_config = self._get_trial_config(trial)
        print(f"\n[Trial {trial.number}] Params: {trial.params}")
        # 2. Prepare Data Indices for CV
        # We assume the file is static, so we process it to find the valid rows
        (proc_data, proc_mask, p1_idx, p2_idx, _, _, _, _) = process_data(self.file_path, self.data_info)
        # Identify Valid Phase 2 Rows (The observed samples to split)
        p2_mask = proc_mask[:, p2_idx.astype(int)]
        valid_rows = np.where(p2_mask.mean(axis=1) > 0.5)[0]
        # Shuffle rows for random CV splits
        np.random.shuffle(valid_rows)
        fold_size = len(valid_rows) // self.n_folds
        fold_scores = []
        fold_epochs = []
        # 3. K-Fold Cross Validation Loop
        for k in range(self.n_folds):
            # A. Define Train/Val indices
            val_start = k * fold_size
            val_end = (k + 1) * fold_size
            val_idx_subset = valid_rows[val_start:val_end]
            train_idx_subset = np.concatenate([valid_rows[:val_start], valid_rows[val_end:]])
            # B. Initialize Model with Trial Config
            model = TPVMI_RDDM(trial_config, self.data_info)
            # C. Fit using ONLY the training subset
            model.fit(self.file_path, train_indices=train_idx_subset, val_indices=val_idx_subset)
            fold_epochs.append(model.best_epoch_)
            # D. Validate on the hold-out set
            val_loss = model._validate(
                model._global_p1[val_idx_subset],
                model._global_p2[val_idx_subset],
                model._global_aux[val_idx_subset],
                bs=trial_config["train"].get("eval_batch_size", 1024)
            )

            fold_scores.append(val_loss)
            # E. Report to Optuna (Pruning)
            trial.report(val_loss, step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        avg_epoch = int(np.mean(fold_epochs))
        trial.set_user_attr("avg_epoch", avg_epoch)
        # Return average loss across all folds
        return np.mean(fold_scores)

    def tune(self, save_best_config=True, config_path="best_config.yaml"):
        """
        Executes the optimization study.
        """
        self.study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        print(f"\n[RDDMTuner] Starting Optimization: {self.n_trials} trials, {self.n_folds}-Fold CV")

        self.study.optimize(self.objective, n_trials=self.n_trials)

        print("\n[RDDMTuner] Optimization Finished.")
        print(f"Best Value (Avg Val Loss): {self.study.best_value:.6f}")

        best_trial = self.study.best_trial
        best_cv_epoch = best_trial.user_attrs.get("avg_epoch", self.base_config["train"]["epochs"])
        final_production_epochs = int(best_cv_epoch * 2)

        print("Best Params:")
        for k, v in self.study.best_params.items():
            print(f"    {k}: {v}")
        print(f"Optimal CV Epoch (Avg): {best_cv_epoch}")
        print(f"Final Production Epochs (2x): {final_production_epochs}")
        # Reconstruct the best config dictionary
        best_config = copy.deepcopy(self.base_config)

        for param_name, chosen_value in self.study.best_params.items():
            self._update(best_config, param_name, chosen_value)
        self._update(best_config, "epochs", final_production_epochs)

        if save_best_config:
            with open(config_path, "w") as f:
                yaml.dump(best_config, f)
            print(f"[RDDMTuner] Best config saved to {config_path}")

        return best_config