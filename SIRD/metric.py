import pandas as pd
from scipy.spatial.distance import cdist


class Loss:
    def __init__(self, data_info, weight_mse=None, weight_ce=None, weight_ed=None):
        # Extract relevant variables targeting phase 2
        self.phase2_vars = data_info.get('phase2_vars', [])
        self.num_vars = data_info.get('num_vars', [])
        self.cat_vars = data_info.get('cat_vars', [])

        self.target_num_vars = [v for v in self.num_vars if v in self.phase2_vars]
        self.target_cat_vars = [v for v in self.cat_vars if v in self.phase2_vars]

        self.weight_mse = weight_mse
        self.weight_ce = weight_ce
        self.weight_ed = weight_ed

        # Flag for auto-balancing if any target weight is missing
        self.auto_balance = (weight_mse is None) or (weight_ce is None) or (weight_ed is None)

        # Track attempts and running sums for stable locking
        self.attempts = 0
        self._sum_raw_mse = 0.0
        self._sum_raw_ce = 0.0
        self._sum_raw_ed = 0.0

    def calculate_loss(self, true_data, fake_data):
        # Ensure targeted numeric variables are strictly evaluated as floats
        cast_dict = {col: float for col in self.target_num_vars}
        true_data = true_data.astype(cast_dict)
        fake_data = fake_data.astype(cast_dict)

        raw_mse = self._compute_mse(true_data, fake_data)
        raw_ce = self._compute_categorical_loss(true_data, fake_data)
        raw_ed = self._compute_energy_distance(true_data, fake_data)

        # Compute and assign inverse weights if auto-balancing is triggered
        if self.auto_balance:
            self.attempts += 1

            # Accumulate raw losses to compute a stable average for the lock
            self._sum_raw_mse += raw_mse
            self._sum_raw_ce += raw_ce
            self._sum_raw_ed += raw_ed

            # Dynamically scale the current batch (Attempts 1 through 5)
            # Using max(..., 1e-8) prevents ZeroDivisionError
            self.weight_mse = 1.0 / max(raw_mse, 1e-8)
            self.weight_ce = 1.0 / max(raw_ce, 1e-8)
            self.weight_ed = 1.0 / max(raw_ed, 1e-8)

            # On the 5th attempt, lock the weights using the average of the first 5 batches
            if self.attempts == 5:
                self.weight_mse = 1.0 / max(self._sum_raw_mse / 5.0, 1e-8)
                self.weight_ce = 1.0 / max(self._sum_raw_ce / 5.0, 1e-8)
                self.weight_ed = 1.0 / max(self._sum_raw_ed / 5.0, 1e-8)
                self.auto_balance = False  # Lock the weights permanently

        norm_mse = raw_mse * self.weight_mse
        norm_ce = raw_ce * self.weight_ce
        norm_ed = raw_ed * self.weight_ed

        n_num = len(self.target_num_vars)
        n_cat = len(self.target_cat_vars)
        n_total = n_num + n_cat

        prop_num = (n_num / n_total) if n_total > 0 else 0.0
        prop_cat = (n_cat / n_total) if n_total > 0 else 0.0

        # Calculate the final combined loss metrics
        final_mse = norm_mse * prop_num
        final_ce = norm_ce * prop_cat
        final_ed = norm_ed

        total_loss = final_mse + final_ce + final_ed

        return {
            'total_loss': total_loss,
            'weighted_mse': final_mse,
            'weighted_ce': final_ce,
            'weighted_ed': final_ed
        }

    def _compute_mse(self, true_data, fake_data):
        # Compute Mean Squared Error strictly on targeted numeric variables
        if not self.target_num_vars:
            return 0.0

        true_num = true_data[self.target_num_vars]
        fake_num = fake_data[self.target_num_vars]

        means = true_num.mean()
        stds = true_num.std().replace(0, 1e-8)

        true_num_scaled = (true_num - means) / stds
        fake_num_scaled = (fake_num - means) / stds

        mse = ((true_num_scaled - fake_num_scaled) ** 2).mean().mean()
        return float(mse)

    def _compute_categorical_loss(self, true_data, fake_data):
        # Compute misclassification error rates for categorical targets
        if not self.target_cat_vars:
            return 0.0

        error_rate_sum = 0.0

        for col in self.target_cat_vars:
            t_col = true_data[col].astype(str)
            f_col = fake_data[col].astype(str)
            error_rate = (t_col != f_col).mean()
            error_rate_sum += error_rate

        return float(error_rate_sum / len(self.target_cat_vars))

    def _compute_energy_distance(self, true_data, fake_data):
        # Compute statistical energy distance between true and generated distributions
        combined = pd.concat([true_data, fake_data[true_data.columns]], axis=0, ignore_index=True)

        if self.num_vars:
            combined[self.num_vars] = (combined[self.num_vars] - combined[self.num_vars].mean()) / combined[
                self.num_vars].std().replace(0, 1e-8)

        if self.cat_vars:
            combined = pd.get_dummies(combined, columns=self.cat_vars, drop_first=False)

        n_true = len(true_data)
        X = combined.iloc[:n_true].to_numpy(dtype=float)
        Y = combined.iloc[n_true:].to_numpy(dtype=float)
        dist_XY = cdist(X, Y, metric='euclidean').mean()
        dist_XX = cdist(X, X, metric='euclidean').mean()
        dist_YY = cdist(Y, Y, metric='euclidean').mean()
        energy_dist = (2 * dist_XY) - dist_XX - dist_YY

        return float(max(energy_dist, 0.0))


class Regularization:
    def __init__(self, reg_config):
        # Configures bounded parameters and direction tracking for regularization mapping
        self.reg_config = reg_config

    def compute_score(self, trial_params):
        # Standardizes heterogeneous model parameters into a unified 0-to-1 score
        aligned_scores = []
        for param_name, config in self.reg_config.items():
            if param_name not in trial_params:
                continue

            val = trial_params[param_name]
            p_min = config['min']
            p_max = config['max']
            higher_is_more = config['higher_is_more_reg']

            val_clipped = max(p_min, min(val, p_max))

            if p_max == p_min:
                norm_val = 0.0
            else:
                norm_val = (val_clipped - p_min) / (p_max - p_min)

            if not higher_is_more:
                norm_val = 1.0 - norm_val
            aligned_scores.append(norm_val)

        if not aligned_scores:
            return 0.0

        final_reg_score = sum(aligned_scores) / len(aligned_scores)

        return float(final_reg_score)