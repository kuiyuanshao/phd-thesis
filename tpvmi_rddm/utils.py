import pandas as pd
import numpy as np
import statsmodels.genmod.generalized_linear_model as sm_glm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
import copy

def process_data(filepath, data_info):
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    p1_vars = data_info.get('phase1_vars', [])
    p2_vars = data_info.get('phase2_vars', [])
    cat_vars_set = set(data_info.get('cat_vars', []))

    if len(p1_vars) != len(p2_vars):
        raise ValueError("Phase 1 and Phase 2 variable lists must be of equal length.")

    reserved_vars = set(p1_vars) | set(p2_vars)
    all_vars = df.columns.tolist()
    aux_vars = sorted([c for c in all_vars if c not in reserved_vars])

    processed_data_list = []
    processed_mask_list = []
    variable_schema = []
    p1_indices = []
    p2_indices = []
    current_col_idx = 0
    normalization_stats = {}

    weight_idx = None

    print(f"\n[Data Processing] Targets: {len(p2_vars)} pairs | Context: {len(aux_vars)} aux variables")

    # ==========================================
    # BLOCK A: PHASE 1 & PHASE 2 (Target Pairs)
    # ==========================================
    for p1_name, p2_name in zip(p1_vars, p2_vars):
        if p1_name not in df.columns or p2_name not in df.columns:
            raise ValueError(f"Missing pair: {p1_name} or {p2_name}")

        p1_raw = df[p1_name].values
        p2_raw = df[p2_name].values
        m1 = (~df[p1_name].isna()).values.astype(float)
        m2 = (~df[p2_name].isna()).values.astype(float)

        is_categorical = (p2_name in cat_vars_set) or (p1_name in cat_vars_set)

        if is_categorical:
            # --- CATEGORICAL ---
            u1 = pd.unique(p1_raw[pd.notna(p1_raw)])
            u2 = pd.unique(p2_raw[pd.notna(p2_raw)])
            unique_set = set(u1) | set(u2)
            master_categories = sorted(list(unique_set))
            cat_to_int = {val: i for i, val in enumerate(master_categories)}
            K = max(len(master_categories), 1)

            d1 = np.array([cat_to_int.get(x, 0) for x in p1_raw]).reshape(-1, 1)
            d2 = np.array([cat_to_int.get(x, 0) for x in p2_raw]).reshape(-1, 1)

            # [CHANGE]: Append Phase-1 to Schema
            variable_schema.append({'name': p2_name, 'type': 'categorical', 'num_classes': K})

            normalization_stats[p2_name] = {'type': 'categorical', 'categories': np.array(master_categories)}
            normalization_stats[p1_name] = normalization_stats[p2_name]
        else:
            # --- NUMERIC LOG TRANSFORMATION ---
            v1_float = p1_raw.astype(float)
            v2_float = p2_raw.astype(float)

            # Combined shift logic to ensure strictly positive values for log
            combined_min = np.nanmin(np.concatenate([v1_float, v2_float]))
            shift = 0.0
            if combined_min <= 0:
                shift = abs(combined_min) + 1.0

            # Step 1: Log-transform
            v1_log = np.log1p(v1_float + shift)
            v2_log = np.log1p(v2_float + shift)

            # Step 2: Combined Normalization Stats (P1 + P2)
            # We use all available observed points from both columns
            valid_p1_log = v1_log[m1 == 1]
            valid_p2_log = v2_log[m2 == 1]

            mu, sigma = np.mean(valid_p2_log), np.std(valid_p2_log)
            if sigma < 1e-6: sigma = 1.0

            # Standardize both using the combined stats
            d1 = (v1_log - mu) / sigma
            d2 = (v2_log - mu) / sigma

            d1 = np.nan_to_num(d1, nan=0.0).reshape(-1, 1)
            d2 = np.nan_to_num(d2, nan=0.0).reshape(-1, 1)

            # [CHANGE]: Append Phase-1 to Schema
            variable_schema.append({'name': p2_name, 'type': 'numeric'})

            normalization_stats[p2_name] = {'type': 'numeric', 'mu': mu, 'sigma': sigma, 'shift': shift}
            normalization_stats[p1_name] = normalization_stats[p2_name]

        processed_data_list.extend([d1, d2])
        processed_mask_list.extend([m1.reshape(-1, 1), m2.reshape(-1, 1)])
        p1_indices.append(current_col_idx)
        p2_indices.append(current_col_idx + 1)
        current_col_idx += 2

    # ==========================================
    # BLOCK B: AUXILIARY CONTEXT
    # ==========================================
    for aux_name in aux_vars:
        raw_vals = df[aux_name].values
        mask = (~df[aux_name].isna()).values.astype(float).reshape(-1, 1)
        is_categorical = aux_name in cat_vars_set

        if is_categorical:
            u_vals = pd.unique(raw_vals[pd.notna(raw_vals)])
            master_categories = sorted(list(u_vals))
            cat_to_int = {val: i for i, val in enumerate(master_categories)}
            K = max(len(master_categories), 1)
            d_aux = np.array([cat_to_int.get(x, 0) for x in raw_vals]).reshape(-1, 1)

            variable_schema.append({'name': aux_name, 'type': 'categorical_aux', 'num_classes': K})
            normalization_stats[aux_name] = {'type': 'categorical', 'categories': np.array(master_categories)}
        else:
            v_float = raw_vals.astype(float)
            v_min = np.nanmin(v_float)
            shift = 0.0
            if v_min <= 0:
                shift = abs(v_min) + 1.0

            v_log = np.log1p(v_float + shift)
            valid_log = v_log[mask.flatten() == 1]

            if len(valid_log) > 0:
                mu, sigma = np.mean(valid_log), np.std(valid_log)
                if sigma < 1e-6: sigma = 1.0
            else:
                mu, sigma = 0.0, 1.0

            d_aux = (v_log - mu) / sigma
            d_aux = np.nan_to_num(d_aux, nan=0.0).reshape(-1, 1)

            variable_schema.append({'name': aux_name, 'type': 'numeric_aux'})
            normalization_stats[aux_name] = {'type': 'numeric', 'mu': mu, 'sigma': sigma, 'shift': shift}

        processed_data_list.append(d_aux)
        processed_mask_list.append(mask)
        current_col_idx += 1

    final_data = np.hstack(processed_data_list)
    final_mask = np.hstack(processed_mask_list)

    return (final_data, final_mask, np.array(p1_indices), np.array(p2_indices),
            weight_idx, variable_schema, normalization_stats, df)


def inverse_transform_data(generated_data, normalization_stats, data_info):
    """
    Reverses normalization AND log-transformation for the GENERATED data.
    """
    reconstructed_df = pd.DataFrame()
    p2_vars = data_info['phase2_vars']

    for i, p2_name in enumerate(p2_vars):
        stats = normalization_stats[p2_name]
        col_data = generated_data[:, i]

        if stats['type'] == 'numeric':
            mu, sigma, shift = stats['mu'], stats['sigma'], stats['shift']
            # 1. Reverse Z-score
            val_log = col_data * sigma + mu
            # 2. Reverse Log (Exponential)
            final_val = np.expm1(val_log) - shift
            reconstructed_df[p2_name] = final_val
        else:
            categories = stats['categories']
            indices = np.clip(np.round(col_data), 0, len(categories) - 1).astype(int)
            if len(categories) > 0:
                reconstructed_df[p2_name] = categories[indices]
            else:
                reconstructed_df[p2_name] = indices

    return reconstructed_df


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