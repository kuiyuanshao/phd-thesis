import pandas as pd
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import QuantileTransformer


class DataTransformer:
    def __init__(self, data_info, config):
        # Initialize configuration and state variables
        self.data_info = data_info
        self.config = config
        self.df_raw = None
        self.df_transformed = None
        self.valid_rows = None

        # Transformation parameters
        self.mins = {}
        self.shift = {}
        self.num_models = {}
        self.cat_encoders = {}
        self.Q_matrices = {}

        self.pair_map = None
        self.generated_columns = None
        self.norm_type = config['processing'].get('normalization')

        # Metadata for tensorization
        self.q_block_matrix = None

    def fit(self, df):
        # Fit the transformer by preparing categorical strings and numeric parameters
        self.df_raw = df.copy()
        self._clean_categorical_strings(self.df_raw)
        self.df_transformed = self.df_raw.copy()

        if self.config['processing']['log']:
            self._fit_apply_log()
        self._fit_numeric()
        self._fit_categorical()

        return self

    def transform(self, df=None):
        # Apply the learned transformations to the dataset
        if df is None:
            target_df = self.df_transformed
        else:
            target_df = df.copy()
            self._clean_categorical_strings(target_df)
            target_df = self._apply_log(target_df)

        numeric_part = self._transform_numeric(target_df)
        categorical_part = self._transform_categorical(target_df)

        combined_data = {**numeric_part, **categorical_part}
        out_df = pd.DataFrame(combined_data, index=target_df.index)
        self.generated_columns = out_df.columns.tolist()
        return out_df

    def inverse_transform(self, df):
        # Revert transformed data back to its original scale and format
        out_df = df.copy()
        out_df = self._inverse_categorical(out_df)
        out_df = self._inverse_numeric(out_df)
        if self.config['processing']['log']:
            out_df = self._inverse_log(out_df)

        return out_df

    def _clean_categorical_strings(self, df):
        # Strip string columns and handle anomalous '.0' endings for integers stored as floats
        for col in self.data_info['cat_vars']:
            if col in df.columns:
                def clean_val(x):
                    s = str(x).strip()
                    if s.endswith('.0') and s[:-2].isdigit():
                        return s[:-2]
                    return s

                df[col] = df[col].apply(clean_val)

    def _fit_apply_log(self):
        # Calculate minimum shifts to handle zero or negative values before applying log1p
        p1_vars = self.data_info.get('phase1_vars', [])
        p2_vars = self.data_info.get('phase2_vars', [])
        processed_numerics = set()

        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in self.data_info[
                'num_vars'] and p1 in self.df_transformed.columns and p2 in self.df_transformed.columns:
                v1 = pd.to_numeric(self.df_transformed[p1], errors='coerce')
                v2 = pd.to_numeric(self.df_transformed[p2], errors='coerce')
                combined_min = min(v1.min(), v2.min())
                shift = abs(combined_min) + 1.0 if combined_min <= 0 else 0.0

                self.mins[p1] = shift
                self.mins[p2] = shift
                vals1 = np.log1p(v1 + shift)
                vals2 = np.log1p(v2 + shift)
                self.df_transformed[p1] = vals1
                self.df_transformed[p2] = vals2
                processed_numerics.update([p1, p2])

        for col in self.data_info['num_vars']:
            if col not in processed_numerics and col in self.df_transformed.columns:
                vals = pd.to_numeric(self.df_transformed[col], errors='coerce')
                shift = vals.min()
                shift = abs(shift) + 1.0 if shift <= 0 else 0.0
                self.mins[col] = shift
                self.df_transformed[col] = np.log1p(vals + shift)

    def _apply_log(self, df):
        for col, shift in self.mins.items():
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce')
                if col in set(self.data_info["phase1_vars"] + self.data_info['phase2_vars']):
                    logged = np.log1p(vals + shift)
                else:
                    logged = np.log1p(vals + shift)
                df[col] = logged
        return df

    def _inverse_log(self, df):
        for col, shift in self.mins.items():
            if col in df.columns:
                if col in set(self.data_info["phase1_vars"] + self.data_info['phase2_vars']):
                    unlogged = np.expm1(df[col]) - shift
                else:
                    unlogged = np.expm1(df[col]) - shift
                df[col] = unlogged
        return df

    def _fit_numeric(self):
        # Fit statistical models (GMM, Quantile, or Z-score) for scaling numeric variables
        p1_vars = self.data_info.get('phase1_vars', [])
        p2_vars = self.data_info.get('phase2_vars', [])
        self.pair_map = {p2: p1 for p1, p2 in
                         zip(self.data_info.get('phase1_vars', []), self.data_info.get('phase2_vars', []))}

        if self.config['processing']['anchor']:
            for p1, p2 in zip(p1_vars, p2_vars):
                if p1 in self.data_info['num_vars']:
                    vals1 = pd.to_numeric(self.df_transformed[p1], errors='coerce')
                    vals2 = pd.to_numeric(self.df_transformed[p2], errors='coerce')
                    self.shift[p1] = 0
                    self.shift[p2] = np.nanmean(vals1) - np.nanmean(vals2)
                    self.df_transformed[p1] = (vals1 + self.shift[p1])
                    self.df_transformed[p2] = (vals2 + self.shift[p2])
        for col in self.data_info['num_vars']:
            if col not in self.df_transformed.columns: continue
            data = self.df_transformed[col].dropna().values.reshape(-1, 1)

            if col in p1_vars or (col not in p1_vars and col not in p2_vars):
                if self.norm_type == "gmm":
                    max_components = self.config['processing']['gmm'].get('max_components', 1)
                    self.num_models[col] = self._fit_gmm(data, max_components)
                elif self.norm_type == "quantile":
                    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(data), 1000),
                                             random_state=42)
                    qt.fit(data)
                    self.num_models[col] = {
                        'type': 'quantile',
                        'model': qt
                    }
                else:
                    self.num_models[col] = self._fit_zscore(data)

        # Handle phase 2 variables mapping back to their phase 1 anchors if configured
        for col in self.data_info['num_vars']:
            if col in p2_vars:
                data_p2 = self.df_transformed[col].dropna().values.reshape(-1, 1)
                if self.config['processing']['anchor']:
                    anchor_col = self.pair_map.get(col)
                    if anchor_col and anchor_col in self.num_models:
                        anchor_model = self.num_models[anchor_col]
                        if self.norm_type == "gmm":
                            probs = anchor_model['model'].predict_proba(data_p2)
                            modes = probs.argmax(axis=1)

                            n_modes = len(anchor_model['means'])
                            p2_means = np.zeros(n_modes, dtype=np.float32)
                            p2_stds = np.zeros(n_modes, dtype=np.float32)

                            for m in range(n_modes):
                                mask_m = (modes == m)
                                if mask_m.sum() > 0:
                                    p2_means[m] = data_p2[mask_m].mean()
                                    p2_stds[m] = data_p2[mask_m].std()
                                else:
                                    p2_means[m] = data_p2.mean()
                                    p2_stds[m] = data_p2.std()

                            self.num_models[col] = {
                                'type': 'gmm',
                                'model': anchor_model['model'],  # Keep anchor model for predict_proba
                                'means': p2_means,
                                'stds': p2_stds,
                                'global_mean': anchor_model['global_mean'],
                                'global_std': anchor_model['global_std']
                            }
                        else:
                            self.num_models[col] = anchor_model
                    else:
                        self.num_models[col] = self._fit_zscore(data_p2)
                else:
                    if self.norm_type == "quantile":
                        qt = QuantileTransformer(output_distribution='normal', n_quantiles=len(data_p2) // 5,
                                                 random_state=42)
                        qt.fit(data_p2)
                        self.num_models[col] = {
                            'type': 'quantile',
                            'model': qt
                        }
                    else:
                        self.num_models[col] = self._fit_zscore(data_p2)

    def _fit_gmm(self, data, max_k):
        # Select best Gaussian Mixture Model using BIC
        best_gmm, best_bic = None, np.inf
        for k in range(1, max_k + 1):
            try:
                gmm = GaussianMixture(n_components=k, reg_covar=1.0e-3, max_iter=100, random_state=42)
                gmm.fit(data)
                bic = gmm.bic(data)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            except:
                continue

        if best_gmm is None: return self._fit_zscore(data)

        weights = best_gmm.weights_
        means = best_gmm.means_.flatten()
        covars = best_gmm.covariances_.flatten()
        global_mean = np.sum(weights * means)
        global_var = np.sum(weights * (covars + means ** 2)) - global_mean ** 2
        return {
            'type': 'gmm', 'model': best_gmm, 'means': means.astype(np.float32),
            'stds': np.sqrt(covars).astype(np.float32),
            'global_mean': float(global_mean), 'global_std': np.sqrt(global_var)
        }

    def _fit_zscore(self, data):
        return {'type': 'zscore', 'mean': float(np.mean(data)), 'std': float(np.std(data))}

    def _transform_numeric(self, df):
        # Apply scaling based on the fitted distribution model
        collected = {}
        epsilon = 1e-6
        for col in self.data_info['num_vars']:
            if col not in df.columns: continue
            vals = pd.to_numeric(df[col], errors='coerce').values.astype(np.float32)
            mask = ~np.isnan(vals)

            model = self.num_models.get(col)

            if model is None:
                collected[col] = np.zeros(vals.shape, dtype=np.float32)
                continue

            res_val = np.zeros(vals.shape, dtype=np.float32)
            valid_data = vals[mask].reshape(-1, 1)

            if model['type'] == 'zscore':
                if mask.sum() > 0:
                    res_val[mask] = (vals[mask] - model['mean']) / (model['std'] + epsilon)
                collected[col] = res_val
            elif model['type'] == 'quantile':
                if mask.sum() > 0:
                    res_val[mask] = model['model'].transform(valid_data).flatten()
                res_val[~mask] = 0.0
                collected[col] = res_val
            elif model['type'] == 'gmm':
                if mask.sum() > 0:
                    gmm = model['model']
                    probs = gmm.predict_proba(valid_data)
                    modes = probs.argmax(axis=1)
                    res_val[mask] = (vals[mask] - model['global_mean']) / (model['global_std'] + epsilon)
                    collected[col] = res_val
                    n_modes = len(model['means'])
                    mode_matrix = np.eye(n_modes)[modes]
                    for i in range(n_modes):
                        m_res = np.zeros(vals.shape, dtype=np.float32)
                        m_res[mask] = mode_matrix[:, i]
                        collected[f"{col}_mode_{i}"] = m_res
                else:
                    collected[col] = res_val
                    if 'means' in model:
                        for i in range(len(model['means'])):
                            collected[f"{col}_mode_{i}"] = np.zeros(vals.shape, dtype=np.float32)
        return collected

    def _inverse_numeric(self, df):
        epsilon = 1e-6
        p1_vars = self.data_info.get('phase1_vars', [])
        p2_vars = self.data_info.get('phase2_vars', [])
        for col in self.data_info['num_vars']:
            if col not in df.columns: continue
            anchor = self.pair_map.get(col)
            model = self.num_models.get(col)
            model_anchor = self.num_models.get(anchor)
            if not model: continue

            if model['type'] == 'zscore':
                df[col] = df[col] * (model['std'] + epsilon) + model['mean']
            elif model['type'] == 'quantile':
                data = df[col].values.reshape(-1, 1)
                df[col] = model['model'].inverse_transform(data).flatten()
            elif model['type'] == 'gmm':
                # Restore values using robust empirical statistics for anchored components
                n_modes = len(model['means'])
                mode_cols = [f"{col}_mode_{i}" for i in range(n_modes)]
                existing_modes = [c for c in mode_cols if c in df.columns]

                if existing_modes:
                    mode_probs = df[existing_modes].values
                    predicted_modes = np.argmax(mode_probs, axis=1)
                    means = model['means'][predicted_modes]
                    stds = model['stds'][predicted_modes]
                    x_smooth = df[col] * (model['global_std'] + epsilon) + model['global_mean']
                    temp_df = pd.DataFrame({'x_smooth': x_smooth, 'mode': predicted_modes})
                    target_means = model_anchor['means'][predicted_modes]
                    target_stds = model_anchor['stds'][predicted_modes]

                    def get_robust_scale(x):
                        mad = median_abs_deviation(x, scale='normal')
                        return mad if mad > 1e-6 else np.nan

                    empirical_centers = temp_df.groupby('mode')['x_smooth'].transform('median')
                    empirical_scales = temp_df.groupby('mode')['x_smooth'].transform(get_robust_scale)
                    empirical_scales = empirical_scales.fillna(1.0)
                    z_robust = (x_smooth - target_means) / empirical_scales
                    df[col] = target_means + (z_robust * target_stds)
                    df.drop(columns=existing_modes, inplace=True)
            if self.config['processing']['anchor']:
                if col in p2_vars:
                    df[col] = df[col] - self.shift[col]
        return df

    def _fit_categorical(self):
        # Build OneHotEncoders and transition (Q) matrices between paired variables
        p1_vars = self.data_info.get('phase1_vars', [])
        p2_vars = self.data_info.get('phase2_vars', [])

        for col in self.data_info['cat_vars']:
            if col in self.df_transformed.columns:
                series = self.df_transformed[col].dropna()
                mask_valid = ~series.astype(str).isin(["nan", "None", "NaN"])
                vals = series[mask_valid].unique().reshape(-1, 1)
                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                enc.fit(vals)
                self.cat_encoders[col] = enc

        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in self.cat_encoders and p2 in self.cat_encoders:
                self._compute_q_matrix(p1, p2)

        for p1, p2 in zip(p1_vars, p2_vars):
            anchor = self.pair_map.get(p2, p1)
            model = self.num_models.get(anchor)
            if model and model[
                'type'] == 'gmm' and p1 in self.df_transformed.columns and p2 in self.df_transformed.columns:
                self._compute_q_matrix_gmm(p1, p2, model)

        print("\n[DEBUG] Checking Category Consistency for Pairs:")
        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in self.cat_encoders and p2 in self.cat_encoders:
                cats1 = self.cat_encoders[p1].categories_[0]
                cats2 = self.cat_encoders[p2].categories_[0]
                if len(cats1) != len(cats2) or not np.array_equal(cats1, cats2):
                    print(f"!!! MISMATCH FOUND in Pair: {p1} (Proxy) vs {p2} (True)")
                    print(f"    {p1} count: {len(cats1)}")
                    print(f"    {p2} count: {len(cats2)}")
        print("[DEBUG] Check Complete.\n")

    def _compute_q_matrix(self, p1, p2):
        valid = self.df_transformed[p1].notna() & self.df_transformed[p2].notna()
        d1 = self.df_transformed.loc[valid, p1].values.reshape(-1, 1)
        d2 = self.df_transformed.loc[valid, p2].values.reshape(-1, 1)
        idx1 = self.cat_encoders[p1].transform(d1).argmax(axis=1)
        idx2 = self.cat_encoders[p2].transform(d2).argmax(axis=1)
        K1, K2 = len(self.cat_encoders[p1].categories_[0]), len(self.cat_encoders[p2].categories_[0])
        Q = np.zeros((K1, K2))
        np.add.at(Q, (idx1, idx2), 1)
        self.Q_matrices[p2] = torch.tensor((Q + 1) / (Q + 1).sum(axis=1, keepdims=True), dtype=torch.float32)

    def _compute_q_matrix_gmm(self, p1, p2, model):
        valid = self.df_transformed[p1].notna() & self.df_transformed[p2].notna()
        d1 = self.df_transformed.loc[valid, p1].values.reshape(-1, 1)
        d2 = self.df_transformed.loc[valid, p2].values.reshape(-1, 1)
        if len(d1) == 0: return
        idx1 = model['model'].predict(d1)
        idx2 = model['model'].predict(d2)
        K = model['means'].shape[0]
        Q = np.zeros((K, K))
        np.add.at(Q, (idx1, idx2), 1)
        self.Q_matrices[f"{p2}_mode"] = torch.tensor((Q + 1) / (Q + 1).sum(axis=1, keepdims=True), dtype=torch.float32)

    def _transform_categorical(self, df):
        # Explode categorical variables into distinct integer-encoded columns
        collected = {}
        for col, enc in self.cat_encoders.items():
            if col not in df.columns: continue
            cats = enc.categories_[0]
            mask = df[col].notna().values
            safe_vals = df[col].fillna(cats[0]).values.reshape(-1, 1)
            encoded = enc.transform(safe_vals)
            encoded[~mask] = 0
            for i, c in enumerate(cats):
                collected[f"{col}_{c}"] = encoded[:, i].astype(np.float32)
        return collected

    def _inverse_categorical(self, df):
        for col, enc in self.cat_encoders.items():
            cats = enc.categories_[0]
            col_names = [f"{col}_{c}" for c in cats]
            present = [c for c in col_names if c in df.columns]
            if not present: continue
            indices = np.argmax(df[present].values, axis=1)
            df[col] = enc.categories_[0][indices]
            df.drop(columns=present, inplace=True)
        return df

    def get_sird_schema(self):
        # Generate schema mapping for the fully transformed tensor layouts
        if self.generated_columns is None: raise RuntimeError("Run transform first.")
        schema = []
        df_cols = self.generated_columns
        p1_vars = self.data_info.get('phase1_vars', [])
        p2_vars = self.data_info.get('phase2_vars', [])

        def get_type_pair(name):
            if name in p1_vars: return 'p1', p2_vars[p1_vars.index(name)]
            if name in p2_vars: return 'p2', p1_vars[p2_vars.index(name)]
            return 'aux', None

        for col in self.data_info['num_vars']:
            if col in df_cols:
                idx = df_cols.index(col)
                v_role, pair = get_type_pair(col)
                schema.append({'name': col, 'type': f'numeric_{v_role}', 'num_classes': 1,
                               'start_idx': idx, 'end_idx': idx + 1, 'pair_partner': pair})
            mode_cols = sorted([c for c in df_cols if c.startswith(f"{col}_mode_")])
            if mode_cols:
                idxs = [df_cols.index(c) for c in mode_cols]
                v_role, pair = get_type_pair(col)
                schema.append({'name': f"{col}_mode", 'type': f'categorical_{v_role}',
                               'num_classes': len(idxs), 'start_idx': idxs[0], 'end_idx': idxs[-1] + 1,
                               'pair_partner': f"{pair}_mode" if pair else None})

        for col in self.data_info['cat_vars']:
            enc = self.cat_encoders.get(col)
            if not enc: continue
            cats = enc.categories_[0]
            idxs = sorted([df_cols.index(f"{col}_{c}") for c in cats if f"{col}_{c}" in df_cols])
            if idxs:
                v_role, pair = get_type_pair(col)
                schema.append({'name': col, 'type': f'categorical_{v_role}',
                               'num_classes': len(idxs), 'start_idx': idxs[0], 'end_idx': idxs[-1] + 1,
                               'pair_partner': pair})
        return schema