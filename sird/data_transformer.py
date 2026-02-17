import pandas as pd
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from scipy.stats import median_abs_deviation
from scipy.special import softmax

class DataTransformer:
    def __init__(self, data_info, config):
        self.data_info = data_info
        self.config = config

        self.df_raw = None
        self.df_transformed = None

        # Statistics & Models
        self.mins = {}  # For Log shift
        self.num_models = {}  # GMM / Z-score stats
        self.cat_encoders = {}  # OneHotEncoders (Keys: col_name)
        self.Q_matrices = {}  # Confusion Matrices (Keys: p2_col_name)

        self.generated_columns = None
        self.norm_type = config['processing'].get('normalization', 'gmm')

    # =========================================================================
    # CORE PIPELINE
    # =========================================================================

    def fit(self, df):
        """
        Fits all transformations on the provided dataframe.
        Order: Log -> GMM (Anchored) -> Categorical (Q-Matrix + Encoders).
        """
        self.df_raw = df.copy()

        # 0. Robust String Cleaning for Categorical (Pre-pass)
        self._clean_categorical_strings(self.df_raw)

        self.df_transformed = self.df_raw.copy()

        # 1. Log Transform (Jointly for Pairs, Independent for Aux)
        if self.config['processing'].get('log', False):
            self._fit_apply_log()

        # 2. Numeric Normalization (Proxy-Anchored GMM - Two Pass)
        self._fit_numeric()

        # 3. Categorical (Encoders + Q Matrix)
        # FIX: Now also computes Q-Matrices for GMM Modes
        self._fit_categorical()

        return self

    def transform(self, df=None):
        """
        Transforms data using fitted stats.
        If df is None, transforms the fitted internal df.
        """
        if df is None:
            target_df = self.df_transformed
        else:
            target_df = df.copy()
            self._clean_categorical_strings(target_df)
            if self.config['processing'].get('log', False):
                target_df = self._apply_log(target_df)

        numeric_part = self._transform_numeric(target_df)
        categorical_part = self._transform_categorical(target_df)

        combined_data = {**numeric_part, **categorical_part}
        out_df = pd.DataFrame(combined_data, index=target_df.index)
        self.generated_columns = out_df.columns.tolist()
        return out_df

    def inverse_transform(self, df):
        """
        Reverses transformation to original scale/categories.
        """
        out_df = df.copy()
        out_df = self._inverse_categorical(out_df)
        out_df = self._inverse_numeric(out_df)

        if self.config['processing'].get('log', False):
            out_df = self._inverse_log(out_df)

        return out_df

    # =========================================================================
    # 0. PRE-PROCESSING HELPERS
    # =========================================================================

    def _clean_categorical_strings(self, df):
        """
        Robustly converts value to string for categorical comparison.
        Handles '1.0' vs '1' mismatch by converting integer-like floats to ints.
        """
        for col in self.data_info['cat_vars']:
            if col in df.columns:
                def clean_val(x):
                    s = str(x).strip()
                    if s.endswith('.0') and s[:-2].isdigit():
                        return s[:-2]
                    return s

                df[col] = df[col].apply(clean_val)

    # =========================================================================
    # 1. LOG TRANSFORM (Joint & Independent)
    # =========================================================================

    def _fit_apply_log(self):
        # A. Pairs: Calculate Joint Shift
        p1_vars = self.data_info.get('phase1_vars', [])
        p2_vars = self.data_info.get('phase2_vars', [])
        processed_numerics = set()

        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in self.data_info[
                'num_vars'] and p1 in self.df_transformed.columns and p2 in self.df_transformed.columns:
                v1 = pd.to_numeric(self.df_transformed[p1], errors='coerce').fillna(0)
                v2 = pd.to_numeric(self.df_transformed[p2], errors='coerce').fillna(0)

                # Joint Min for safety
                combined_min = min(v1.min(), v2.min())
                shift = abs(combined_min) + 1.0 if combined_min <= 0 else 0.0

                self.mins[p1] = shift
                self.mins[p2] = shift

                self.df_transformed[p1] = np.log1p(v1 + shift)
                self.df_transformed[p2] = np.log1p(v2 + shift)
                processed_numerics.update([p1, p2])

        # B. Aux/Unpaired: Independent Shift
        for col in self.data_info['num_vars']:
            if col not in processed_numerics and col in self.df_transformed.columns:
                vals = pd.to_numeric(self.df_transformed[col], errors='coerce').fillna(0)
                shift = vals.min()
                shift = abs(shift) + 1.0 if shift <= 0 else 0.0

                self.mins[col] = shift
                self.df_transformed[col] = np.log1p(vals + shift)

    def _apply_log(self, df):
        for col, shift in self.mins.items():
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[col] = np.log1p(vals + shift)
        return df

    def _inverse_log(self, df):
        for col, shift in self.mins.items():
            if col in df.columns:
                df[col] = np.expm1(df[col]) - shift
        return df

    # =========================================================================
    # 2. NUMERIC STRATEGIES (Proxy-Anchored)
    # =========================================================================

    def _fit_numeric(self):
        p1_vars = set(self.data_info.get('phase1_vars', []))
        p2_vars = set(self.data_info.get('phase2_vars', []))

        # Map P2 -> P1 for easy lookup
        # We assume strict ordering in lists matches
        p1_list = self.data_info.get('phase1_vars', [])
        p2_list = self.data_info.get('phase2_vars', [])
        pair_map = {p2: p1 for p1, p2 in zip(p1_list, p2_list)}

        max_components = self.config['processing']['gmm'].get('max_components', 5)

        # PASS 1: Fit Anchors (Phase 1) and Aux Variables
        for col in self.data_info['num_vars']:
            if col not in self.df_transformed.columns: continue

            data = self.df_transformed[col].dropna().values.reshape(-1, 1)

            # Fit Anchor or Aux
            if col in p1_vars or (col not in p1_vars and col not in p2_vars):
                if col in p1_vars and max_components > 1:
                    self.num_models[col] = self._fit_gmm(data, max_components)
                else:
                    self.num_models[col] = self._fit_zscore(data)

        # PASS 2: Link Phase 2 Variables to Anchors
        for col in self.data_info['num_vars']:
            if col not in self.df_transformed.columns: continue

            if col in p2_vars:
                anchor_col = pair_map.get(col)
                if anchor_col and anchor_col in self.num_models:
                    # P2 uses P1's model implicitly.
                    # We do not store a separate model for P2 to enforce shared space.
                    pass
                else:
                    # Fallback if pairing is broken or P1 failed
                    data = self.df_transformed[col].dropna().values.reshape(-1, 1)
                    self.num_models[col] = self._fit_zscore(data)

    def _fit_gmm(self, data, max_k):
        best_gmm = None
        best_bic = np.inf

        for k in range(1, max_k + 1):
            try:
                gmm = GaussianMixture(n_components=k, reg_covar=1e-3, max_iter=100, random_state=42)
                gmm.fit(data)
                bic = gmm.bic(data)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            except:
                continue

        if best_gmm is None:
            return self._fit_zscore(data)

        weights = best_gmm.weights_
        means = best_gmm.means_.flatten()
        covars = best_gmm.covariances_.flatten()

        global_mean = np.sum(weights * means)
        # Law of total variance
        global_var = np.sum(weights * (covars + means ** 2)) - global_mean ** 2
        global_std = np.sqrt(global_var)

        return {
            'type': 'gmm',
            'model': best_gmm,
            'means': means.astype(np.float32),
            'stds': np.sqrt(covars).astype(np.float32),
            'global_mean': float(global_mean),   # <--- Add this
            'global_std': float(global_std)
        }

    def _fit_zscore(self, data):
        return {
            'type': 'zscore',
            'mean': float(np.mean(data)),
            'std': float(np.std(data))
        }

    def _transform_numeric(self, df):
        collected = {}
        epsilon = 1e-6

        p1_list = self.data_info.get('phase1_vars', [])
        p2_list = self.data_info.get('phase2_vars', [])
        pair_map = {p2: p1 for p1, p2 in zip(p1_list, p2_list)}

        for col in self.data_info['num_vars']:
            if col not in df.columns: continue

            vals = pd.to_numeric(df[col], errors='coerce').values.astype(np.float32)
            mask = ~np.isnan(vals)

            # Identify model source
            if col in pair_map:
                anchor = pair_map[col]
                model = self.num_models.get(anchor)
                # Fallback to self if anchor missing
                if not model: model = self.num_models.get(col)
            else:
                model = self.num_models.get(col)

            if model is None:
                collected[col] = np.zeros(vals.shape, dtype=np.float32)
                continue

            res_val = np.zeros(vals.shape, dtype=np.float32)
            valid_data = vals[mask].reshape(-1, 1)

            # Transform
            if model['type'] == 'zscore':
                if mask.sum() > 0:
                    res_val[mask] = (vals[mask] - model['mean']) / (model['std'] + epsilon)
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
                        mode_col_name = f"{col}_mode_{i}"
                        m_res = np.zeros(vals.shape, dtype=np.float32)
                        m_res[mask] = mode_matrix[:, i]
                        collected[mode_col_name] = m_res
                else:
                    collected[col] = res_val
                    # Ensure mode columns exist even if model fell back or empty
                    if 'means' in model:  # Just in case
                        n_modes = len(model['means'])
                        for i in range(n_modes):
                            collected[f"{col}_mode_{i}"] = np.zeros(vals.shape, dtype=np.float32)

        return collected

    def _inverse_numeric(self, df):
        epsilon = 1e-6
        p1_list = self.data_info.get('phase1_vars', [])
        p2_list = self.data_info.get('phase2_vars', [])
        pair_map = {p2: p1 for p1, p2 in zip(p1_list, p2_list)}

        for col in self.data_info['num_vars']:
            if col not in df.columns: continue

            # Identify model source
            if col in pair_map:
                anchor = pair_map[col]
                model = self.num_models.get(anchor)
                if not model: model = self.num_models.get(col)
            else:
                model = self.num_models.get(col)

            if model is None: continue

            if model['type'] == 'zscore':
                df[col] = df[col] * (model['std'] + epsilon) + model['mean']

            elif model['type'] == 'gmm':
                # Reconstruct based on predicted modes
                n_modes = len(model['means'])
                mode_cols = [f"{col}_mode_{i}" for i in range(n_modes)]

                # Check if mode columns exist in df
                existing_modes = [c for c in mode_cols if c in df.columns]

                if existing_modes:
                    mode_probs = df[existing_modes].values
                    predicted_modes = np.argmax(mode_probs, axis=1)

                    means = model['means'][predicted_modes]
                    stds = model['stds'][predicted_modes]
                    residual_local = df[col] * (model['global_std'] + epsilon) + model['global_mean'] - means
                    temp_df = pd.DataFrame({
                        'residual': residual_local,
                        'mode': predicted_modes
                    })
                    def calculate_robust_sigma(x):
                        if len(x) < 2: return 0.0
                        return median_abs_deviation(x, scale='normal')

                    empirical_stds = temp_df.groupby('mode')['residual'].transform(calculate_robust_sigma)

                    target_stds_series = pd.Series(stds, index=temp_df.index)
                    empirical_stds = empirical_stds.replace(0, np.nan).fillna(target_stds_series)

                    scaling_factors = target_stds_series / empirical_stds
                    df[col] = means + (residual_local * scaling_factors)
                    df.drop(columns=existing_modes, inplace=True)

        return df

    # =========================================================================
    # 3. CATEGORICAL (Independent Encoders & Q Matrix)
    # =========================================================================

    def _fit_categorical(self):
        p1_vars = self.data_info.get('phase1_vars', [])
        p2_vars = self.data_info.get('phase2_vars', [])
        pair_map = {p2: p1 for p1, p2 in zip(p1_vars, p2_vars)}

        # 1. Fit Independent Encoders for ALL categorical columns
        for col in self.data_info['cat_vars']:
            if col in self.df_transformed.columns:
                # Get unique values
                series = self.df_transformed[col].dropna()

                # 2. STRICTLY filter out string "nan" / "None" ghosts
                # (This ensures they are never learned as a valid category 'nan')
                mask_valid = ~series.astype(str).isin(["nan", "None", "NaN"])
                vals = series[mask_valid].unique().reshape(-1, 1)
                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                enc.fit(vals)
                self.cat_encoders[col] = enc

        # 2. Compute Q Matrix for EXPLICIT Categorical Pairs
        for p1, p2 in zip(p1_vars, p2_vars):
            is_cat_pair = (p1 in self.cat_encoders) and (p2 in self.cat_encoders)

            if is_cat_pair:
                # Get valid paired rows
                valid_mask = self.df_transformed[p1].notna() & self.df_transformed[p2].notna()
                d_p1 = self.df_transformed.loc[valid_mask, p1].values.reshape(-1, 1)
                d_p2 = self.df_transformed.loc[valid_mask, p2].values.reshape(-1, 1)

                enc1 = self.cat_encoders[p1]
                enc2 = self.cat_encoders[p2]

                # Transform to indices
                idx1 = enc1.transform(d_p1).argmax(axis=1)
                idx2 = enc2.transform(d_p2).argmax(axis=1)

                K1 = len(enc1.categories_[0])
                K2 = len(enc2.categories_[0])

                Q_raw = np.zeros((K1, K2))
                np.add.at(Q_raw, (idx1, idx2), 1)

                # Laplace Smoothing & Normalization
                cm_smooth = Q_raw + 1
                row_sums = cm_smooth.sum(axis=1, keepdims=True)
                Q = cm_smooth / row_sums

                self.Q_matrices[p2] = torch.tensor(Q, dtype=torch.float32)

        # 3. FIX: Compute Q Matrix for GMM MODES (Generated Categoricals)
        # We perform the same Q-Matrix logic but on the predicted GMM components
        for p1, p2 in zip(p1_vars, p2_vars):
            # Check if this pair is numeric and managed by a GMM
            anchor_col = pair_map.get(p2)  # Should match p1
            if not anchor_col: anchor_col = p1

            model = self.num_models.get(anchor_col)

            # Ensure model exists and is GMM type
            if model and model[
                'type'] == 'gmm' and p1 in self.df_transformed.columns and p2 in self.df_transformed.columns:

                # Get valid overlapping data
                valid_mask = self.df_transformed[p1].notna() & self.df_transformed[p2].notna()
                d_p1 = self.df_transformed.loc[valid_mask, p1].values.reshape(-1, 1)
                d_p2 = self.df_transformed.loc[valid_mask, p2].values.reshape(-1, 1)

                if len(d_p1) == 0:
                    continue

                gmm = model['model']
                # Predict the mode (component index) for both proxy and true
                # This puts them in the same latent space defined by the anchor GMM
                idx1 = gmm.predict(d_p1)
                idx2 = gmm.predict(d_p2)

                # Number of components
                K = model['means'].shape[0]

                # Compute Q (Confusion Matrix) for Modes
                # Shape (K, K). Rows=ProxyMode, Cols=TrueMode
                Q_raw = np.zeros((K, K))
                np.add.at(Q_raw, (idx1, idx2), 1)

                cm_smooth = Q_raw + 1
                row_sums = cm_smooth.sum(axis=1, keepdims=True)
                Q = cm_smooth / row_sums

                # Register this Q-matrix under the name generated for the Target mode
                # schema name is "{p2}_mode"
                self.Q_matrices[f"{p2}_mode"] = torch.tensor(Q, dtype=torch.float32)
                # print(f"[DEBUG] Generated Q-Matrix for GMM pair: {p1} -> {p2} (K={K})")

        print("\n[DEBUG] Checking Category Consistency for Pairs:")
        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in self.cat_encoders and p2 in self.cat_encoders:
                cats1 = self.cat_encoders[p1].categories_[0]
                cats2 = self.cat_encoders[p2].categories_[0]

                # Check for length mismatch or content mismatch
                if len(cats1) != len(cats2) or not np.array_equal(cats1, cats2):
                    print(f"!!! MISMATCH FOUND in Pair: {p1} (Proxy) vs {p2} (True)")
                    print(f"    {p1} count: {len(cats1)}")
                    print(f"    {p2} count: {len(cats2)}")

                    # Find specific differences
                    set1, set2 = set(cats1), set(cats2)
                    only_in_p1 = set1 - set2
                    only_in_p2 = set2 - set1

                    if only_in_p1: print(f"    In {p1} only: {only_in_p1}")
                    if only_in_p2: print(f"    In {p2} only: {only_in_p2}")
        print("[DEBUG] Check Complete.\n")

    def _transform_categorical(self, df):
        collected = {}
        for col, enc in self.cat_encoders.items():
            if col not in df.columns: continue

            cats = enc.categories_[0]
            col_names = [f"{col}_{c}" for c in cats]

            mask = df[col].notna().values
            # Placeholder for NaN to avoid error during transform (masked later)
            safe_vals = df[col].fillna(cats[0]).values.reshape(-1, 1)

            encoded = enc.transform(safe_vals)
            encoded[~mask] = 0

            for i, name in enumerate(col_names):
                collected[name] = encoded[:, i].astype(np.float32)

        return collected

    def _inverse_categorical(self, df):
        for col, enc in self.cat_encoders.items():
            cats = enc.categories_[0]
            col_names = [f"{col}_{c}" for c in cats]
            present_cols = [c for c in col_names if c in df.columns]

            if not present_cols: continue

            probs = df[present_cols].values
            indices = np.argmax(probs, axis=1)
            decoded = enc.categories_[0][indices]

            df[col] = decoded
            df.drop(columns=present_cols, inplace=True)

        return df

    def get_sird_schema(self):
        if self.generated_columns is None:
            raise RuntimeError("Run transform first.")

        schema = []
        df_cols = self.generated_columns

        p1_vars = self.data_info.get('phase1_vars', [])
        p2_vars = self.data_info.get('phase2_vars', [])

        # Helper to classify variable
        def get_type_pair(name):
            if name in p1_vars: return 'p1', p2_vars[p1_vars.index(name)]
            if name in p2_vars: return 'p2', p1_vars[p2_vars.index(name)]
            return 'aux', None

        # 1. Numerics & Their Modes
        for col in self.data_info['num_vars']:
            if col in df_cols:
                idx = df_cols.index(col)
                v_role, pair = get_type_pair(col)

                # Add the Numeric Variable itself
                schema.append({
                    'name': col, 'type': f'numeric_{v_role}', 'num_classes': 1,
                    'start_idx': idx, 'end_idx': idx + 1, 'pair_partner': pair
                })

            # Check for associated Mode columns (GMM components)
            mode_cols = [c for c in df_cols if c.startswith(f"{col}_mode_")]
            if mode_cols:
                # Group all one-hot mode columns into a single variable
                indices = sorted([df_cols.index(c) for c in mode_cols])
                start, end = indices[0], indices[-1] + 1

                v_role, pair = get_type_pair(col)
                m_type = f'categorical_{v_role}'

                # If the parent numeric has a partner (e.g. 'bp_p1'),
                # then the mode variable ('bp_p2_mode') MUST partner with 'bp_p1_mode'.
                if pair:
                    mode_partner = f"{pair}_mode"
                else:
                    mode_partner = None

                schema.append({
                    'name': f"{col}_mode", 'type': m_type, 'num_classes': len(indices),
                    'start_idx': start, 'end_idx': end, 'pair_partner': mode_partner
                })

        # 2. Categoricals
        for col in self.data_info['cat_vars']:
            enc = self.cat_encoders.get(col)
            if not enc: continue

            cats = enc.categories_[0]
            col_names = [f"{col}_{c}" for c in cats]
            indices = [df_cols.index(c) for c in col_names if c in df_cols]

            if indices:
                indices.sort()
                start, end = indices[0], indices[-1] + 1
                v_role, pair = get_type_pair(col)

                schema.append({
                    'name': col, 'type': f'categorical_{v_role}', 'num_classes': len(indices),
                    'start_idx': start, 'end_idx': end, 'pair_partner': pair
                })

        return schema
