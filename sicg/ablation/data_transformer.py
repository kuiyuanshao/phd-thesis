import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder


class DataTransformer:
    def __init__(self, data_info, config):
        """
        config structure:
        {
            'model': {'residual_modeling': bool},
            'processing': {
                'normalization': 'gmm' or 'min_max',
                'gmm': {'max_components': 5, 'epsilon': 1e-6}
            }
        }
        """
        self.data_info = data_info
        self.config = config

        # Internal State Storage
        self.df_raw = None  # Original input
        self.df_transformed = None  # Working copy (Log/Residuals applied here)

        # Model Parameters
        self.mins = {}  # Log shifts
        self.num_models = {}  # Normalization params (GMM or MinMax)
        self.cat_encoders = {}  # Categorical Encoders
        self.mode_encoders = {}  # GMM Mode Encoders

        self.generated_columns = None
        self.norm_type = config['processing'].get('normalization', 'gmm')

    # =========================================================================
    # CORE PIPELINE METHODS
    # =========================================================================

    def fit(self, df):
        """
        1. Stores df.
        2. Applies Log Transformation & Residuals (updates self.df_transformed).
        3. Fits Normalizers (GMM/MinMax) on the processed data.
        """
        self.df_raw = df
        self.df_transformed = df.copy()  # We operate on this copy

        # 1. Log Transformation (Learns shift AND transforms self.df_transformed)
        if self.config['processing']['log']:
            self._fit_apply_log()

        # 2. Residual Calculation (Modifies self.df_transformed in-place)
        self._apply_residuals()

        # 3. Numeric Modeling (Learns stats from self.df_transformed)
        self._fit_strategy_gmm()

        # 4. Categorical Modeling
        self._fit_categorical()

        return self

    def transform(self):
        """
        Uses self.df_transformed (already log/residual processed).
        Applies Normalization (GMM/MinMax) and OneHot Encoding.
        Returns a new DataFrame.
        """
        if self.df_transformed is None:
            raise RuntimeError("You must call fit(df) before calling transform()")

        # 1. Numeric Transformation (Scaling)
        # We pass self.df_transformed, which already has Log/Residuals applied.
        numeric_part = self._transform_numeric()

        # 2. Categorical Transformation (Encoding)
        categorical_part = self._transform_categorical()

        # 3. Merge
        combined_data = {**numeric_part, **categorical_part}
        out_df = pd.DataFrame(combined_data, index=self.df_transformed.index)
        self.generated_columns = out_df.columns.tolist()
        return out_df

    def inverse_transform(self, df):
        """
        Takes generated data (df) and reverses the process.
        Input: df (Scaled/Encoded)
        Output: df (Original Scale)
        """
        out_df = df.copy()

        # 1. Inverse Categorical
        out_df = self._inverse_categorical(out_df)

        # 2. Inverse Numeric (Scaling -> Log Space)
        out_df = self._inverse_numeric(out_df)

        # 3. Inverse Residuals (Reverse interaction between P1 and P2)
        out_df = self._inverse_residuals(out_df)

        # 4. Inverse Log (Expm1)
        if self.config['processing']['log']:
            out_df = self._inverse_log(out_df)

        return out_df

    # =========================================================================
    # 1. PRE-PROCESSING (LOG & RESIDUALS)
    # =========================================================================

    def _fit_apply_log(self):
        """
        Calculates shifts AND applies log transform to self.df_transformed immediately.
        """
        p1_vars = self.data_info['phase1_vars']
        p2_vars = self.data_info['phase2_vars']
        processed = set()

        # Handle Pairs (ensure same shift)
        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in self.data_info['num_vars']:
                vals1 = self.df_transformed[p1].dropna().astype(np.float32)
                vals2 = self.df_transformed[p2].dropna().astype(np.float32)
                combined_min = np.min(np.concatenate([vals1, vals2]))

                shift = abs(combined_min) + 1.0 if combined_min <= 0 else 0.0
                self.mins[p1] = {'shift': shift}
                self.mins[p2] = {'shift': shift}

                # Apply in-place
                self.df_transformed[p1] = np.log1p(vals1 + shift)
                self.df_transformed[p2] = np.log1p(vals2 + shift)
                processed.update([p1, p2])

        # Handle remaining numerics
        for num in self.data_info['num_vars']:
            if num not in processed:
                vals = self.df_transformed[num].astype(np.float32)
                shift = np.min(vals)
                shift = abs(shift) + 1.0 if shift <= 0 else 0.0
                self.mins[num] = {'shift': shift}
                self.df_transformed[num] = np.log1p(vals + shift)

    def _inverse_log(self, df):
        for col, params in self.mins.items():
            df[col] = np.expm1(df[col]) - params['shift']
        return df

    def _apply_residuals(self):
        """
        If residuals enabled: P2 = P1 - P2
        Modifies df in-place.
        """
        if not self.config['model'].get('residual_modeling', False):
            return

        p1_vars = self.data_info['phase1_vars']
        p2_vars = self.data_info['phase2_vars']
        num_vars_set = set(self.data_info['num_vars'])

        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in num_vars_set:
                self.df_transformed[p2] = self.df_transformed[p1] - self.df_transformed[p2]

    def _inverse_residuals(self, df):
        """
        If residuals enabled: Target(P2) = Proxy(P1) - Residual(P2)
        """
        if not self.config['model'].get('residual_modeling', False):
            return df

        p1_vars = self.data_info['phase1_vars']
        p2_vars = self.data_info['phase2_vars']
        num_vars_set = set(self.data_info['num_vars'])

        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in num_vars_set:
                df[p2] = df[p1] - df[p2]
        return df

    # =========================================================================
    # 2. NUMERIC STRATEGIES (FIT)
    # =========================================================================

    def _fit_strategy_gmm(self):
        p2_vars_set = set(self.data_info['phase2_vars'])
        max_components = self.config['processing']['gmm'].get('max_components', 5)

        for col in self.data_info['num_vars']:
            series = pd.to_numeric(self.df_transformed[col], errors='coerce')
            data = series.dropna().values.reshape(-1, 1)

            if col in p2_vars_set:
                if max_components == 1:
                    self.num_models[col] = self._fit_standard(data)
                    continue

                best_gmm = None
                best_bic = np.inf

                for k in range(1, max_components + 1):
                    try:
                        gmm = GaussianMixture(n_components=k, reg_covar=1e-3, max_iter=100)
                        gmm.fit(data)
                        bic = gmm.bic(data)
                        if bic < best_bic:
                            best_bic = bic
                            best_gmm = gmm
                    except Exception:
                        continue

                if best_gmm is not None:
                    if best_gmm.n_components > 1:
                        print(f"Variable {col} fitted with {best_gmm.n_components} components.")

                    self.num_models[col] = {
                        'type': 'gmm',
                        'model': best_gmm,
                        'means': best_gmm.means_.flatten().astype(np.float32),
                        'stds': np.sqrt(best_gmm.covariances_.flatten()).astype(np.float32),
                    }

                    mode_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    mode_enc.fit(np.arange(best_gmm.n_components).reshape(-1, 1))
                    self.mode_encoders[col] = mode_enc
            else:
                # Standard Normal for Phase 1
                self.num_models[col] = self._fit_standard(data)


    def _fit_standard(self, data):
        mu = float(np.mean(data))
        sigma = float(np.std(data))
        return {'type': 'zscore', 'mean': mu, 'std': sigma}

    # =========================================================================
    # 3. NUMERIC TRANSFORM (SCALING)
    # =========================================================================

    def _transform_numeric(self):
        collected = {}
        epsilon = self.config['processing']['gmm'].get('epsilon', 1e-6)

        for col, model in self.num_models.items():
            vals = pd.to_numeric(self.df_transformed[col], errors='coerce').values.astype(np.float32)
            mask = ~np.isnan(vals)
            res_val = np.zeros(vals.shape, dtype=np.float32)

            # 2. Extract Valid Data (Ignore NAs for transformation)
            valid_data = vals[mask]

            # 3. Apply Logic
            if model['type'] == 'const':
                res_val[mask] = valid_data - model['mean']
                collected[col] = res_val

            elif model['type'] == 'zscore':
                res_val[mask] = (valid_data - model['mean']) / (model['std'] + epsilon)
                collected[col] = res_val

            elif model['type'] == 'gmm':
                gmm = model['model']
                v_data_reshaped = valid_data.reshape(-1, 1)

                # GMM requires valid data for prediction
                probs = gmm.predict_proba(v_data_reshaped)
                comps = probs.argmax(axis=1)

                # Normalize
                norm_vals = (valid_data - model['means'][comps]) / (model['stds'][comps] + epsilon)
                res_val[mask] = norm_vals
                collected[col] = res_val

                # Mode Columns
                mode_enc = self.mode_encoders[col]
                mode_1hot = mode_enc.transform(comps.reshape(-1, 1))

                for i in range(mode_1hot.shape[1]):
                    mode_col = f"{col}_mode_{i}"
                    # Init with zeros, fill valid indices
                    res_mode = np.zeros(vals.shape, dtype=np.float32)
                    res_mode[mask] = mode_1hot[:, i]
                    collected[mode_col] = res_mode

        return collected

    def _inverse_numeric(self, out_df):
        epsilon = self.config['processing']['gmm'].get('epsilon', 1e-6)

        for col, model in self.num_models.items():
            if col not in out_df.columns: continue

            if model['type'] == 'const':
                out_df[col] = out_df[col] + model['mean']

            elif model['type'] == 'zscore':
                out_df[col] = out_df[col] * (model['std'] + epsilon) + model['mean']

            elif model['type'] == 'gmm':
                mode_enc = self.mode_encoders.get(col)
                if mode_enc:
                    n_modes = len(model['means'])
                    mode_cols = [f"{col}_mode_{i}" for i in range(n_modes)]
                    pres = [c for c in mode_cols if c in out_df.columns]

                    if pres:
                        modes = np.argmax(out_df[pres].values, axis=1)
                        stds = model['stds'][modes]
                        means = model['means'][modes]
                        out_df[col] = out_df[col] * (stds + epsilon) + means
                        out_df = out_df.drop(columns=pres, errors='ignore')
        return out_df

    # =========================================================================
    # CATEGORICAL
    # =========================================================================

    def _fit_categorical(self):
        p1_vars = self.data_info['phase1_vars']
        p2_vars = self.data_info['phase2_vars']
        processed = set()

        # Shared encoder
        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in self.data_info['cat_vars']:
                vals1 = self.df_transformed[p1].dropna().unique().astype(str)
                vals2 = self.df_transformed[p2].dropna().unique().astype(str)
                all_vals = np.union1d(vals1, vals2).reshape(-1, 1)

                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                enc.fit(all_vals)
                self.cat_encoders[p1] = enc
                self.cat_encoders[p2] = enc
                processed.update([p1, p2])

        # Independent encoder
        for col in self.data_info['cat_vars']:
            if col not in processed and col in self.df_transformed.columns:
                vals = self.df_transformed[col].dropna().unique().astype(str).reshape(-1, 1)
                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                enc.fit(vals)
                self.cat_encoders[col] = enc

    def _transform_categorical(self):
        collected = {}
        for col, enc in self.cat_encoders.items():
            if col not in self.df_transformed.columns: continue

            vals = self.df_transformed[col].astype(str).values.reshape(-1, 1)
            mask = ~self.df_transformed[col].isna().values
            cat_names = [f"{col}_{c}" for c in enc.categories_[0]]

            if mask.sum() > 0:
                encoded = enc.transform(vals[mask])
                for i, name in enumerate(cat_names):
                    res = np.zeros(vals.shape[0], dtype=np.float32)
                    res[mask] = encoded[:, i]
                    collected[name] = res
            else:
                for name in cat_names:
                    collected[name] = np.zeros(vals.shape[0], dtype=np.float32)
        return collected

    def _inverse_categorical(self, out_df):
        for col, enc in self.cat_encoders.items():
            cat_names = [f"{col}_{c}" for c in enc.categories_[0]]
            present = [c for c in cat_names if c in out_df.columns]
            if not present: continue

            logits = out_df[present].values
            indices = np.argmax(logits, axis=1)
            decoded = enc.categories_[0][indices]

            try:
                out_df[col] = pd.to_numeric(decoded)
            except:
                out_df[col] = decoded

            out_df = out_df.drop(columns=present, errors='ignore')
        return out_df

    def get_dims(self):
        p1, p2, cond = [], [], []
        p1_b = set(self.data_info['phase1_vars'])
        p2_b = set(self.data_info['phase2_vars'])

        cols = self.generated_columns if self.generated_columns else []
        for col in cols:
            if "_mode_" in col:
                base = col.split("_mode_")[0]
            elif "_" in col:
                parts = col.split('_')
                base = col
                for i in range(len(parts) - 1, 0, -1):
                    cand = "_".join(parts[:i])
                    if cand in self.data_info['cat_vars']:  # <--- Only checks Categoricals
                        base = cand
                        break
            else:
                base = col

            if base in p1_b:
                p1.append(col)
            elif base in p2_b:
                p2.append(col)
            else:
                cond.append(col)
        return p1, p2, cond


class ImputationDataset(Dataset):
    def __init__(self, data, p1_cols, p2_cols, cond_cols):
        self.data = data
        self.p1_cols = p1_cols
        self.p2_cols = p2_cols
        self.cond_cols = cond_cols

        self.A = torch.FloatTensor(data[p1_cols].to_numpy().copy())

        # P2 contains NaNs -> Replace with 0 for safety in MatMul
        # Mask M keeps track of where real data is.
        self.X_raw = data[p2_cols].to_numpy().copy()
        self.M = torch.FloatTensor(1 - np.isnan(self.X_raw))
        self.X = torch.FloatTensor(np.nan_to_num(self.X_raw))

        self.C = torch.FloatTensor(data[cond_cols].to_numpy().copy())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'A': self.A[idx],
            'X': self.X[idx],
            'M': self.M[idx],
            'C': self.C[idx]
        }