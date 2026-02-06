import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder


class DataTransformer:
    def __init__(self, data_info, config):
        self.data_info = data_info
        self.config = config
        self.num_models = {}
        self.cat_encoders = {}
        self.mode_encoders = {}
        self.generated_columns = None
        self.mins = {}
        self.df_raw = None
        self.df_transformed = None

    def fit(self, df):
        self.df_raw = df
        df = df.copy()
        p1_vars = self.data_info['phase1_vars']
        p2_vars = self.data_info['phase2_vars']

        processed_nums = set()
        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in self.data_info['num_vars']:
                vals1 = df[p1].dropna()
                vals2 = df[p2].dropna()
                vals1, vals2 = vals1.astype(np.float32), vals2.astype(np.float32)
                combined_min = np.nanmin(np.concatenate([vals1, vals2]))
                shift = 0.0
                if combined_min <= 0:
                    shift = abs(combined_min) + 1.0
                self.mins[p1] = {'shift': shift}
                self.mins[p2] = {'shift': shift}
                v1_log = np.log1p(vals1 + shift)
                v2_log = np.log1p(vals2 + shift)
                df[p1] = v1_log
                df[p2] = v2_log

                processed_nums.add(p1)
                processed_nums.add(p2)

        for num in self.data_info['num_vars']:
            if num not in processed_nums:
                vals = df[num]
                vals = vals.astype(np.float32)
                shift = np.min(vals)
                self.mins[num] = {'shift': shift}
                v_log = np.log1p(vals + shift)
                df[num] = v_log
        
        if self.config['model']['residual_modeling']:
            print("Residual Modeling Enabled.")
            num_vars = set(self.data_info['num_vars'])
            for p1, p2 in zip(p1_vars, p2_vars):
                if p1 in num_vars:
                    if p1 in df and p2 in df:
                        df[p2] = df[p1] - df[p2]
        self.df_transformed = df.copy()
        p2_vars_set = set(self.data_info['phase2_vars'])
        # Cap components to prevent explosion, but allow flexibility
        max_components = self.config['processing']['gmm']['max_components']
        # --- 1. Fit Numerical Models ---
        for col in self.data_info['num_vars']:
            series = pd.to_numeric(df[col], errors='coerce')
            full_data = series.dropna().values.astype(np.float32).reshape(-1, 1)
            # --- PHASE 2: GMM WITH BIC SELECTION ---
            if col in p2_vars_set:
                best_gmm = None
                best_bic = np.inf

                for k in range(1, max_components + 1):
                    try:
                        gmm = GaussianMixture(n_components=k, reg_covar=1e-3, max_iter=50)
                        gmm.fit(full_data)
                        bic = gmm.bic(full_data)
                        if bic < best_bic:
                            best_bic = bic
                            best_gmm = gmm
                    except Exception:
                        continue

                # Store Model
                # Even if k=1 is selected, we keep it as GMM structure to satisfy requirement
                # If everything failed (unlikely with reg_covar), fallback to simple stats
                if best_gmm is not None:
                    zero_modes = []
                    for i, (m, s) in enumerate(
                            zip(best_gmm.means_.flatten(), np.sqrt(best_gmm.covariances_.flatten()))):
                        # Thresholds: Adjust based on your data scale.
                        # For residuals, exact equality usually creates a std < 1e-4
                        if abs(m) < 0.01 and s < 0.05:
                            zero_modes.append(i)

                    self.num_models[col] = {
                        'type': 'gmm',
                        'model': best_gmm,
                        'means': best_gmm.means_.flatten().astype(np.float32),
                        'stds': np.sqrt(best_gmm.covariances_.flatten()).astype(np.float32),
                        'zero_modes': zero_modes
                    }
                    # Mode Encoder
                    mode_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    mode_enc.fit(np.arange(best_gmm.n_components).reshape(-1, 1))
                    self.mode_encoders[col] = mode_enc
                else:
                    # Safety fallback
                    mu, sigma = np.mean(full_data), np.std(full_data)
                    self.num_models[col] = {'type': 'zscore', 'mean': float(mu), 'std': float(sigma)}

            # --- PHASE 1 / AUX: Z-SCORE ---
            else:
                mu = np.mean(full_data)
                sigma = np.std(full_data)
                self.num_models[col] = {'type': 'zscore', 'mean': float(mu), 'std': float(sigma)}

        # --- 2. Fit Categorical Encoders ---
        processed_cats = set()

        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in self.data_info['cat_vars']:
                vals1 = df[p1].dropna().unique() 
                vals2 = df[p2].dropna().unique() 
                vals1, vals2 = vals1.astype(str), vals2.astype(str)
                all_vals = np.union1d(vals1, vals2).reshape(-1, 1)

                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                enc.fit(all_vals)
                self.cat_encoders[p1] = enc
                self.cat_encoders[p2] = enc
                processed_cats.add(p1)
                processed_cats.add(p2)

        for col in self.data_info['cat_vars']:
            if col not in processed_cats and col in df.columns:
                vals = df[col].dropna().unique().astype(str).reshape(-1, 1)
                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                enc.fit(vals)
                self.cat_encoders[col] = enc

    def transform(self):
        collected_data = {}
        epsilon = self.config['processing']['gmm'].get('epsilon', 1e-6)

        # 1. Numerical
        for col in self.data_info['num_vars']:
            if col not in self.df_transformed.columns: continue

            vals = pd.to_numeric(self.df_transformed[col], errors='coerce').values.astype(np.float32)
            mask = ~np.isnan(vals)
            m_info = self.num_models.get(col)

            if not m_info:
                collected_data[col] = np.full(vals.shape, np.nan, dtype=np.float32)
                continue

            # Default NaN array
            res_val = np.full(vals.shape, np.nan, dtype=np.float32)

            if m_info['type'] == 'const':
                res_val[mask] = vals[mask] - m_info['mean']
                collected_data[col] = res_val

            elif m_info['type'] == 'zscore':
                res_val[mask] = (vals[mask] - m_info['mean']) / (m_info['std'] + epsilon)
                collected_data[col] = res_val

            elif m_info['type'] == 'gmm':
                gmm = m_info['model']
                v_data = vals[mask].reshape(-1, 1)

                if len(v_data) > 0:
                    probs = gmm.predict_proba(v_data)
                    comps = probs.argmax(axis=1)

                    # Normalize based on component statistics
                    norm_vals = (v_data.flatten() - m_info['means'][comps]) / (m_info['stds'][comps] + epsilon)
                    res_val[mask] = norm_vals
                    collected_data[col] = res_val

                    # Create Mode Columns (One Hot)
                    mode_enc = self.mode_encoders[col]
                    mode_1hot = mode_enc.transform(comps.reshape(-1, 1))

                    for i in range(mode_1hot.shape[1]):
                        mode_col = f"{col}_mode_{i}"
                        res_mode = np.zeros(vals.shape, dtype=np.float32)
                        res_mode[mask] = mode_1hot[:, i]
                        collected_data[mode_col] = res_mode
                else:
                    collected_data[col] = res_val
                    # Fill modes with 0/NaN
                    mode_enc = self.mode_encoders.get(col)
                    if mode_enc:
                        for i in range(len(mode_enc.categories_[0])):
                            collected_data[f"{col}_mode_{i}"] = np.full(vals.shape, 0.0, dtype=np.float32)

        # 2. Categorical
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
                    collected_data[name] = res
            else:
                for name in cat_names:
                    collected_data[name] = np.full(vals.shape[0], np.nan, dtype=np.float32)

        out_df = pd.DataFrame(collected_data, index=self.df_transformed.index)
        self.generated_columns = out_df.columns.tolist()
        return out_df

    def inverse_transform(self, df):
        out_df = df.copy()
        epsilon = self.config['processing']['gmm'].get('epsilon', 1e-6)

        # 1. Categorical
        for col, enc in self.cat_encoders.items():
            cat_names = [f"{col}_{c}" for c in enc.categories_[0]]
            present_cols = [c for c in cat_names if c in df.columns]
            if not present_cols: continue

            logits = df[present_cols].values
            indices = np.argmax(logits, axis=1)
            decoded_vals = enc.categories_[0][indices]

            try:
                out_df[col] = pd.to_numeric(decoded_vals)
            except (ValueError, TypeError):
                out_df[col] = decoded_vals

            out_df.drop(columns=present_cols, inplace=True, errors='ignore')

        # 2. Numerical
        p1_vars = self.data_info['phase1_vars']
        p2_vars = self.data_info['phase2_vars']
        for col in self.data_info['num_vars']:
            m = self.num_models.get(col)

            if m['type'] == 'const':
                out_df[col] = out_df[col] + m['mean']

            elif m['type'] == 'zscore':
                out_df[col] = out_df[col] * (m['std'] + epsilon) + m['mean']

            elif m['type'] == 'gmm':
                # GMM Inverse Logic
                mode_enc = self.mode_encoders.get(col)
                if mode_enc:
                    n_modes = len(m['means'])
                    mode_cols = [f"{col}_mode_{i}" for i in range(n_modes)]
                    pres = [c for c in mode_cols if c in df.columns]

                    if pres:
                        modes = np.argmax(df[pres].values, axis=1)
                        # Component-specific de-standardization
                        out_df[col] = out_df[col] * (m['stds'][modes] + epsilon) + m['means'][modes]
                        if 'zero_modes' in m and m['zero_modes']:
                            for z_mode in m['zero_modes']:
                                is_zero = (modes == z_mode)
                                out_df.loc[is_zero, col] = 0.0
                        out_df.drop(columns=pres, inplace=True, errors='ignore')

        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in self.data_info['num_vars']:
                if p1 in df and p2 in out_df:
                    if self.config["model"]["residual_modeling"]:
                        val_log = out_df[p1] - out_df[p2]
                    else:
                        val_log = out_df[p2]
                    final_val = np.expm1(val_log) - self.mins[p2]['shift']
                    out_df[p2] = final_val

        return out_df

    def get_dims(self):
        p1, p2, cond = [], [], []
        p1_b = set(self.data_info['phase1_vars'])
        p2_b = set(self.data_info['phase2_vars'])

        for col in self.generated_columns:
            if "_mode_" in col:
                base = col.split("_mode_")[0]
            elif "_" in col:
                parts = col.split('_')
                base = col
                for i in range(len(parts) - 1, 0, -1):
                    cand = "_".join(parts[:i])
                    if cand in self.data_info['cat_vars']:
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

        self.A = torch.FloatTensor(data[p1_cols].values)
        self.X_raw = data[p2_cols].values
        self.M = torch.FloatTensor(1 - np.isnan(self.X_raw))
        self.X = torch.FloatTensor(np.nan_to_num(self.X_raw))
        self.C = torch.FloatTensor(data[cond_cols].values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'A': self.A[idx],
            'X': self.X[idx],
            'M': self.M[idx],
            'C': self.C[idx]
        }