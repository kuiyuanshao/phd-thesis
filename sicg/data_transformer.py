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

        self.df_raw = None
        self.df_transformed = None

        self.mins = {}
        self.num_models = {}
        self.cat_encoders = {}
        self.mode_encoders = {}

        self.generated_columns = None
        self.norm_type = config['processing'].get('normalization', 'gmm')


    def fit(self, df):
        self.df_raw = df
        self.df_transformed = df.copy()

        if self.config['processing']['log']:
            self._fit_apply_log()

        self._apply_residuals()
        self._fit_strategy_gmm()
        self._fit_categorical()

        return self

    def transform(self):
        if self.df_transformed is None:
            raise RuntimeError("You must call fit(df) before calling transform()")

        numeric_part = self._transform_numeric()
        categorical_part = self._transform_categorical()

        combined_data = {**numeric_part, **categorical_part}
        out_df = pd.DataFrame(combined_data, index=self.df_transformed.index)
        self.generated_columns = out_df.columns.tolist()
        return out_df

    def inverse_transform(self, df):
        out_df = df.copy()
        out_df = self._inverse_categorical(out_df)
        out_df = self._inverse_numeric(out_df)
        out_df = self._inverse_residuals(out_df)

        if self.config['processing']['log']:
            out_df = self._inverse_log(out_df)

        return out_df

    @staticmethod
    def _robust_string(val):
        s = str(val)
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
        except (ValueError, TypeError):
            pass
        return s.strip()

    def _fit_apply_log(self):
        p1_vars = self.data_info['phase1_vars']
        p2_vars = self.data_info['phase2_vars']
        processed = set()

        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in self.data_info['num_vars']:
                vals1 = self.df_transformed[p1].dropna().astype(np.float32)
                vals2 = self.df_transformed[p2].dropna().astype(np.float32)
                combined_min = np.min(np.concatenate([vals1, vals2]))

                shift = abs(combined_min) + 1.0 if combined_min <= 0 else 0.0
                self.mins[p1] = {'shift': shift}
                self.mins[p2] = {'shift': shift}

                self.df_transformed[p1] = np.log1p(vals1 + shift)
                self.df_transformed[p2] = np.log1p(vals2 + shift)
                processed.update([p1, p2])

        for num in self.data_info['num_vars']:
            if num not in processed:
                vals = self.df_transformed[num].astype(np.float32)
                shift = np.min(vals)
                shift = abs(shift) + 1.0 if shift <= 0 else 0.0
                self.mins[num] = {'shift': shift}
                self.df_transformed[num] = np.log1p(vals + shift)

    def _inverse_log(self, df):
        for col, params in self.mins.items():
            if col in df.columns:
                df[col] = np.expm1(df[col]) - params['shift']
        return df

    def _apply_residuals(self):
        if not self.config['model'].get('residual_modeling', False):
            return

        p1_vars = self.data_info['phase1_vars']
        p2_vars = self.data_info['phase2_vars']
        num_vars_set = set(self.data_info['num_vars'])

        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in num_vars_set:
                self.df_transformed[p2] = self.df_transformed[p1] - self.df_transformed[p2]

    def _inverse_residuals(self, df):
        if not self.config['model'].get('residual_modeling', False):
            return df

        p1_vars = self.data_info['phase1_vars']
        p2_vars = self.data_info['phase2_vars']
        num_vars_set = set(self.data_info['num_vars'])

        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in num_vars_set and p1 in df.columns and p2 in df.columns:
                df[p2] = df[p1] - df[p2]
        return df

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
                self.num_models[col] = self._fit_standard(data)

    def _fit_standard(self, data):
        mu = float(np.mean(data))
        sigma = float(np.std(data))
        return {'type': 'zscore', 'mean': mu, 'std': sigma}

    def _transform_numeric(self):
        collected = {}
        epsilon = self.config['processing']['gmm'].get('epsilon', 1e-6)

        for col, model in self.num_models.items():
            vals = pd.to_numeric(self.df_transformed[col], errors='coerce').values.astype(np.float32)
            mask = ~np.isnan(vals)
            res_val = np.zeros(vals.shape, dtype=np.float32)
            valid_data = vals[mask]

            if model['type'] == 'const':
                res_val[mask] = valid_data - model['mean']
                collected[col] = res_val

            elif model['type'] == 'zscore':
                res_val[mask] = (valid_data - model['mean']) / (model['std'] + epsilon)
                collected[col] = res_val

            elif model['type'] == 'gmm':
                gmm = model['model']
                v_data_reshaped = valid_data.reshape(-1, 1)
                probs = gmm.predict_proba(v_data_reshaped)
                comps = probs.argmax(axis=1)

                norm_vals = (valid_data - model['means'][comps]) / (model['stds'][comps] + epsilon)
                res_val[mask] = norm_vals
                collected[col] = res_val

                mode_enc = self.mode_encoders[col]
                mode_1hot = mode_enc.transform(comps.reshape(-1, 1))

                for i in range(mode_1hot.shape[1]):
                    mode_col = f"{col}_mode_{i}"
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

    def _fit_categorical(self):
        for col in self.data_info['cat_vars']:
            if col in self.df_transformed.columns:
                non_null_mask = self.df_transformed[col].notna()
                if non_null_mask.sum() == 0:
                    continue
                normalized_series = self.df_transformed.loc[non_null_mask, col].apply(self._robust_string)
                if self.df_transformed[col].dtype != 'object':
                    self.df_transformed[col] = self.df_transformed[col].astype('object')

                self.df_transformed.loc[non_null_mask, col] = normalized_series
                vals = normalized_series.unique().reshape(-1, 1)
                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                enc.fit(vals)
                self.cat_encoders[col] = enc

    def _transform_categorical(self):
        collected = {}
        for col, enc in self.cat_encoders.items():
            if col not in self.df_transformed.columns: continue

            series = self.df_transformed[col]
            mask = series.notna().values

            cat_names = [f"{col}_{c}" for c in enc.categories_[0]]

            if mask.sum() > 0:
                raw_vals = series[mask].apply(self._robust_string).values.reshape(-1, 1)
                encoded = enc.transform(raw_vals)

                for i, name in enumerate(cat_names):
                    res = np.zeros(len(series), dtype=np.float32)
                    res[mask] = encoded[:, i]
                    collected[name] = res
            else:
                for name in cat_names:
                    collected[name] = np.zeros(len(series), dtype=np.float32)

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
                # Robustly find base variable name by checking against known cat vars
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

        self.A = torch.FloatTensor(data[p1_cols].to_numpy().copy())
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