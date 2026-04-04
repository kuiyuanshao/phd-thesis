import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm, gaussian_kde
import numpy as np

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

        processing_cfg = self.config.get("processing", {})
        self.norm_type = processing_cfg.get('normalization', "zscore")
        self.log = processing_cfg.get('log', True)
        self.anchor = processing_cfg.get('anchor', False)
        self.discrete = self.config.get("diffusion", {}).get("discrete", "Multinomial")
        # Metadata for tensorization
        self.q_block_matrix = None

    def fit(self, df):
        # Fit the transformer by preparing categorical strings and numeric parameters
        self.df_raw = df.copy()
        self._clean_categorical_strings(self.df_raw)
        self.df_transformed = self.df_raw.copy()

        if self.log:
            self._fit_apply_log()
        self._fit_numeric()
        self._fit_categorical()

        return self

    def transform(self):
        target_df = self.df_transformed
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
        if self.log:
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


    def _inverse_log(self, df):
        for col, shift in self.mins.items():
            if col in df.columns:
                df[col] = np.expm1(df[col]) - shift
        return df

    def _fit_numeric(self):
        p1_vars = self.data_info.get('phase1_vars', [])
        p2_vars = self.data_info.get('phase2_vars', [])
        self.pair_map = {p2: p1 for p1, p2 in
                         zip(self.data_info.get('phase1_vars', []), self.data_info.get('phase2_vars', []))}

        for col in self.data_info['num_vars']:
            if col not in self.df_transformed.columns:
                continue

            data = self.df_transformed[col].dropna().values.reshape(-1, 1)
            n_quant = min(len(data), 1000)

            if col in p1_vars or (col not in p1_vars and col not in p2_vars):
                if self.norm_type == "quantile":
                    qt = QuantileTransformer(output_distribution='normal', n_quantiles=n_quant,
                                             random_state=42)
                    qt.fit(data)
                    self.num_models[col] = {
                        'type': 'quantile',
                        'model': qt
                    }
                elif self.norm_type == "minmax":
                    self.num_models[col] = self._fit_minmax(data)

                elif self.norm_type == "spline":
                    self.num_models[col] = self._fit_spline(data)
                else:
                    self.num_models[col] = self._fit_zscore(data)

        for col in self.data_info['num_vars']:
            if col in p2_vars:
                data_p2 = self.df_transformed[col].dropna().values.reshape(-1, 1)
                n_quant_p2 = min(len(data_p2), 1000)

                if self.anchor:
                    anchor_col = self.pair_map.get(col)
                    if anchor_col and anchor_col in self.num_models:
                        anchor_model = self.num_models[anchor_col]
                        self.num_models[col] = anchor_model
                    else:
                        self.num_models[col] = self._fit_zscore(data_p2)
                else:
                    if self.norm_type == "quantile":
                        qt = QuantileTransformer(output_distribution='normal', n_quantiles=n_quant_p2,
                                                 random_state=42)
                        qt.fit(data_p2)
                        self.num_models[col] = {
                            'type': 'quantile',
                            'model': qt
                        }
                    elif self.norm_type == "kde_spline":
                        self.num_models[col] = self._fit_kde_spline(data_p2)
                    elif self.norm_type == "minmax":
                        self.num_models[col] = self._fit_minmax(data_p2)
                    elif self.norm_type == "spline":
                        self.num_models[col] = self._fit_spline(data_p2)
                    else:
                        self.num_models[col] = self._fit_zscore(data_p2)

    def _fit_zscore(self, data):
        return {'type': 'zscore', 'mean': float(np.mean(data)), 'std': float(np.std(data))}

    def _fit_spline(self, data):
        data = data.flatten()
        x_unique, indices = np.unique(data, return_inverse=True)
        if len(x_unique) < 2:
            return {'type': 'zscore', 'mean': float(np.mean(data)), 'std': float(1e-6)}
        counts = np.bincount(indices)
        u_unique = (np.cumsum(counts) - 0.5 * counts) / data.size
        forward_spline = PchipInterpolator(x_unique, u_unique)
        inverse_spline = PchipInterpolator(u_unique, x_unique)

        return {
            'type': 'spline',
            'forward': forward_spline,
            'inverse': inverse_spline,
            'u_min': float(u_unique.min()),
            'u_max': float(u_unique.max())
        }

    def _fit_kde_spline(self, data):
        data = data.flatten()
        x_unique = np.unique(data)

        if len(x_unique) < 2 or np.std(data) < 1e-6:
            return {'type': 'zscore', 'mean': float(np.mean(data)), 'std': float(1e-6)}

        try:
            kde = gaussian_kde(data)
        except np.linalg.LinAlgError:
            return {'type': 'zscore', 'mean': float(np.mean(data)), 'std': float(np.std(data) + 1e-6)}

        data_min, data_max = np.min(data), np.max(data)
        margin = (data_max - data_min) * 0.1
        margin = margin if margin > 0 else 1.0

        x_grid = np.linspace(data_min - margin, data_max + margin, 1000)
        pdf_grid = kde(x_grid)
        pdf_grid = np.maximum(pdf_grid, 1e-10)

        cdf_grid = np.cumsum(pdf_grid)
        cdf_grid = cdf_grid / cdf_grid[-1]
        forward_spline = PchipInterpolator(x_grid, cdf_grid)
        inverse_spline = PchipInterpolator(cdf_grid, x_grid)

        return {
            'type': 'kde_spline',
            'forward': forward_spline,
            'inverse': inverse_spline,
            'u_min': float(cdf_grid[0]),
            'u_max': float(cdf_grid[-1])
        }

    def _fit_minmax(self, data):
        return {'type': 'minmax', 'min': float(np.min(data)), 'max': float(np.max(data))}

    def _transform_numeric(self, df):
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
            elif model['type'] == 'minmax':
                if mask.sum() > 0:
                    res_val[mask] = (vals[mask] - model['min']) / (model['max'] - model['min'] + epsilon)
                collected[col] = res_val
            elif model['type'] == 'spline':
                if mask.sum() > 0:
                    valid_x = vals[mask]
                    u = model['forward'](valid_x)
                    u_clamped = np.clip(u, epsilon, 1.0 - epsilon)
                    res_val[mask] = norm.ppf(u_clamped).astype(np.float32)
                res_val[~mask] = 0.0
                collected[col] = res_val
            elif model['type'] == 'kde_spline':
                if mask.sum() > 0:
                    valid_x = vals[mask]
                    # Map x -> u via the smoothed spline
                    u = model['forward'](valid_x)
                    # Clamp probabilities to prevent Probit explosion (NaNs)
                    u_clamped = np.clip(u, epsilon, 1.0 - epsilon)
                    # Map u -> N(0,1)
                    res_val[mask] = norm.ppf(u_clamped).astype(np.float32)

                res_val[~mask] = 0.0  # Standard missing mask
                collected[col] = res_val
        return collected

    def _inverse_numeric(self, df):
        epsilon = 1e-6
        for col in self.data_info['num_vars']:
            if col not in df.columns: continue
            model = self.num_models.get(col)
            if not model: continue
            if model['type'] == 'zscore':
                df[col] = df[col] * (model['std'] + epsilon) + model['mean']
            elif model['type'] == 'quantile':
                data = df[col].values.reshape(-1, 1)
                df[col] = model['model'].inverse_transform(data).flatten()
            elif model['type'] == 'minmax':
                df[col] = df[col] * (model['max'] - model['min'] + epsilon) + model['min']
            elif model['type'] == 'spline':
                z = df[col].values
                u = norm.cdf(z)
                u_clamped = np.clip(u, model['u_min'], model['u_max'])
                df[col] = model['inverse'](u_clamped).astype(np.float32)
            elif model['type'] == 'kde_spline':
                z = df[col].values
                u = norm.cdf(z)
                u_clamped = np.clip(u, model['u_min'], model['u_max'])
                df[col] = model['inverse'](u_clamped).astype(np.float32)
        return df

    def _fit_categorical(self):
        p1_vars = self.data_info.get('phase1_vars', [])
        p2_vars = self.data_info.get('phase2_vars', [])
        is_bits = self.discrete == 'AnalogBits'

        pair_unions = {}
        # Pre-calculate category unions for pairs
        for p1, p2 in zip(p1_vars, p2_vars):
            if p1 in self.df_transformed.columns and p2 in self.df_transformed.columns:
                v1 = self.df_transformed[p1].dropna().astype(str)
                v1 = v1[~v1.isin(["nan", "None", "NaN"])]
                v2 = self.df_transformed[p2].dropna().astype(str)
                v2 = v2[~v2.isin(["nan", "None", "NaN"])]
                pair_unions[p1] = np.union1d(v1.unique(), v2.unique())
                pair_unions[p2] = pair_unions[p1]

        for col in self.data_info['cat_vars']:
            if col not in self.df_transformed.columns: continue
            series = self.df_transformed[col].dropna().astype(str)
            series_valid = series[~series.isin(["nan", "None", "NaN"])]
            original_cats = series_valid.unique()
            fallback_cat = series_valid.mode()[0] if not series_valid.empty else "Missing"

            cats = pair_unions.get(col, original_cats)

            if is_bits:
                num_classes = len(cats)
                num_bits = max(1, int(np.ceil(np.log2(num_classes))))

                self.cat_encoders[col] = {
                    'type': 'bits',
                    'num_bits': num_bits,
                    'cat_to_int': {c: i for i, c in enumerate(cats)},
                    'int_to_cat': {i: c for i, c in enumerate(cats)},
                    'union_categories': cats,
                    'original_categories': set(original_cats),  # Strict domain
                    'fallback': fallback_cat
                }
            else:
                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                enc.fit(cats.reshape(-1, 1))
                self.cat_encoders[col] = {
                    'type': 'onehot',
                    'encoder': enc,
                    'union_categories': cats,
                    'original_categories': set(original_cats),  # Strict domain
                    'fallback': fallback_cat
                }

    def _compute_q_matrix(self, p1, p2):
        valid = self.df_transformed[p1].notna() & self.df_transformed[p2].notna()
        d1 = self.df_transformed.loc[valid, p1].astype(str)
        d2 = self.df_transformed.loc[valid, p2].astype(str)

        enc1 = self.cat_encoders[p1]
        enc2 = self.cat_encoders[p2]

        if enc1['type'] == 'onehot':
            idx1 = enc1['encoder'].transform(d1.values.reshape(-1, 1)).argmax(axis=1)
            idx2 = enc2['encoder'].transform(d2.values.reshape(-1, 1)).argmax(axis=1)
        elif enc1['type'] == 'bits':
            idx1 = np.array([enc1['cat_to_int'].get(val, 0) for val in d1])
            idx2 = np.array([enc2['cat_to_int'].get(val, 0) for val in d2])
        K1, K2 = len(enc1['union_categories']), len(enc2['union_categories'])

        Q = np.zeros((K1, K2))
        np.add.at(Q, (idx1, idx2), 1)
        self.Q_matrices[p2] = torch.tensor((Q + 1) / (Q + 1).sum(axis=1, keepdims=True), dtype=torch.float32)

    def _transform_categorical(self, df):
        collected = {}
        for col, enc in self.cat_encoders.items():
            if col not in df.columns: continue
            mask = df[col].notna().values
            series_filled = df[col].fillna(enc['fallback']).astype(str)
            if enc['type'] == 'onehot':
                encoded = enc['encoder'].transform(series_filled.values.reshape(-1, 1))
                encoded[~mask] = 0.0
                for i, c in enumerate(enc['union_categories']):
                    collected[f"{col}_{c}"] = encoded[:, i].astype(np.float32)

            elif enc['type'] == 'bits':
                num_bits = enc['num_bits']
                int_vals = np.array([enc['cat_to_int'].get(val, 0) for val in series_filled])
                for b in range(num_bits - 1, -1, -1):
                    bit_col = ((int_vals >> b) & 1).astype(np.float32) * 2.0 - 1.0
                    collected[f"{col}_bit_{num_bits - 1 - b}"] = bit_col

        return collected

    def _inverse_categorical(self, df):
        for col, enc in self.cat_encoders.items():
            if enc['type'] == 'onehot':
                cats = enc['union_categories']
                col_names = [f"{col}_{c}" for c in cats]
                present = [c for c in col_names if c in df.columns]
                if not present: continue

                indices = np.argmax(df[present].values, axis=1)
                raw_decoded = cats[indices]
                df[col] = [
                    val if val in enc['original_categories'] else enc['fallback']
                    for val in raw_decoded
                ]
                df.drop(columns=present, inplace=True)

            elif enc['type'] == 'bits':
                num_bits = enc['num_bits']
                col_names = [f"{col}_bit_{b}" for b in range(num_bits)]
                present = [c for c in col_names if c in df.columns]
                if len(present) != num_bits: continue
                bit_vals = df[present].values > 0.0
                powers_of_two = 2 ** np.arange(num_bits - 1, -1, -1)
                reconstructed_ints = (bit_vals * powers_of_two).sum(axis=1)
                def map_back(val):
                    decoded_cat = enc['int_to_cat'].get(int(val), None)
                    if decoded_cat is None or decoded_cat not in enc['original_categories']:
                        return enc['fallback']
                    return decoded_cat
                df[col] = [map_back(v) for v in reconstructed_ints]
                df.drop(columns=present, inplace=True)

        return df

    def get_sird_schema(self):
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

        for col in self.data_info['cat_vars']:
            enc = self.cat_encoders.get(col)
            if not enc: continue

            if enc['type'] == 'onehot':
                cats = enc['union_categories']
                idxs = sorted([df_cols.index(f"{col}_{c}") for c in cats if f"{col}_{c}" in df_cols])
            elif enc['type'] == 'bits':
                num_bits = enc['num_bits']
                idxs = sorted([df_cols.index(f"{col}_bit_{b}") for b in range(num_bits) if f"{col}_bit_{b}" in df_cols])

            if idxs:
                v_role, pair = get_type_pair(col)
                schema.append({'name': col, 'type': f'categorical_{v_role}',
                               'num_classes': len(idxs), 'start_idx': idxs[0], 'end_idx': idxs[-1] + 1,
                               'pair_partner': pair})
        return schema
