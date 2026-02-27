import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
import torch

class DataTransformer:
    def __init__(self, data_info):
        self.data_info = data_info
        self.df_raw = None
        self.df_transformed = None

        self.mins = {}
        self.maxs = {}
        self.cat_encoders = {}
        self.generated_columns = None

    def fit(self, df):
        self.df_raw = df
        self.df_transformed = df.copy()
        self._fit_numeric()
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
        return out_df

    def _fit_numeric(self):
        for col in self.data_info['num_vars']:
            if col in self.df_transformed.columns:
                series = pd.to_numeric(self.df_transformed[col], errors='coerce')
                self.mins[col] = series.min(skipna=True)
                self.maxs[col] = series.max(skipna=True)

    def _transform_numeric(self):
        collected = {}
        for col in self.data_info['num_vars']:
            if col in self.df_transformed.columns:
                vals = pd.to_numeric(self.df_transformed[col], errors='coerce').values.astype(np.float32)
                mask = ~np.isnan(vals)
                res_val = np.full(vals.shape, np.nan, dtype=np.float32)

                min_val = self.mins[col]
                max_val = self.maxs[col]
                valid_data = vals[mask]

                res_val[mask] = (valid_data - min_val) / (max_val - min_val + 1e-6)
                collected[col] = res_val
        return collected

    def _inverse_numeric(self, out_df):
        for col in self.data_info['num_vars']:
            if col in out_df.columns:
                min_val = self.mins[col]
                max_val = self.maxs[col]
                out_df[col] = out_df[col] * (max_val - min_val + 1e-6) + min_val
        return out_df


    def _fit_categorical(self):
        for col in self.data_info['cat_vars']:
            if col in self.df_transformed.columns:
                non_null_mask = self.df_transformed[col].notna()
                if non_null_mask.sum() == 0:
                    continue

                normalized_series = self.df_transformed.loc[non_null_mask, col]
                self.df_transformed.loc[non_null_mask, col] = normalized_series

                vals = normalized_series.unique().reshape(-1, 1)
                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                enc.fit(vals)
                self.cat_encoders[col] = enc

    def _transform_categorical(self):
        collected = {}
        for col, enc in self.cat_encoders.items():
            if col not in self.df_transformed.columns:
                continue

            series = self.df_transformed[col]
            mask = series.notna().values
            cat_names = [f"{col}_{c}" for c in enc.categories_[0]]

            if mask.sum() > 0:
                raw_vals = series[mask].values.reshape(-1, 1)
                encoded = enc.transform(raw_vals)

                for i, name in enumerate(cat_names):
                    res = np.full(len(series), np.nan, dtype=np.float32)
                    res[mask] = encoded[:, i]
                    collected[name] = res
            else:
                for name in cat_names:
                    collected[name] = np.full(len(series), np.nan, dtype=np.float32)

        return collected

    def _inverse_categorical(self, out_df):
        for col, enc in self.cat_encoders.items():
            cat_names = [f"{col}_{c}" for c in enc.categories_[0]]
            present = [c for c in cat_names if c in out_df.columns]
            if not present:
                continue

            logits = out_df[present].values
            indices = np.argmax(logits, axis=1)
            decoded = enc.categories_[0][indices]

            try:
                out_df[col] = pd.to_numeric(decoded)
            except:
                out_df[col] = decoded

            out_df = out_df.drop(columns=present, errors='ignore')
        return out_df


class ImputationDataset(Dataset):
    def __init__(self, observed_values, observed_masks):
        self.observed_values = observed_values
        self.observed_masks = observed_masks
        self.gt_masks = observed_masks.copy()
        self.eval_length = observed_values.shape[1]

    def __len__(self):
        return len(self.observed_values)

    def __getitem__(self, index):
        return {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length)
        }

