import pandas as pd
import numpy as np
import category_encoders as ce


class DataTransformer:
    def __init__(self, config, data_info):
        self.data_info = data_info
        self.config = config
        self.task = self.config.get('model', {}).get('task', 'ft')

        self.num_vars = data_info.get('num_vars', [])
        self.cat_vars = data_info.get('cat_vars', [])

        self.mins = {}
        self.maxs = {}
        self.encoder = None

        self.cont_list = []
        self.num_cate_list = []
        self.transformed_columns = []
        self.onehot_cols = []

    def fit(self, df):
        # Fit continuous variables
        for col in self.num_vars:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce')
                self.mins[col] = series.min(skipna=True)
                self.maxs[col] = series.max(skipna=True)
                self.cont_list.append(col)

        # Fit categorical variables
        present_cat_vars = [col for col in self.cat_vars if col in df.columns]
        if present_cat_vars:
            if self.task == "ft":
                self.encoder = ce.ordinal.OrdinalEncoder(cols=present_cat_vars)
                self.encoder.fit(df)
                temp_transformed = self.encoder.transform(df)
                for col in present_cat_vars:
                    self.num_cate_list.append(temp_transformed[col].nunique())
                self.transformed_columns = self.num_vars + present_cat_vars

            elif self.task == "onehot":
                # handle_missing='return_nan' allows us to accurately build the observed_mask later
                self.encoder = ce.OneHotEncoder(cols=present_cat_vars, use_cat_names=True, handle_missing='return_nan')
                self.encoder.fit(df)
                temp_transformed = self.encoder.transform(df)

                self.onehot_cols = [c for c in temp_transformed.columns if c not in self.num_vars]
                self.num_cate_list = [2] * len(self.onehot_cols)  # 2 categories for binary columns (0 and 1)
                self.transformed_columns = self.num_vars + self.onehot_cols
        else:
            self.transformed_columns = self.num_vars

        return self

    def transform(self, df):
        out_df = df.copy()

        # Apply categorical encoder
        if self.encoder is not None:
            out_df = self.encoder.transform(out_df)

        # Apply continuous scaler
        for col in self.num_vars:
            if col in out_df.columns:
                vals = pd.to_numeric(out_df[col], errors='coerce').values
                mask = ~np.isnan(vals)

                min_val = self.mins[col]
                max_val = self.maxs[col]
                out_df[col] = out_df[col].astype(float)
                out_df.loc[mask, col] = (vals[mask] - (min_val - 1.0)) / (max_val - min_val + 1.0)
                out_df[col] = out_df[col].fillna(0.0)

        out_df = out_df[self.transformed_columns]

        # Calculate mask based on the task
        if self.task == "ft":
            # Original logic: check original df for missing values
            observed_mask = ~df[self.transformed_columns].isna().values
        elif self.task == "onehot":
            # One-hot logic: check the expanded one-hot dataframe for NaNs, then fill them with 0 for the model
            observed_mask = ~out_df.isna().values
            out_df = out_df.fillna(0.0)

        return out_df.values.astype(np.float32), observed_mask.astype(np.float32)

    def inverse_transform(self, transformed_data):
        out_df = pd.DataFrame(transformed_data, columns=self.transformed_columns)

        # Reverse continuous variables
        for col in self.num_vars:
            if col in out_df.columns:
                min_val = self.mins[col]
                max_val = self.maxs[col]
                out_df[col] = out_df[col] * (max_val - min_val + 1.0) + (min_val - 1.0)

        # Reverse categorical variables
        if self.encoder is not None:
            if self.task == "ft":
                out_df = self.encoder.inverse_transform(out_df)

            elif self.task == "onehot":
                present_cat_vars = [
                    col for col in self.cat_vars
                    if any(c.startswith(f"{col}_") for c in self.onehot_cols)
                ]
                for cat_var in present_cat_vars:
                    cat_cols = [c for c in self.onehot_cols if c.startswith(f"{cat_var}_")]
                    if cat_cols:
                        max_cols = out_df[cat_cols].idxmax(axis=1)
                        out_df[cat_cols] = 0
                        for row_index, max_col_name in max_cols.items():
                            out_df.at[row_index, max_col_name] = 1
                out_df = self.encoder.inverse_transform(out_df)

        return out_df

    def get_model_metadata(self):
        cont_indices = list(range(len(self.num_vars)))
        return cont_indices, self.num_cate_list