# utils.py: Adapted for Hybrid D3PM (Discrete) and RDDM (Residual) Pipelines
import pandas as pd
import numpy as np
import math


def process_data(filepath, data_info):
    """
    Reads data and encodes variables for the Unified Diffusion Model.
    """
    df = pd.read_csv(filepath)
    # Remove artifacts from saving index=True
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    p1_vars = data_info['phase1_vars']
    p2_vars = data_info['phase2_vars']

    defined_vars = set(p1_vars + p2_vars)
    if data_info.get('weight_var'):
        defined_vars.add(data_info['weight_var'])

    cat_vars_set = set(data_info.get('cat_vars', []))

    processed_data_list = []
    processed_mask_list = []
    variable_schema = []

    p1_indices = []
    p2_indices = []

    normalization_stats = {}
    current_col_idx = 0

    print(f"\n[Data Processing] Encoding for Unified RDDM/D3PM...")

    for p1_name, p2_name in zip(p1_vars, p2_vars):
        if p2_name not in cat_vars_set:
            # --- NUMERIC (RDDM) ---
            # Joint Normalization
            combined = pd.concat([df[p1_name], df[p2_name]]).dropna()
            mu = combined.mean()
            sigma = combined.std() + 1e-6

            normalization_stats[p2_name] = {'type': 'numeric', 'mu': mu, 'sigma': sigma}

            p1_val = ((df[p1_name] - mu) / sigma).fillna(0).values[:, None]
            p2_val = ((df[p2_name] - mu) / sigma).fillna(0).values[:, None]

            p1_mask = df[p1_name].notna().astype(float).values[:, None]
            p2_mask = df[p2_name].notna().astype(float).values[:, None]

            p1_indices.append(current_col_idx)
            p2_indices.append(current_col_idx + 1)

            variable_schema.append({
                'name': p2_name, 'type': 'numeric', 'dim': 1,
                'p1_col': current_col_idx, 'p2_col': current_col_idx + 1
            })

            processed_data_list.extend([p1_val, p2_val])
            processed_mask_list.extend([p1_mask, p2_mask])
            current_col_idx += 2

        else:
            # --- CATEGORICAL (D3PM) ---
            s1 = df[p1_name]
            s2 = df[p2_name]

            combined_cats = sorted(list(pd.concat([s1, s2]).dropna().unique()))
            n_cats = len(combined_cats)
            cat_to_id = {cat: i for i, cat in enumerate(combined_cats)}

            normalization_stats[p2_name] = {
                'type': 'categorical',
                'categories': combined_cats,
                'num_classes': n_cats,
                'cat_to_id': cat_to_id
            }

            p1_int = s1.map(cat_to_id).fillna(0).astype(float).values[:, None]
            p2_int = s2.map(cat_to_id).fillna(0).astype(float).values[:, None]

            p1_mask = df[p1_name].notna().astype(float).values[:, None]
            p2_mask = df[p2_name].notna().astype(float).values[:, None]

            p1_indices.append(current_col_idx)
            p2_indices.append(current_col_idx + 1)

            variable_schema.append({
                'name': p2_name, 'type': 'categorical', 'num_classes': n_cats,
                'p1_col': current_col_idx, 'p2_col': current_col_idx + 1
            })

            processed_data_list.extend([p1_int, p2_int])
            processed_mask_list.extend([p1_mask, p2_mask])
            current_col_idx += 2

    aux_vars = [c for c in df.columns if c not in defined_vars]

    for aux_name in aux_vars:
        if aux_name not in cat_vars_set:
            val_series = df[aux_name]
            mu = val_series.mean()
            sigma = val_series.std() + 1e-6
            aux_val = ((val_series - mu) / sigma).fillna(0).values[:, None]
            aux_mask = val_series.notna().astype(float).values[:, None]

            processed_data_list.append(aux_val)
            processed_mask_list.append(aux_mask)

            variable_schema.append({'name': aux_name, 'type': 'numeric_aux', 'col': current_col_idx})
            current_col_idx += 1
        else:
            s_aux = df[aux_name]
            cats = sorted(list(s_aux.dropna().unique()))
            n_cats = len(cats)
            cat_to_id = {cat: i for i, cat in enumerate(cats)}

            aux_int = s_aux.map(cat_to_id).fillna(0).astype(float).values[:, None]
            aux_mask = df[aux_name].notna().astype(float).values[:, None]

            processed_data_list.append(aux_int)
            processed_mask_list.append(aux_mask)

            variable_schema.append({
                'name': aux_name, 'type': 'categorical_aux', 'num_classes': n_cats,
                'col': current_col_idx
            })
            current_col_idx += 1

    final_data = np.concatenate(processed_data_list, axis=1)
    final_mask = np.concatenate(processed_mask_list, axis=1)

    weight_idx = None
    if data_info.get('weight_var'):
        w_name = data_info['weight_var']
        w_val = df[w_name].fillna(df[w_name].mean()).values[:, None]
        final_data = np.concatenate([final_data, w_val], axis=1)
        final_mask = np.concatenate([final_mask, np.ones((len(df), 1))], axis=1)
        weight_idx = final_data.shape[1] - 1

    return (final_data, final_mask, np.array(p1_indices), np.array(p2_indices),
            weight_idx, variable_schema, normalization_stats, df)


def inverse_transform_data(processed_data, normalization_stats, data_info):
    """
    Decodes ONLY Phase 2 variables from the model output.
    Assumes 'processed_data' has shape (N_Samples, N_Phase2_Vars)
    ordered exactly as data_info['phase2_vars'].
    """
    reconstructed_cols = {}
    p2_vars = data_info['phase2_vars']

    # We iterate through the variables and pull the corresponding column from processed_data
    # Since processed_data IS ONLY P2 data, the index is simply 0, 1, 2...

    for i, p2_name in enumerate(p2_vars):
        stats = normalization_stats[p2_name]
        col_data = processed_data[:, i]  # Extract the i-th column

        if stats['type'] == 'numeric':
            mu, sigma = stats['mu'], stats['sigma']
            # Inverse Z-score
            reconstructed_cols[p2_name] = col_data * sigma + mu

        else:
            # Categorical decoding
            categories = stats['categories']

            def decode_indices(idx_array):
                idx_safe = np.clip(np.round(idx_array), 0, len(categories) - 1).astype(int)
                return np.array(categories)[idx_safe]

            reconstructed_cols[p2_name] = decode_indices(col_data)

    return pd.DataFrame(reconstructed_cols)