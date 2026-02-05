import pandas as pd
import numpy as np
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

            combined_logs = np.concatenate([valid_p1_log, valid_p2_log])
            mu, sigma = np.mean(combined_logs), np.std(combined_logs)
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
