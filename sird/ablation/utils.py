import pandas as pd
import numpy as np


def process_data(filepath, data_info):
    # Load data and remove unnamed index columns
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Retrieve variable configurations
    p1_vars = data_info.get('phase1_vars', [])
    p2_vars = data_info.get('phase2_vars', [])
    cat_vars_set = set(data_info.get('cat_vars', []))

    # Validate that phase variables are paired correctly
    if len(p1_vars) != len(p2_vars):
        raise ValueError("Phase 1 and Phase 2 variable lists must be of equal length.")

    # Identify auxiliary variables (those not in phase 1 or phase 2)
    reserved_vars = set(p1_vars) | set(p2_vars)
    all_vars = df.columns.tolist()
    aux_vars = sorted([c for c in all_vars if c not in reserved_vars])

    processed_data_list = []
    processed_mask_list = []
    variable_schema = []

    current_col_idx = 0
    normalization_stats = {}

    # Extract weights for model training
    weights = df[data_info.get('weight_var')].values

    def append_data(name, data, mask, v_type, num_classes=1, **kwargs):
        """
        Helper function to append processed data blocks and update schema.
        """
        nonlocal current_col_idx
        width = data.shape[1]
        processed_data_list.append(data)
        processed_mask_list.append(mask)

        schema_entry = {
            'name': name,
            'type': v_type,
            'num_classes': num_classes,
            'start_idx': current_col_idx,
            'end_idx': current_col_idx + width
        }
        # Add pair information to schema if provided
        schema_entry.update(kwargs)
        variable_schema.append(schema_entry)
        current_col_idx += width

    # Process paired variables
    for p1_name, p2_name in zip(p1_vars, p2_vars):
        if p1_name not in df.columns or p2_name not in df.columns:
            raise ValueError(f"Missing pair: {p1_name} or {p2_name}")

        p1_raw = df[p1_name].values
        p2_raw = df[p2_name].values

        # Create missing value masks (1 = present, 0 = missing)
        m1 = (~df[p1_name].isna()).values.astype(float).reshape(-1, 1)
        m2 = (~df[p2_name].isna()).values.astype(float).reshape(-1, 1)

        is_categorical = (p2_name in cat_vars_set) or (p1_name in cat_vars_set)

        if is_categorical:
            # Unite categories from both phases to ensure consistent encoding
            u1 = pd.unique(p1_raw[pd.notna(p1_raw)])
            u2 = pd.unique(p2_raw[pd.notna(p2_raw)])
            unique_set = set(u1) | set(u2)
            master_categories = sorted(list(unique_set))
            cat_to_int = {val: i for i, val in enumerate(master_categories)}
            K = max(len(master_categories), 1)

            # Map values to integer indices
            idx1 = np.array([cat_to_int.get(x, 0) for x in p1_raw])
            idx2 = np.array([cat_to_int.get(x, 0) for x in p2_raw])

            # One-hot encoding
            d1_ohe = np.eye(K)[idx1]
            d2_ohe = np.eye(K)[idx2]

            # Zero out rows where data is missing
            d1_ohe[m1.flatten() == 0] = 0
            d2_ohe[m2.flatten() == 0] = 0

            # Store stats and categories
            normalization_stats[p2_name] = {'type': 'categorical', 'categories': np.array(master_categories)}
            normalization_stats[p1_name] = normalization_stats[p2_name]

            append_data(p1_name, d1_ohe, m1, 'categorical_p1', K, pair_partner=p2_name)
            append_data(p2_name, d2_ohe, m2, 'categorical_p2', K, pair_partner=p1_name)
        else:
            # Numeric processing
            v1_float = p1_raw.astype(float)
            v2_float = p2_raw.astype(float)

            # Calculate shift to ensure positivity before log transform
            combined_min = np.nanmin(np.concatenate([v1_float, v2_float]))
            shift = 0.0
            if combined_min <= 0:
                shift = abs(combined_min) + 1.0

            v1_log = np.log1p(v1_float + shift)
            v2_log = np.log1p(v2_float + shift)

            # Calculate mean and std using valid data from both phases
            valid_p1_log = v1_log[m1.flatten() == 1]
            valid_p2_log = v2_log[m2.flatten() == 1]
            combined_logs = np.concatenate([valid_p1_log, valid_p2_log])

            mu, sigma = np.mean(combined_logs), np.std(combined_logs)
            if sigma < 1e-6: sigma = 1.0

            # Z-score normalization
            d1 = (v1_log - mu) / sigma
            d2 = (v2_log - mu) / sigma

            # Handle NaNs and reshape
            d1 = np.nan_to_num(d1, nan=0.0).reshape(-1, 1)
            d2 = np.nan_to_num(d2, nan=0.0).reshape(-1, 1)

            normalization_stats[p2_name] = {'type': 'numeric', 'mu': mu, 'sigma': sigma, 'shift': shift}
            normalization_stats[p1_name] = normalization_stats[p2_name]

            append_data(p1_name, d1, m1, 'numeric_p1', 1, pair_partner=p2_name)
            append_data(p2_name, d2, m2, 'numeric_p2', 1, pair_partner=p1_name)

    # Process auxiliary variables
    for aux_name in aux_vars:
        raw_vals = df[aux_name].values
        mask = (~df[aux_name].isna()).values.astype(float).reshape(-1, 1)
        is_categorical = aux_name in cat_vars_set

        if is_categorical:
            u_vals = pd.unique(raw_vals[pd.notna(raw_vals)])
            master_categories = sorted(list(u_vals))
            cat_to_int = {val: i for i, val in enumerate(master_categories)}
            K = max(len(master_categories), 1)

            indices = np.array([cat_to_int.get(x, 0) for x in raw_vals])
            d_aux = np.eye(K)[indices]
            d_aux[mask.flatten() == 0] = 0

            normalization_stats[aux_name] = {'type': 'categorical', 'categories': np.array(master_categories)}
            append_data(aux_name, d_aux, mask, 'categorical_aux', K)
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

            normalization_stats[aux_name] = {'type': 'numeric', 'mu': mu, 'sigma': sigma, 'shift': shift}
            append_data(aux_name, d_aux, mask, 'numeric_aux')

    # Concatenate all processed data
    final_data = np.hstack(processed_data_list)
    final_mask = np.hstack(processed_mask_list)

    return (final_data, final_mask, weights, variable_schema, normalization_stats, df)


def inverse_transform_data(generated_data, normalization_stats, variable_schema, data_info):
    """
    Reconstructs Phase 2 variables from generated data matrix.
    Assumes generated_data contains only the relevant Phase 2 columns sequentially.
    """
    reconstructed_df = pd.DataFrame()
    p2_vars = data_info['phase2_vars']

    current_idx = 0

    for p2_name in p2_vars:
        var_schema = next(v for v in variable_schema if v['name'] == p2_name)
        stats = normalization_stats[p2_name]

        if 'categorical' in var_schema['type']:
            width = var_schema['num_classes']
            # Extract relevant columns for this variable
            col_data = generated_data[:, current_idx: current_idx + width]
            current_idx += width

            # Decode one-hot encoding
            indices = np.argmax(col_data, axis=1)
            categories = stats['categories']

            if len(categories) > 0:
                reconstructed_df[p2_name] = categories[indices]
            else:
                reconstructed_df[p2_name] = indices
        else:
            # Numeric reconstruction
            col_data = generated_data[:, current_idx]
            current_idx += 1

            mu, sigma, shift = stats['mu'], stats['sigma'], stats['shift']
            val_log = col_data * sigma + mu
            final_val = np.expm1(val_log) - shift
            reconstructed_df[p2_name] = final_val

    return reconstructed_df