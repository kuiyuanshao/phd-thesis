# Scores an imputed dataset vs truth: z-scored MSE, categorical error, energy distance.
create_loss_calculator <- function(data_info) {
  target_num <- intersect(data_info$num_vars, data_info$phase2_vars)
  target_cat <- intersect(data_info$cat_vars, data_info$phase2_vars)

  calc <- function(true_data, fake_data) {
    true_data <- as.data.frame(true_data)
    fake_data <- as.data.frame(fake_data)

    raw_mse <- 0.0
    if (length(target_num) > 0) {
      t_num <- as.matrix(sapply(true_data[, target_num, drop=FALSE], as.numeric))
      f_num <- as.matrix(sapply(fake_data[, target_num, drop=FALSE], as.numeric))

      means <- colMeans(t_num, na.rm=TRUE)
      stds <- apply(t_num, 2, sd, na.rm=TRUE)
      stds[is.na(stds) | stds == 0] <- 1e-8

      t_scaled <- scale(t_num, center=means, scale=stds)
      f_scaled <- scale(f_num, center=means, scale=stds)

      raw_mse <- mean((t_scaled - f_scaled)^2, na.rm=TRUE)
      if(is.na(raw_mse)) raw_mse <- 0.0
    }

    raw_ce <- 0.0
    if (length(target_cat) > 0) {
      error_rate_sum <- 0.0
      for (col in target_cat) {
        t_col <- as.character(true_data[[col]])
        f_col <- as.character(fake_data[[col]])
        error_rate_sum <- error_rate_sum + mean(t_col != f_col, na.rm=TRUE)
      }
      raw_ce <- error_rate_sum / length(target_cat)
    }

    combined <- rbind(true_data, fake_data[, colnames(true_data)])
    num_vars <- data_info$num_vars
    cat_vars <- data_info$cat_vars

    if (length(num_vars) > 0) {
      for(col in num_vars) combined[[col]] <- as.numeric(combined[[col]])
      combined[, num_vars] <- scale(combined[, num_vars])
      combined[, num_vars][is.na(combined[, num_vars])] <- 0
    }

    if (length(cat_vars) > 0) {
      dummy_list <- lapply(cat_vars, function(col) {
        f_val <- as.factor(combined[[col]])
        if (length(levels(f_val)) < 2) {
          mat <- matrix(1, nrow = nrow(combined), ncol = 1)
          colnames(mat) <- paste0(col, "_", levels(f_val)[1])
          return(mat)
        } else {
          df_tmp <- data.frame(f = f_val)
          mat <- model.matrix(~ 0 + f, data = df_tmp)
          colnames(mat) <- paste0(col, "_", levels(f_val))
          return(mat)
        }
      })
      cat_matrix <- do.call(cbind, dummy_list)

      if (length(num_vars) > 0) {
        final_mat <- cbind(as.matrix(combined[, num_vars, drop=FALSE]), cat_matrix)
      } else {
        final_mat <- cat_matrix
      }
    } else {
      final_mat <- as.matrix(combined[, num_vars, drop=FALSE])
    }

    n_true <- nrow(true_data)
    X <- final_mat[1:n_true, , drop=FALSE]
    Y <- final_mat[(n_true+1):nrow(final_mat), , drop=FALSE]

    dist_XY <- mean(fields::rdist(X, Y))
    dist_XX <- mean(fields::rdist(X, X))
    dist_YY <- mean(fields::rdist(Y, Y))

    raw_ed <- max((2 * dist_XY) - dist_XX - dist_YY, 0.0)

    n_num <- length(target_num)
    n_cat <- length(target_cat)
    n_total <- n_num + n_cat

    prop_num <- if(n_total > 0) n_num / n_total else 0.0
    prop_cat <- if(n_total > 0) n_cat / n_total else 0.0

    final_mse <- raw_mse * prop_num
    final_ce  <- raw_ce * prop_cat
    final_ed  <- raw_ed

    return(list(
      weighted_mse = final_mse,
      weighted_ce = final_ce,
      weighted_ed = final_ed
    ))
  }
  return(calc)
}
