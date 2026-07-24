
pmm <- function(yhatobs, yhatmis, yobs, k) {
  idx <- mice::matchindex(d = yhatobs, t = yhatmis, k = k)
  yobs[idx]
}

create_bfi <- function(data_nrow, batch_size, tensor_list){
  n <- ceiling(data_nrow / batch_size)
  batches <- vector("list", n)
  idx <- 1
  for (i in 1:n){
    end <- min(idx + batch_size - 1, data_nrow)
    batches[[i]] <- lapply(tensor_list, function(tensor) tensor[idx:end, , drop = FALSE])
    idx <- idx + batch_size
  }
  return(batches)
}

sample_full_swag_weights <- function(swag_stats, scale = 1) {
  sampled_state <- list()
  mean_dict <- swag_stats$mean
  sq_mean_dict <- swag_stats$sq_mean
  cov_mat_dict <- swag_stats$cov_mat
  
  for (key in names(mean_dict)) {
    mu <- mean_dict[[key]]
    
    if (inherits(mu, "torch_tensor") && mu$dtype == torch_float()) {
      sq_mu <- sq_mean_dict[[key]]
      var_val <- torch_clamp(torch_abs(sq_mu - mu^2), min = 1e-30)
      sigma <- torch_sqrt(var_val)
      z1 <- torch_randn_like(mu)
      
      term_diag <- (scale / sqrt(2)) * sigma * z1
      dev_queue <- cov_mat_dict[[key]]
      K <- length(dev_queue)
      
      term_lr <- torch_zeros_like(mu)
      
      if (K > 1) {
        z2 <- torch_randn(K, device = mu$device)
        for (k_idx in 1:K) {
          term_lr <- term_lr + dev_queue[[k_idx]] * z2[k_idx]
        }
        term_lr <- term_lr * (scale / sqrt(2 * (K - 1)))
      }
      sampled_state[[key]] <- mu + term_diag + term_lr
      
    } else {
      sampled_state[[key]] <- mu
    }
  }
  return(sampled_state)
}

acc_prob_row <- function(mat, lb, ub, alpha = 1) {
  nr <- nrow(mat)
  UB <- matrix(ub, nr, length(ub), byrow = TRUE)
  LB <- matrix(lb, nr, length(lb), byrow = TRUE)
  span <- matrix((ub - lb) / 2, nr, length(lb), byrow = TRUE)
  
  hi <- pmax(mat - UB, 0)
  lo <- pmax(LB - mat, 0)
  d <- pmax(hi, lo) / span
  
  total_violation <- rowSums(d)
  exp(-alpha * total_violation ^ 2)
}

generateImpute <- function(gnet_list, m = 5, 
                           data_original, data_info, data_norm, 
                           data_encode, data_training,
                           phase1_vars, phase2_vars,
                           num.normalizing, cat.encoding, 
                           device, params, 
                           cat_groups_p2, num_inds_p2,
                           tensor_list, swag_stats){
  
  imputed_data_list <- vector("list", m)
  gsample_data_list <- vector("list", m)
  
  denormalize <- paste("denormalize", num.normalizing, sep = ".")
  decode <- paste("decode", cat.encoding, sep = ".")
  batchforimpute <- create_bfi(nrow(data_original), params$batch_size, tensor_list)
  data_mask <- as.matrix(1 - is.na(data_original))
  
  p2_num <- intersect(phase2_vars, data_info$num_vars)
  lb <- vapply(p2_num, function(v) min(data_original[[v]], na.rm = TRUE), numeric(1))
  ub <- vapply(p2_num, function(v) max(data_original[[v]], na.rm = TRUE), numeric(1))
  names(ub) <- names(lb) <- p2_num
  max_attempt <- 5
  
  num_col <- names(data_original) %in% data_info$num_vars
  suppressWarnings(data_original[is.na(data_original)] <- 0)
  
  out_names  <- names(data_original)
  p2_idx_out <- sort(match(data_info$phase2_vars, out_names))
  p2_num_idx_out <- match(p2_num, out_names[p2_idx_out])
  num_inds_gen <- match(phase2_vars[phase2_vars %in% data_info$num_vars], names(data_original))
  num_inds_ori <- match(phase1_vars[phase1_vars %in% data_info$num_vars], names(data_original))

  M_t <- tensor_list[[1]]
  C_t <- tensor_list[[2]]
  X_t <- tensor_list[[3]]
  A_t <- tensor_list[[4]]

  process_generated_batch <- function(gen_raw, A_curr, C_curr, rows_idx = NULL) {
    gen_act <- activationFun(gen_raw, cat_groups_p2, num_inds_p2, params, gen = T)
    
    gen_combined <- torch_cat(list(gen_act, A_curr, C_curr), dim = 2)
    df_raw <- as.data.frame(as.matrix(gen_combined$detach()$cpu()))
    names(df_raw) <- names(data_training)
    df_decoded <- do.call(decode, args = list(data = df_raw, encode_obj = data_encode))
    df_final <- do.call(denormalize, args = list(
      data = df_decoded,
      num_vars = data_info$num_vars, 
      norm_obj = data_norm
    ))$data
    
    df_final <- df_final[, names(data_original)]
    if (params$num == "mmer") {
      ref_data <- if (is.null(rows_idx)) data_original else data_original[rows_idx, ]
      df_final[, num_inds_gen] <- ref_data[, num_inds_ori] - df_final[, num_inds_gen]
    }
    return(df_final)
  }
  
  for (z in 1:m){
    if (params$mi_approx == "bootstrap"){
      gnet <- gnet_list[[z]]
    } else {
      gnet <- gnet_list[[1]]
    }
    if (params$mi_approx == "SWAG"){
      new_weights <- sample_full_swag_weights(swag_stats, scale = 1)
      gnet$load_state_dict(new_weights)
    }
    output_df_list <- vector("list", length(batchforimpute))
    
    for (i in seq_along(batchforimpute)){
      batch <- batchforimpute[[i]]
      M <- batch[[1]]; C <- batch[[2]]; X <- batch[[3]]; A <- batch[[4]]
      fakez <- torch_normal(mean=0, std=1, size=c(X$size(1), params$noise_dim), device=device)
      gsample <- gnet(fakez, A, C)
      
      gsample <- activationFun(gsample, cat_groups_p2, num_inds_p2, params, gen = T)
      output_df_list[[i]] <- torch_cat(list(gsample, A, C), dim = 2)
    }
    output_mat <- torch_cat(output_df_list)
    output_frame <- as.data.frame(as.matrix(output_mat$detach()$cpu()))
    names(output_frame) <- names(data_training)
    
    curr_gsample <- do.call(decode, args = list(data = output_frame, encode_obj = data_encode))
    curr_gsample <- do.call(denormalize, args = list(
      data = curr_gsample,
      num_vars = data_info$num_vars, 
      norm_obj = data_norm
    ))
    gsamples <- curr_gsample$data[, names(data_original)]
    if (params$num == "mmer"){
      gsamples[, num_inds_gen] <- data_original[, num_inds_ori] - gsamples[, num_inds_gen]
    }
    M_out <- gsamples[, p2_idx_out, drop = FALSE]
    row_acc <- acc_prob_row(as.matrix(M_out[, p2_num_idx_out, drop = FALSE]), lb, ub)
    accept_row <- runif(nrow(M_out)) < row_acc
    
    iter <- 0L
    while (iter < max_attempt & any(!accept_row)) {
      oob_rows <- which(!accept_row)
      n_oob <- length(oob_rows)
      idx_oob <- torch_tensor(oob_rows, dtype = torch_long(), device = device)
      A_sub <- A_t[idx_oob]; C_sub <- C_t[idx_oob]
      
      fakez <- torch_normal(mean=0, std=1, size=c(n_oob, params$noise_dim), device=device)
      new_samp <- gnet(fakez, A_sub, C_sub)
      new_df <- process_generated_batch(new_samp, A_sub, C_sub, rows_idx = oob_rows)
      M_out[oob_rows, ] <- new_df[, p2_idx_out, drop = FALSE]
      gsamples[oob_rows, ] <- new_df
      row_acc <- acc_prob_row(as.matrix(M_out[, p2_num_idx_out, drop = FALSE]), lb, ub)
      accept_row <- runif(nrow(M_out)) < row_acc
      iter <- iter + 1L
    }
    gsamples[, p2_idx_out] <- M_out
    for (v in names(lb)){
      oob_ind <- which(gsamples[[v]] < lb[[v]] | gsamples[[v]] > ub[[v]])
      if (length(oob_ind) > 0){
        yhatmis <- pmm(gsamples[[v]][data_original$R == 1],
                       gsamples[[v]][oob_ind],
                       data_original[[v]][data_original$R == 1], 5)
        gsamples[[v]][oob_ind] <- yhatmis
      }
    }
    
    imputations <- data_original
    if (any(num_col)) {
      out_num <- data_mask[, num_col] * as.matrix(data_original[, num_col]) +
        (1 - data_mask[, num_col]) * as.numeric(as.matrix(gsamples[, num_col]))
      imputations[, num_col] <- out_num
    }
    if (any(!num_col)) {
      fac_cols <- which(!num_col)
      for (j in fac_cols) {
        new_lvls <- unique(as.character(gsamples[, j]))
        levels(imputations[[j]]) <- union(levels(imputations[[j]]), new_lvls)
      }
      to_replace <- data_mask[, fac_cols] == 0
      idx <- which(to_replace, arr.ind = TRUE)
      if (nrow(idx) > 0){
        col_ids <- fac_cols[idx[, "col"]]
        imputations[cbind(idx[, "row"], col_ids)] <- gsamples[cbind(idx[, "row"], col_ids)]
      }
    }
    imputed_data_list[[z]] <- imputations
    gsample_data_list[[z]] <- gsamples
  }
  
  return (list(imputation = imputed_data_list, gsample = gsample_data_list))
}