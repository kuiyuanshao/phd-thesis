normalize.mode <- function(data, num_vars, cond_vars) {
  data_norm <- data
  mode_params <- list()
  for (col in num_vars) {
    curr_col <- data[[col]]
    curr_col_obs <- curr_col[!is.na(curr_col)]
    if (length(unique(curr_col_obs)) == 1 | col %in% cond_vars) {
      curr_col_norm <- rep(NA, length(curr_col))
      mu <- mean(curr_col)
      sigma <- sd(curr_col)
      if (is.na(sigma) | sigma == 0){
        curr_col_norm <- (curr_col - mu)
      }else{
        curr_col_norm <- (curr_col - mu) / sigma
      }
      data_norm[[col]] <- curr_col_norm
      mode_params[[col]] <- list(mode_means = mu, mode_sds = sigma)
      next
    } 
    mc <- mclust::Mclust(curr_col_obs, G = 1:9, verbose = F, modelNames = "V")
    pred <- predict(mc, newdata = curr_col_obs)
    mode_labels <- pred$classification
    
    mode_means <- mc$parameters$mean + 1e-6
    mode_sds <- sqrt(mc$parameters$variance$sigmasq) + 1e-6
    
    if (length(mode_sds) != length(mode_means)){
      mode_sds <- rep(mode_sds, length(mode_means))
    }
    labs <- as.character(unique(mode_labels))
    names(mode_means) <- names(mode_sds) <- as.character(seq_along(mode_means))
    mode_means <- mode_means[labs]
    mode_sds <- mode_sds[labs]
    
    curr_col_norm <- rep(NA, length(curr_col_obs))
    for (mode in sort(unique(mode_labels))) {
      idx <- which(mode_labels == mode)
      mode <- as.character(mode)
      if (is.na(mode_sds[mode]) | mode_sds[mode] == 0){
        curr_col_norm[idx] <- (curr_col_obs[idx] - mode_means[mode])
      }else{
        curr_col_norm[idx] <- (curr_col_obs[idx] - mode_means[mode]) / (mode_sds[mode])
      }
    }
    mode_labels_curr_col <- rep(NA, length(curr_col))
    mode_labels_curr_col[!is.na(curr_col)] <- mode_labels
    curr_col[!is.na(curr_col)] <- curr_col_norm
    data_norm[[col]] <- curr_col
    if (length(unique(mode_labels)) > 1){
      data_norm[[paste0(col, "_mode")]] <- mode_labels_curr_col
    }
    mode_params[[col]] <- list(mode_means = mode_means, mode_sds = mode_sds)
  }
  return(list(data = data_norm, mode_params = mode_params))
}

denormalize.mode <- function(data, num_vars, norm_obj){
  num_vars <- num_vars[num_vars %in% names(data)]
  mode_params <- norm_obj$mode_params
  for (col in num_vars){
    curr_col <- data[[col]]
    curr_labels <- data[[paste0(col, "_mode")]]
    curr_transform <- rep(NA, length(curr_col))
    
    mode_means <- mode_params[[col]][["mode_means"]]
    mode_sds <- mode_params[[col]][["mode_sds"]]
    
    if (length(mode_means) == 1){
      if (!is.na(mode_sds) & mode_sds != 0){
        curr_transform <- curr_col * mode_sds + mode_means
      }else{
        curr_transform <- curr_col + mode_means
      }
    }else {
      for (mode in unique(curr_labels)){# 1:length(unique(mode_means))){
        idx <- which(curr_labels == mode)
        curr_transform[idx] <- curr_col[idx] * mode_sds[mode] + mode_means[mode] 
      }
    }
    data[[col]] <- curr_transform
  }
  data_denorm <- data[, !grepl("_mode$", names(data))]
  return (list(data = data_denorm, data_mode = data))
}



normalize.modepair <- function(data, phase1_vars, phase2_vars, num_vars) {
  data_norm <- data
  mode_params <- list()
  p1_nums <- phase1_vars[phase1_vars %in% num_vars]
  p2_nums <- phase2_vars[phase2_vars %in% num_vars]
  other_nums <- num_vars[!(num_vars %in% c(phase1_vars, phase2_vars))]
  for (i in 1:length(p1_nums)){
    curr_p1 <- data[[p1_nums[i]]]
    curr_p2 <- data[[p2_nums[i]]]
    curr_p2_obs <- curr_p2[!is.na(curr_p2)]
    
    mc <- mclust::Mclust(curr_p1, G = 1:9, verbose = F, modelNames = "V")
    pred <- predict(mc, newdata = curr_p1)
    mode_labels <- as.numeric(as.factor(pred$classification))
    
    mode_means <- mc$parameters$mean + 1e-6
    mode_sds <- sqrt(mc$parameters$variance$sigmasq) + 1e-6
    
    if (length(mode_sds) != length(mode_means)){
      mode_sds <- rep(mode_sds, length(mode_means))
    }
    curr_p1_norm <- curr_p2_norm <- rep(NA, length(curr_p1))
    for (mode in sort(unique(mode_labels))) {
      mode <- as.numeric(mode)
      idx <- which(mode_labels == mode)
      if (is.na(mode_sds[mode]) | mode_sds[mode] == 0){
        curr_p1_norm[idx] <- (curr_p1[idx] - mode_means[mode])
        curr_p2_norm[idx] <- (curr_p2[idx] - mode_means[mode])
      }else{
        curr_p1_norm[idx] <- (curr_p1[idx] - mode_means[mode]) / (mode_sds[mode])
        curr_p2_norm[idx] <- (curr_p2[idx] - mode_means[mode]) / (mode_sds[mode])
      }
    }

    mode_labels_curr_p1 <- rep(NA, length(curr_p1))
    mode_labels_curr_p1 <- mode_labels
    mode_labels_curr_p2 <- rep(NA, length(curr_p2))
    mode_labels_curr_p2[!is.na(curr_p2)] <- mode_labels[!is.na(curr_p2)]

    data_norm[[p1_nums[i]]] <- curr_p1_norm
    data_norm[[p2_nums[i]]] <- curr_p2_norm
    if (length(unique(mode_labels)) > 1){
      data_norm[[paste0(p1_nums[i], "_mode")]] <- mode_labels_curr_p1
      data_norm[[paste0(p2_nums[i], "_mode")]] <- mode_labels_curr_p2
    }
    mode_params[[p1_nums[i]]] <- list(mode_means = mode_means, mode_sds = mode_sds)
    mode_params[[p2_nums[i]]] <- list(mode_means = mode_means, mode_sds = mode_sds)
  }
  
  for (col in other_nums){
    curr_col <- data[[col]]
    curr_col_norm <- rep(NA, length(curr_col))
    mu <- mean(curr_col)
    sigma <- sd(curr_col)
    if (is.na(sigma) | sigma == 0){
      curr_col_norm[idx] <- (curr_col[idx] - mu)
    }else{
      curr_col_norm[idx] <- (curr_col[idx] - mu) / sigma
    }
    data_norm[[col]] <- curr_col_norm
    mode_params[[col]] <- list(mode_means = mu, mode_sds = sigma)
  }
  return(list(data = data_norm, mode_params = mode_params))
}