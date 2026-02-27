pacman::p_load(progress, torch, mclust)

create_data_loaders <- function(data_original, data_encode, data_info, params, 
                                phase1_rows, phase2_rows, p1set, p2set, 
                                p1b_t, p2b_t, epochs){
  
  if (params$balancebatch){
    phase1_bins <- data_info$cat_vars[!(data_info$cat_vars %in% data_info$phase2_vars)] 
    phase1_bins <- if (length(phase1_bins) > 0) {
      phase1_bins[sapply(phase1_bins, function(col) {
        length(unique(data_original[phase1_rows, col])) > 1 & length(unique(data_original[phase2_rows, col])) > 1
      })]
    } else {
      character(0)
    }
    
    p1sampler <- BalancedSampler(data_original[phase1_rows,], phase1_bins, p1b_t, epochs)
    p2sampler <- BalancedSampler(data_original[phase2_rows,], phase1_bins, p2b_t, epochs)
    p1loader <- dataloader(p1set, sampler = p1sampler, pin_memory = T,
                           collate_fn = function(bl) bl[[1]])
    p2loader <- dataloader(p2set, sampler = p2sampler, pin_memory = T,
                           collate_fn = function(bl) bl[[1]])
    Dloader <- dataloader(p2set, 
                          sampler = BalancedSampler(data_original[phase2_rows,], 
                                                    phase1_bins, params$batch_size, epochs), pin_memory = T,
                          collate_fn = function(bl) bl[[1]])
  } else {
    phase1_bins <- NULL
    p1sampler <- SRSSampler(length(phase1_rows), p1b_t, epochs)
    p2sampler <- SRSSampler(length(phase2_rows), p2b_t, epochs)
    p1loader <- dataloader(p1set, sampler = p1sampler, pin_memory = T,
                           collate_fn = function(bl) bl[[1]])
    p2loader <- dataloader(p2set, sampler = p2sampler, pin_memory = T,
                           collate_fn = function(bl) bl[[1]])
    Dloader <- dataloader(p2set, 
                          sampler = SRSSampler(length(phase2_rows), params$batch_size, epochs),
                          pin_memory = T, collate_fn = function(bl) bl[[1]])
  }
  
  list(p1loader = p1loader, p2loader = p2loader, Dloader = Dloader, phase1_bins = phase1_bins)
}

sicg_default <- function(batch_size = 500, lambda = 10, 
                            alpha = 0, beta = 1, proj_weight = 1,
                            at_least_p = 1, g_dropout = 0.5,
                            lr_g = 2e-4, lr_d = 2e-4, g_betas = c(0.5, 0.9), d_betas = c(0.5, 0.9), 
                            g_weight_decay = 1e-6, d_weight_decay = 1e-6, noise_dim = 64, cond_dim = 128,
                            g_dim = c(256, 256), d_dim = c(256, 256), pac = 2, discriminator_steps = 1,
                            tau = 0.2, hard = F, type_g = "mlp", type_d = "mlp",
                            num = "mmer", cat = "projp1", balancebatch = T, mi_approx = "dropout"){
  
  batch_size <- pac * round(batch_size / pac)
  
  if (mi_approx == "dropout" & g_dropout == 0){
    g_dropout <- 0.5
  }
  
  list(batch_size = batch_size, at_least_p = at_least_p, 
       lambda = lambda, alpha = alpha, beta = beta, proj_weight = proj_weight,
       g_dropout = g_dropout, 
       lr_g = lr_g, lr_d = lr_d, g_betas = g_betas, d_betas = d_betas, 
       g_weight_decay = g_weight_decay, d_weight_decay = d_weight_decay, 
       noise_dim = noise_dim, cond_dim = cond_dim, g_dim = g_dim, d_dim = d_dim, 
       pac = pac, discriminator_steps = discriminator_steps, 
       tau = tau, hard = hard, type_g = type_g, type_d = type_d, 
       num = num, cat = cat, balancebatch = balancebatch, mi_approx = mi_approx)
}

sicg <- function(data, m = 5, 
                       num.normalizing = "mode", cat.encoding = "onehot", 
                       device = "cpu", epochs = 2000, 
                       params = list(), data_info = list(), 
                       save.step = NULL){
  
  params <- do.call("sicg_default", params)
  device <- torch_device(device)
  
  if (params$at_least_p == 1){
    params$cat <- "general"
  }
  conditions_vars <- names(data)[which(!(names(data) %in% c(data_info$phase1_vars, data_info$phase2_vars)))]
  phase1_rows <- which(is.na(data[, data_info$phase2_vars[1]]))
  phase2_rows <- which(!is.na(data[, data_info$phase2_vars[1]]))
  normalize <- paste("normalize", num.normalizing, sep = ".")
  encode <- paste("encode", cat.encoding, sep = ".")
  
  weights <- as.vector(as.numeric(as.character(data[, names(data) %in% data_info$weight_var])))
  data_original <- data
  if (params$num == "mmer"){
    p2n <- intersect(data_info$phase2_vars, data_info$num_vars)
    p1n <- intersect(data_info$phase1_vars, data_info$num_vars)
    data[p2n] <- Map(function(p1, p2) data[[p1]] - data[[p2]], p1n, p2n)
  }
  
  data_norm <- do.call(normalize, args = list(
    data = data,
    num_vars = data_info$num_vars, 
    c(conditions_vars, data_info$phase1_vars)
  ))
  norm_data <- data_norm$data
  mode_cat_vars <- union(data_info$cat_vars, setdiff(names(norm_data), names(data)))
  phase2_vars_mode <- union(data_info$phase2_vars,
                            names(norm_data)[names(norm_data) %in% 
                                               paste0(data_info$phase2_vars, "_mode")])
  
  data_encode <- do.call(encode, args = list(
    data = norm_data, mode_cat_vars, 
    data_info$cat_vars, data_info$phase1_vars, data_info$phase2_vars
  ))
  nrows <- nrow(data_encode$data)
  ncols <- ncol(data_encode$data)
  data_training <- data_encode$data
  
  phase1_vars_encode <- c(setdiff(data_info$phase1_vars, mode_cat_vars),
                          unlist(data_encode$new_col_names[data_info$phase1_vars]))
  phase2_vars_encode <- c(setdiff(phase2_vars_mode, mode_cat_vars),
                          unlist(data_encode$new_col_names[phase2_vars_mode]))
  conditions_vars_encode <- c(setdiff(conditions_vars, mode_cat_vars),
                              unlist(data_encode$new_col_names[conditions_vars]))
  
  num_inds_p2 <- which(phase2_vars_encode %in% data_info$num_vars)
  cat_inds_p2 <- which(phase2_vars_encode %in% unlist(data_encode$new_col_names))
  
  new_order <- c(phase2_vars_encode[num_inds_p2], 
                 phase2_vars_encode[cat_inds_p2],
                 phase1_vars_encode[which(phase1_vars_encode %in% data_info$num_vars)], 
                 phase1_vars_encode[which(phase1_vars_encode %in% unlist(data_encode$new_col_names))],
                 conditions_vars_encode[which(conditions_vars_encode %in% data_info$num_vars)],
                 conditions_vars_encode[which(conditions_vars_encode %in% unlist(data_encode$new_col_names))],
                 setdiff(names(data_training), 
                         c(phase2_vars_encode, phase1_vars_encode,
                           conditions_vars_encode)))
  
  data_training <- data_training[, new_order]
  data_encode$binary_indices <- 
    lapply(data_encode$binary_indices, 
           function(idx) match(names(data_encode$data)[idx], names(data_training)))
  
  # Pre-load tensors to GPU (Static Data)
  conditions_t <- torch_tensor(as.matrix(data_training[, conditions_vars_encode, drop = F]), 
                               device = device)
  phase2_m <- data_training[, phase2_vars_encode, drop = F]
  data_mask <- torch_tensor(1 - is.na(phase2_m), dtype = torch_long(), device = device)
  phase2_m[is.na(phase2_m)] <- 0 
  phase2_t <- torch_tensor(as.matrix(phase2_m), device = device)
  
  phase1_m <- data_training[, phase1_vars_encode, drop = F]
  phase1_t <- torch_tensor(as.matrix(phase1_m), device = device)
  
  tensor_list <- list(data_mask, conditions_t, phase2_t, phase1_t)
  
  phase1_cats <- intersect(data_info$phase1_vars, data_info$cat_vars)
  phase2_cats <- intersect(data_info$phase2_vars, data_info$cat_vars)
  
  cats_p1 <- relist(match(unlist(data_encode$new_col_names[phase1_cats]), 
                          names(phase1_m)), 
                    skeleton = data_encode$new_col_names[phase1_cats])
  
  nc <- data_encode$new_col_names
  bi <- data_encode$binary_indices
  
  idx_map <- setNames(rep(seq_along(data_encode$new_col_names), 
                          lengths(data_encode$new_col_names)), 
                      unlist(data_encode$new_col_names, use.names = FALSE))
  bins_by_enc <- function(enc) { 
    i <- idx_map[enc]
    i <- i[!is.na(i) & !duplicated(i)]
    data_encode$binary_indices[i] 
  }
  
  allnums <- which(names(data_training) %in% data_info$num_vars)
  allcats <- data_encode$binary_indices
  allcats_p2 <- bins_by_enc(phase2_vars_encode)
  cats_mode <- bins_by_enc(setdiff(phase2_vars_encode, 
                                   unlist(data_encode$new_col_names[phase2_cats], use.names = FALSE)))
  i_order <- idx_map[phase2_vars_encode]; i_order <- i_order[!is.na(i_order) & !duplicated(i_order)]
  cats_p2 <- data_encode$binary_indices[i_order[names(data_encode$binary_indices)[i_order] %in% phase2_cats]]
  
  # --- 1. PRE-COMPUTE GROUPS ---
  CM_tensors <- list()
  proj_groups <- list()
  
  if (length(phase2_cats) > 0 && params$cat == "projp1"){
    ind1 <- match(phase1_cats, names(data_norm$data))
    ind2 <- match(phase2_cats, names(data_norm$data))
    confusmat <- lapply(1:length(ind1), function(i){
      lv <- sort(unique(data_norm$data[, ind1[i]]))
      cm <- prop.table(table(factor(data_norm$data[, ind2[i]], levels = lv),
                             factor(data_norm$data[, ind1[i]], levels = lv)), 1)
      cm[is.na(cm)] <- 0 
      return (cm)
    })
    CM_tensors <- lapply(confusmat, function(cm) torch_tensor(cm, dtype = torch_float(), device = device))
    names(CM_tensors) <- phase2_cats
    
    vars_by_dim <- split(names(cats_p2), vapply(cats_p2, length, numeric(1)))
    proj_groups <- lapply(vars_by_dim, function(var_names) {
      L <- length(cats_p2[[var_names[1]]])
      k <- length(var_names)
      matrices <- torch_stack(CM_tensors[var_names])
      indices_list <- cats_p2[var_names]
      flat_indices <- torch_tensor(unlist(indices_list), dtype = torch_long(), device = device)
      list(k = k, L = L, matrices = matrices, indices = flat_indices)
    })
  }
  
  create_cat_groups <- function(cat_indices_list, dev) {
    if (length(cat_indices_list) == 0) return(list())
    lengths_vec <- vapply(cat_indices_list, length, integer(1))
    vars_by_len <- split(names(cat_indices_list), lengths_vec)
    lapply(vars_by_len, function(nms) {
      L <- length(cat_indices_list[[nms[1]]])
      k <- length(nms)
      flat <- torch_tensor(unlist(cat_indices_list[nms]), dtype = torch_long(), device = dev)
      list(k = k, L = L, indices = flat)
    })
  }
  cat_groups_all <- create_cat_groups(allcats, device)
  cat_groups_p2  <- create_cat_groups(cats_p2, device)
  
  if (length(cats_p2) > 0) {
    p2_lengths <- vapply(cats_p2, length, integer(1))
    vars_by_levels <- split(names(cats_p2), p2_lengths)
    ce_groups_p2 <- lapply(vars_by_levels, function(var_names) {
      L <- length(cats_p2[[var_names[1]]])
      k <- length(var_names)
      p2_flat <- torch_tensor(unlist(cats_p2[var_names]), dtype = torch_long(), device = device)
      p1_flat <- if (exists("cats_p1") && length(cats_p1) > 0) {
        p1_list <- cats_p1[var_names]
        if (any(sapply(p1_list, is.null))) NULL else torch_tensor(unlist(p1_list), dtype = torch_long(), device = device)
      } else NULL
      list(k = k, L = L, p2_idx = p2_flat, p1_idx = p1_flat)
    })
  } else {
    ce_groups_p2 <- list()
  }
  
  if (length(cats_mode) > 0) {
    mode_lengths <- vapply(cats_mode, length, integer(1))
    mode_vars_by_levels <- split(names(cats_mode), mode_lengths)
    ce_groups_mode <- lapply(mode_vars_by_levels, function(var_names) {
      L <- length(cats_mode[[var_names[1]]])
      k <- length(var_names)
      idx_flat <- torch_tensor(unlist(cats_mode[var_names]), dtype = torch_long(), device = device)
      list(k = k, L = L, idx = idx_flat)
    })
  } else {
    ce_groups_mode <- list()
  }
  total_cat_count <- length(cats_p2) + length(cats_mode)
  
  D_loader_list <- list()
  p1_loader_list <- list()
  p2_loader_list <- list()
  num_models_to_train <- if (params$mi_approx == "bootstrap") m else 1
  
  p2b_t <- as.integer(params$batch_size * params$at_least_p)
  p1b_t <- params$batch_size - p2b_t
  params$ncols <- ncols
  params$nphase2 <- length(phase2_vars_encode)
  params$nphase1 <- length(phase1_vars_encode)
  params$num_inds <- num_inds_p2
  params$cat_inds <- cat_inds_p2
  
  gnet_list <- list()
  dnet_list <- list()
  g_solver_list <- list()
  d_solver_list <- list()
  training_loss_list <- list()
  
  for (model_idx in 1:num_models_to_train) {
    if (num_models_to_train > 1) {
      b_phase1_rows <- sample(phase1_rows, length(phase1_rows), replace = TRUE)
      b_phase2_rows <- sample(phase2_rows, length(phase2_rows), replace = TRUE)
      
      p1set <- initset(data_training, b_phase1_rows, phase1_vars_encode, 
                       phase2_vars_encode, conditions_vars_encode, device = device)
      p2set <- initset(data_training, b_phase2_rows, phase1_vars_encode, 
                       phase2_vars_encode, conditions_vars_encode, device = device)
      
      curr_loaders <- create_data_loaders(data_original, data_encode, data_info, params, 
                                          b_phase1_rows, b_phase2_rows, p1set, p2set, 
                                          p1b_t, p2b_t, epochs)
    } else {
      p1set <- initset(data_training, phase1_rows, 
                       phase1_vars_encode, phase2_vars_encode, conditions_vars_encode, device = device)
      p2set <- initset(data_training, phase2_rows, 
                       phase1_vars_encode, phase2_vars_encode, conditions_vars_encode, device = device)
      
      curr_loaders <- create_data_loaders(data_original, data_encode, data_info, params, 
                                          phase1_rows, phase2_rows, p1set, p2set, 
                                          p1b_t, p2b_t, epochs)
    }
    
    D_loader <- curr_loaders$Dloader
    p1_loader <- curr_loaders$p1loader
    p2_loader <- curr_loaders$p2loader
    phase1_bins <- curr_loaders$phase1_bins
    
    cat(sprintf("\n--- Training Model %d/%d ---\n", model_idx, num_models_to_train))
    
    gnet <- do.call(paste("generator", params$type_g, sep = "."), 
                    args = list(params))$to(device = device)
    dnet <- do.call(paste("discriminator", params$type_d, sep = "."), 
                    args = list(params))$to(device = device)
    
    if (params$mi_approx != "SWAG"){
      g_solver <- optim_adam(gnet$parameters, lr = params$lr_g, betas = params$g_betas)
      d_solver <- optim_adam(dnet$parameters, lr = params$lr_d, betas = params$d_betas)
    } else {
      # Assumes 25 times larger learning rate for SGD in SWAG
      g_solver <- optim_sgd(gnet$parameters, lr = params$lr_g * 25, momentum = 0)
      d_solver <- optim_sgd(dnet$parameters, lr = params$lr_d * 25, momentum = 0)
      
      swa_start <- as.integer(0.8 * epochs)
      scheduler_g <- lr_cosine_annealing(g_solver, T_max = swa_start, eta_min = params$lr_g * 25 * 0.5)
      scheduler_d <- lr_cosine_annealing(d_solver, T_max = swa_start, eta_min = params$lr_g * 25 * 0.5)
      swa_n <- 0
      max_num_models <- m * 5
      swa_mean <- list()
      swa_sq_mean <- list()
      swa_cov_mat <- list()
    }
    
    training_loss <- matrix(0, nrow = epochs, ncol = 2)
    
    pb <- progress_bar$new(
      format = paste0("Running [:bar] :percent eta: :eta | G Loss: :g_loss | D Loss: :d_loss | Recon: :recon_loss"),
      clear = FALSE, total = epochs, width = 100)
    
    if (p1b_t > 0){
      it_p1 <- dataloader_make_iter(p1_loader)
    }
    it_p2 <- dataloader_make_iter(p2_loader)
    it_D <- dataloader_make_iter(D_loader)
    
    ones_buf <- torch_ones(c(params$batch_size, 1), device = device)
    
    for (i in 1:epochs){
      gnet$train()
      
      # --- Discriminator Step ---
      for (d_step in 1:params$discriminator_steps) {
        batch <- dataloader_next(it_D)
        if (is.null(batch)) {
          it_D <- dataloader_make_iter(D_loader)
          batch <- dataloader_next(it_D)
        }
        
        A <- batch$A; X <- batch$X; C <- batch$C
        fakez <- torch_normal(mean = 0, std = 1, size = c(params$batch_size, params$noise_dim), device = device) 
        X_fake <- gnet(fakez, A, C)
        X_fakeact <- activationFun(X_fake, cat_groups_p2, num_inds_p2, params)
        fake_AC <- torch_cat(list(X_fakeact, A, C), dim = 2)
        true_AC <- torch_cat(list(X, A, C), dim = 2)
        x_fake <- dnet(fake_AC); x_true <- dnet(true_AC)
        
        d_loss <- -(torch_mean(x_true) - torch_mean(x_fake))
        
        if (params$lambda > 0){
          gp <- gradientPenalty(dnet, fake_AC, true_AC, params, device = device) 
          d_loss <- d_loss + params$lambda * gp
        }
        
        d_solver$zero_grad()
        d_loss$backward()
        d_solver$step()
      }
      
      # --- Generator Step ---
      batch_p2 <- dataloader_next(it_p2)
      if (is.null(batch_p2)) {
        it_p2 <- dataloader_make_iter(p2_loader)
        batch_p2 <- dataloader_next(it_p2)
      }
      
      if (p1b_t > 0){
        batch_p1 <- dataloader_next(it_p1)
        if (is.null(batch_p1)) {
          it_p1 <- dataloader_make_iter(p1_loader)
          batch_p1 <- dataloader_next(it_p1)
        }
        A <- torch_cat(list(batch_p1$A, batch_p2$A), dim = 1)
        X <- torch_cat(list(batch_p1$X, batch_p2$X), dim = 1)
        C <- torch_cat(list(batch_p1$C, batch_p2$C), dim = 1)
        M <- torch_cat(list(batch_p1$M, batch_p2$M), dim = 1)
      } else {
        A <- batch_p2$A; X <- batch_p2$X; C <- batch_p2$C; M <- batch_p2$M
      }
      I <- M[, 1] == 1
      
      fakez <- torch_normal(mean = 0, std = 1, size = c(params$batch_size, params$noise_dim), device = device) 
      X_fake <- gnet(fakez, A, C)
      
      fake_proj <- NULL
      if (length(phase2_cats) > 0 && params$cat == "projp1"){
        fake_proj <- projCat(X_fake, proj_groups)
      }
      
      recon_loss <- reconLoss(X_fake, X, fake_proj, A, I, params, num_inds_p2, cat_inds_p2, ce_groups_p2, ce_groups_mode, total_cat_count)
      X_fakeact <- activationFun(X_fake, cat_groups_p2, num_inds_p2, params)
      fake_AC <- torch_cat(list(X_fakeact, A, C), dim = 2)
      d_fake <- dnet(fake_AC)
      adv_term <- -torch_mean(d_fake)
      
      g_loss <- adv_term + recon_loss
      
      g_solver$zero_grad()
      g_loss$backward()
      
      if (params$mi_approx == "SWAG"){
        nn_utils_clip_grad_norm_(gnet$parameters, max_norm = 1)
      }
      g_solver$step()
      
      # --- SWAG / Scheduling Updates ---
      if (params$mi_approx == "SWAG"){
        is_swa_phase <- i > swa_start
        if (!is_swa_phase) {
          scheduler_g$step()
          scheduler_d$step()
        } else {
          swa_n <- swa_n + 1
          curr_state <- gnet$state_dict()
          for (key in names(curr_state)) {
            val <- curr_state[[key]]
            if (inherits(val, "torch_tensor") && val$dtype == torch_float()) {
              p_data <- val$detach()$clone()
              if (swa_n == 1) {
                swa_mean[[key]] <- p_data
                swa_sq_mean[[key]] <- p_data^2
                swa_cov_mat[[key]] <- list() 
              } else {
                swa_mean[[key]] <- (swa_mean[[key]] * (swa_n - 1) + p_data) / swa_n
                swa_sq_mean[[key]] <- (swa_sq_mean[[key]] * (swa_n - 1) + p_data^2) / swa_n
                dev <- p_data - swa_mean[[key]]
                q <- swa_cov_mat[[key]]
                q[[length(q) + 1]] <- dev
                if (length(q) > max_num_models) q <- q[-1] 
                swa_cov_mat[[key]] <- q
              }
            } else {
              swa_mean[[key]] <- val$detach()$clone()
            }
          }
        }
      }
      
      training_loss[i, ] <- c(g_loss$item(), d_loss$item())
      pb$tick(tokens = list(
        g_loss = sprintf("%.3f", adv_term$item()),
        d_loss = sprintf("%.3f", d_loss$item()),
        recon_loss = sprintf("%.3f", recon_loss$item())
      ))
    }
    pb$terminate()
    
    training_loss <- data.frame(training_loss)
    names(training_loss) <- c("G Loss", "D Loss")
    training_loss_list[[model_idx]] <- training_loss
    gnet_list[[model_idx]] <- gnet
  }
  
  # --- Evaluation / Generation ---
  gnet_list <- lapply(gnet_list, function(m) {
    m$eval()
    if (params$mi_approx == "dropout"){
      for (modu in m$modules) {
        if (inherits(modu, "nn_dropout")) {
          modu$train(TRUE)
        }
      }
    }
    return(m)
  })
  
  if (params$mi_approx == "SWAG"){
    swag_stats <- list(mean = swa_mean, sq_mean = swa_sq_mean, cov_mat = swa_cov_mat, n = swa_n)
  } else {
    swag_stats <- NULL
  }
  
  result <- generateImpute(gnet_list, m = m, 
                           data_original, data_info, data_norm, 
                           data_encode, data_training,
                           phase1_vars_encode, phase2_vars_encode,
                           num.normalizing, cat.encoding, 
                           device, params, cat_groups_p2, num_inds_p2, tensor_list,
                           swag_stats)
  
  out <- list(imputation = result$imputation,
              gsample = result$gsample,
              loss = training_loss_list)
  if (exists("step_result", inherits = FALSE) && !is.null(step_result)){
    out$step_result <- step_result
  }
  return (out)
}