pacman::p_load(mlr3mbo, mlr3, bbotk, paradox, mlr3learners, survival, survey, torch, dplyr, data.table, DiceKriging)

tune_gan <- function(data, data_ori, data_info, target_model, search_space,
                     best_config_path = "best_gan_config.rds",
                     log_path = "gan_tuning_log.csv",
                     n_evals = 20, 
                     m = 3, epochs = 2000, device = "cuda", folds = 4, weights) {
  true_coeffs <- coef(target_model)
  model_formula <- formula(target_model)
  
  is_survey <- inherits(target_model, c("svycoxph", "svyglm"))
  is_cox    <- inherits(target_model, c("svycoxph", "coxph"))
  
  base_design <- if(is_survey) {
    if(inherits(target_model, "svycoxph")) target_model$survey.design else target_model$survey.design
  } else NULL
  
  model_family <- if(inherits(target_model, c("glm", "svyglm"))) family(target_model) else NULL
  
  fold_indices <- split(sample(1:nrow(data)), rep(1:folds, length.out = nrow(data)))
  
  objective_fun = function(xs) {
    xs$g_weight_decay <- xs$d_weight_decay <- xs$weight_decay
    xs$weight_decay <- NULL
    xs$lr_g <- xs$lr_d * xs$g_d_ratio
    xs$g_d_ratio <- NULL
    
    xs$g_width <- xs$common_width
    xs$d_width <- xs$common_width
    xs$g_depth <- xs$common_depth
    xs$d_depth <- xs$common_depth
    
    xs$g_dim <- rep(xs$g_width, xs$g_depth)
    xs$d_dim <- rep(xs$d_width, xs$d_depth)
    
    xs$g_depth <- NULL; xs$g_width <- NULL
    xs$d_depth <- NULL; xs$d_width <- NULL
    
    # Remove the proxy names
    xs$common_width <- NULL
    xs$common_depth <- NULL
    
    base_params <- cwgangp_default(mi_approx = "bootstrap")
    curr_params <- utils::modifyList(base_params, xs)
    
    # --- DYNAMIC ECHO (FIXED) ---
    cat("\n----------------------------------------------------------------\n")
    cat("[Opt] Testing Config:\n")
    for (nm in names(xs)) {
      val <- xs[[nm]]
      if (is.numeric(val) && length(val) == 1) {
        if ((abs(val) < 1e-3 || abs(val) > 1e4) && val != 0) {
          cat(sprintf("   %s: %.2e\n", nm, val))
        } else {
          cat(sprintf("   %s: %s\n", nm, as.character(val)))
        }
      } else {
        cat(sprintf("   %s: %s\n", nm, paste(as.character(val), collapse = ", ")))
      }
    }
    cat("----------------------------------------------------------------\n")
    
    if(curr_params$pac > 1) {
      curr_params$batch_size <- curr_params$pac * round(curr_params$batch_size / curr_params$pac)
    }
    fold_pieces <- vector("list", m) 
    for(i in 1:m) fold_pieces[[i]] <- list()
    all_val_indices <- list()
    
    for (k in 1:folds) {
      val_idx <- fold_indices[[k]]
      all_val_indices[[k]] <- val_idx
      
      train_test_data <- data
      train_test_data[val_idx, data_info$phase2_vars] <- NA
      
      gan_res <- tpvmi_gans(
        data = train_test_data, m = m, data_info = data_info, 
        params = curr_params, epochs = epochs, device = device,
        num.normalizing = "mode", cat.encoding = "onehot"
      )
      
      sliced <- lapply(gan_res$imputation, function(df) df[val_idx, ])
      for (i in 1:m) fold_pieces[[i]][[k]] <- sliced[[i]]
    }
    
    coef_mat <- matrix(NA, nrow = m, ncol = length(true_coeffs))
    flat_indices <- unlist(all_val_indices)
    
    for (i in 1:m) {
      bound_df <- do.call(rbind, fold_pieces[[i]])
      reordered_df <- data 
      reordered_df[flat_indices, ] <- bound_df
      
      reordered_df <- match_types(reordered_df, data_ori)
      if (is_survey) {
        curr_design <- base_design
        curr_design$variables <- reordered_df
        fit <- if(is_cox) svycoxph(model_formula, design = curr_design) else svyglm(model_formula, design = curr_design, family = model_family)
      } else {
        fit <- if(is_cox) coxph(model_formula, data = reordered_df) else glm(model_formula, data = reordered_df, family = model_family)
      }
      coef_mat[i, ] <- coef(fit)
    }
    
    betameans <- colMeans(coef_mat, na.rm = TRUE)
    if (is_survey) {
      if (is_cox){
        fit <- svycoxph(model_formula, design = base_design, init = betameans, iter.max = 0)
      }else{
        fit <- svyglm(model_formula, design = base_design, family = model_family, start = betameans, control = list(maxit = 0))
      }
    } else {
      if (is_cox){
        fit <- coxph(model_formula, data = match_types(data, data_ori), init = betameans, iter.max = 0)
      }else{
        fit <- glm(model_formula, data = match_types(data, data_ori), family = model_family, start = betameans, control = list(maxit = 0))
      }
    }
    nll <- -as.numeric(logLik(fit))
    
    return (nll)
  }
  
  obj = ObjectiveRFun$new(
    fun = objective_fun,
    domain = search_space,
    codomain = ps(bias = p_dbl(tags = "minimize"))
  )
  
  opt_instance = OptimInstanceBatchSingleCrit$new(
    objective = obj,
    terminator = trm("evals", n_evals = n_evals)
  )
  
  learner_km = lrn("regr.km", covtype = "matern5_2", control = list(trace = FALSE))
  surrogate_object = SurrogateLearner$new(learner_km)
  
  optimizer = opt("mbo", 
                  loop_function = bayesopt_ego,
                  acq_function = acqf("ei"),
                  surrogate = surrogate_object 
  )
  
  optimizer$optimize(opt_instance)
  
  if (!is.null(log_path)) {
    log_data <- as.data.table(opt_instance$archive)
    write.csv(log_data, log_path, row.names = FALSE)
  }
  
  # ... (Optimization finished) ...
  
  best_params <- opt_instance$result_x_domain
  
  # 1. Expand Learning Rate Ratio
  best_params$lr_g <- best_params$lr_d * best_params$g_d_ratio
  best_params$g_d_ratio <- NULL
  
  # 2. Expand Common Weight Decay (Matched G & D)
  best_params$g_weight_decay <- best_params$weight_decay
  best_params$d_weight_decay <- best_params$weight_decay
  best_params$weight_decay <- NULL
  
  # 3. Expand Common Architecture to Vector Dimensions
  # The GAN expects 'g_dim' as a vector like c(256, 256, 256)
  best_params$g_dim <- rep(best_params$common_width, best_params$common_depth)
  best_params$d_dim <- rep(best_params$common_width, best_params$common_depth)
  
  # 4. Cleanup Proxy Parameters
  best_params$common_width <- NULL
  best_params$common_depth <- NULL
  
  # 6. Final Config Creation
  full_config <- utils::modifyList(cwgangp_default(), best_params)
  
  if (!is.null(best_config_path)) {
    saveRDS(full_config, best_config_path)
  }
  
  return(full_config)
}