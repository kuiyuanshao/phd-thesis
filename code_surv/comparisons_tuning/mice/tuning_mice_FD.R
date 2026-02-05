pacman::p_load(mlr3mbo, mlr3, bbotk, paradox, mlr3learners, Matrix, expm, dplyr, data.table, DiceKriging, mice, expm)
source("../../00_utils_functions.R") 

calc_frechet <- function(data_true, data_imp, time_var = NULL, event_var = NULL) {
  to_numeric_matrix_robust <- function(target_df, reference_df, t_col, e_col) {
    if (!is.null(t_col) && !is.null(e_col)) {
      eps <- min(target_df[[t_col]][target_df[[t_col]] > 0], na.rm=TRUE) / 2
      if(is.infinite(eps)) eps <- 1e-5
      target_df[[t_col]] <- log(as.numeric(target_df[[t_col]]) + eps)
    }
    
    mat_list <- list()
    for (col_name in colnames(reference_df)) {
      vec_ref <- reference_df[[col_name]]
      vec_target <- target_df[[col_name]]
      
      if (is.numeric(vec_ref) || is.integer(vec_ref)) {
        mat_list[[col_name]] <- as.matrix(vec_target)
      } else if (is.factor(vec_ref) || is.character(vec_ref)) {
        levels_ref <- levels(as.factor(vec_ref))
        if (length(levels_ref) < 2) next
        for (lvl in levels_ref[-1]) {
          dummy_name <- paste0(col_name, "_", lvl)
          mat_list[[dummy_name]] <- as.numeric(vec_target == lvl)
        }
      }
    }
    return(do.call(cbind, mat_list))
  }
  
  tryCatch({
    mat_true <- to_numeric_matrix_robust(data_true, data_true, time_var, event_var)
    mat_imp <- to_numeric_matrix_robust(data_imp, data_true, time_var, event_var)
    mu_ref <- colMeans(mat_true)
    sd_ref <- apply(mat_true, 2, sd)
    
    sd_ref[sd_ref < 1e-9] <- 1 
  
    mat_true_scaled <- scale(mat_true, center = mu_ref, scale = sd_ref)
    mat_imp_scaled  <- scale(mat_imp,  center = mu_ref, scale = sd_ref)
  
    mu_true_s <- colMeans(mat_true_scaled)
    mu_imp_s  <- colMeans(mat_imp_scaled)
    epsilon <- 1e-5
    p <- ncol(mat_true_scaled)
    
    sigma_true <- cov(mat_true_scaled) + diag(epsilon, p)
    sigma_imp  <- cov(mat_imp_scaled)  + diag(epsilon, p)
    diff_mu <- sum((mu_true_s - mu_imp_s)^2)
    product_mat <- sigma_true %*% sigma_imp
    
    eigen_vals <- eigen(product_mat, only.values = TRUE)$values
    
    sqrt_trace <- sum(Re(sqrt(as.complex(eigen_vals))))
    
    trace_term <- sum(diag(sigma_true)) + sum(diag(sigma_imp)) - 2 * sqrt_trace
    
    total_dist <- diff_mu + trace_term
  
    if(is.na(total_dist) || is.infinite(total_dist) || total_dist < 0) return(1e9)
    
    return(total_dist)
    
  }, error = function(e) {
    cat("\n[FD Error]:", e$message, "\n")
    return(1e9)
  })
}

tune_mice <- function(data, data_ori, data_info, search_space,
                      time_var = NULL, event_var = NULL, # New Arguments
                      best_config_path = "best_mice_config.rds",
                      log_path = "mice_tuning_log.csv",
                      n_evals = 20, m = 5, folds = 4) {
  
  set.seed(42)
  N <- nrow(data)
  shuffled_ids <- sample(1:N)
  fixed_fold_indices <- split(shuffled_ids, rep(1:folds, length.out = N))
  
  is_tte <- !is.null(time_var) && !is.null(event_var)
  mode_str <- if(is_tte) "Time-to-Event (Log-Time FD)" else "General Tabular (Standard FD)"
  cat(sprintf("[Setup] Tuning Mode: %s\n", mode_str))
  
  objective_fun = function(xs) {
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
    
    fold_pieces <- vector("list", m) 
    for(i in 1:m) fold_pieces[[i]] <- list()
    all_val_indices <- list()
    
    for (k in 1:folds) {
      val_idx <- fixed_fold_indices[[k]]
      all_val_indices[[k]] <- val_idx
      
      train_test_data <- data
      train_test_data[val_idx, data_info$phase2_vars] <- NA
      
      pred_matrix <- tryCatch({
        quickpred(
          train_test_data, 
          mincor = 0.15, 
          method = "pearson"
        )
      }, error = function(e) return(NULL))
      
      imputed_mids <- tryCatch({
        mice(
          data = train_test_data, 
          m = m, 
          method = "pmm", 
          maxcor = 1.0001, ls.meth = "ridge",
          predictorMatrix = pred_matrix, 
          ridge = xs$ridge, 
          maxit = 25,
          printFlag = FALSE
        )
      }, error = function(e) return(NULL))
      
      if(is.null(imputed_mids)) return(list(bias = 1e9))
      
      for (i in 1:m) {
        completed_data <- complete(imputed_mids, action = i)
        fold_pieces[[i]][[k]] <- completed_data[val_idx, ]
      }
    }
    
    flat_indices <- unlist(all_val_indices)
    fd_scores <- numeric(m)
    
    for (i in 1:m) {
      bound_df <- do.call(rbind, fold_pieces[[i]])
      reordered_df <- data 
      reordered_df[flat_indices, ] <- bound_df
      reordered_df <- reordered_df[order(flat_indices), ]
      reordered_df <- match_types(reordered_df, data)
      
      tryCatch({
        fd_scores[i] <- calc_frechet(data[, -which(names(data)%in%data_info$phase1_vars)], 
                                     reordered_df[, -which(names(data)%in%data_info$phase1_vars)], 
                                     time_var, event_var)
      }, error = function(e) {
        cat("Error in FD calc: ", e$message, "\n")
        fd_scores[i] <- NA
      })
    }
    
    mean_fd <- mean(fd_scores, na.rm = TRUE)
    if(is.na(mean_fd)) mean_fd <- 1e9
    
    cat(sprintf("   Result FD Score: %.4f\n", mean_fd))
    cat("----------------------------------------------------------------\n")
    
    return (list(bias = mean_fd))
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
  
  best_params <- opt_instance$result_x_domain
  if (!is.null(log_path)) write.csv(as.data.table(opt_instance$archive), log_path)
  if (!is.null(best_config_path)) saveRDS(best_params, best_config_path)
  
  return(best_params)
}

load(paste0("../../data/True/0001.RData"))

samp_srs <- read.csv(paste0("../../data/Sample/SRS/0001.csv"))
samp_srs <- match_types(samp_srs, data) 

samp_bal <- read.csv(paste0("../../data/Sample/Balance/0001.csv"))
samp_bal <- match_types(samp_bal, data) 

samp_ney <- read.csv(paste0("../../data/Sample/Neyman/0001.csv"))
samp_ney <- match_types(samp_ney, data)

samp_srs <- samp_srs[samp_srs$R == 1,]
samp_bal <- samp_bal[samp_bal$R == 1,]
samp_ney <- samp_ney[samp_ney$R == 1,]

samp_srs$H0_STAR <- nelsonaalen(samp_srs, T_I_STAR, EVENT_STAR)
samp_srs$H0_TRUE <- nelsonaalen(samp_srs, T_I, EVENT)
samp_srs$H0_STAR[is.na(samp_srs$H0_STAR)] <- 0
samp_srs$H0_TRUE[is.na(samp_srs$H0_TRUE)] <- 0

samp_bal$H0_STAR <- nelsonaalen(samp_bal, T_I_STAR, EVENT_STAR)
samp_bal$H0_TRUE <- nelsonaalen(samp_bal, T_I, EVENT)
samp_bal$H0_STAR[is.na(samp_bal$H0_STAR)] <- 0
samp_bal$H0_TRUE[is.na(samp_bal$H0_TRUE)] <- 0

samp_ney$H0_STAR <- nelsonaalen(samp_ney, T_I_STAR, EVENT_STAR)
samp_ney$H0_TRUE <- nelsonaalen(samp_ney, T_I, EVENT)
samp_ney$H0_STAR[is.na(samp_ney$H0_STAR)] <- 0
samp_ney$H0_TRUE[is.na(samp_ney$H0_TRUE)] <- 0

samp_srs <- samp_srs %>% 
  mutate(across(all_of(data_info_srs$cat_vars), as.factor, .names = "{.col}"),
         across(all_of(data_info_srs$num_vars), as.numeric, .names = "{.col}"))
samp_bal <- samp_bal %>% 
  mutate(across(all_of(data_info_balance$cat_vars), as.factor, .names = "{.col}"),
         across(all_of(data_info_balance$num_vars), as.numeric, .names = "{.col}"))
samp_ney <- samp_ney %>% 
  mutate(across(all_of(data_info_neyman$cat_vars), as.factor, .names = "{.col}"),
         across(all_of(data_info_neyman$num_vars), as.numeric, .names = "{.col}"))


search_space = ps(
  # mincor = p_dbl(lower = 0.1, upper = 0.6),
  ridge = p_dbl(lower = 1e-3, upper = 0.75)
)

# 1. Tuning for SRS (Time-to-Event data)
tune_srs <- tune_mice(
  data = samp_srs, 
  data_ori = data, 
  data_info = data_info_srs, 
  search_space = search_space,
  time_var = "T_I",   # <--- Explicit Time Variable
  event_var = "EVENT", # <--- Explicit Event Variable
  best_config_path = "best_mice_config_fd_srs.rds",
  log_path = "mice_tuning_log_fd_srs.csv",
  n_evals = 40, m = 3, folds = 4
)

tune_bal <- tune_mice(
  data = samp_bal, 
  data_ori = data, 
  data_info = data_info_balance, 
  search_space = search_space,
  time_var = "T_I",   # <--- Explicit Time Variable
  event_var = "EVENT", # <--- Explicit Event Variable
  best_config_path = "best_mice_config_fd_bal.rds",
  log_path = "mice_tuning_log_fd_bal.csv",
  n_evals = 40, m = 3, folds = 4
)

tune_neyman <- tune_mice(
  data = samp_ney, 
  data_ori = data, 
  data_info = data_info_neyman, 
  search_space = search_space,
  time_var = "T_I",   # <--- Explicit Time Variable
  event_var = "EVENT", # <--- Explicit Event Variable
  best_config_path = "best_mice_config_fd_ney.rds",
  log_path = "mice_tuning_log_fd_ney.csv",
  n_evals = 40, m = 3, folds = 4
)

