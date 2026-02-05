pacman::p_load(mlr3mbo, mlr3, bbotk, paradox, mlr3learners, survival, survey, torch, dplyr, data.table, DiceKriging, mice)
source("../../00_utils_functions.R") 

tune_mice <- function(data, data_ori, data_info, target_model, search_space,
                      best_config_path = "best_mice_config.rds",
                      log_path = "mice_tuning_log.csv",
                      n_evals = 20, m = 5, folds = 4, weights) {
  
  set.seed(42)
  N <- nrow(data)
  shuffled_ids <- sample(1:N)
  fixed_fold_indices <- split(shuffled_ids, rep(1:folds, length.out = N))
  
  # 1. Extract Ground Truth Metadata
  true_coeffs <- coef(target_model)
  true_names  <- names(true_coeffs) # Crucial for alignment
  model_formula <- formula(target_model)
  
  # Identify Model Type
  is_survey <- inherits(target_model, c("svycoxph", "svyglm"))
  is_cox    <- inherits(target_model, c("svycoxph", "coxph"))
  
  # Extract Design/Family safely
  base_design <- if(is_survey) target_model$survey.design else NULL
  model_family <- if(inherits(target_model, c("glm", "svyglm"))) family(target_model) else NULL
  
  # ----------------------------------------------------------------------------
  # Optimization Loop
  # ----------------------------------------------------------------------------
  objective_fun = function(xs) {
    
    # Logging
    cat("\n----------------------------------------------------------------\n")
    cat("[Opt] Testing Config:\n")
    for (nm in names(xs)) cat(sprintf("   %s: %s\n", nm, as.character(xs[[nm]])))
    
    fold_pieces <- vector("list", m) 
    for(i in 1:m) fold_pieces[[i]] <- list()
    all_val_indices <- list()
    
    # --- CV Loop ---
    for (k in 1:folds) {
      val_idx <- fixed_fold_indices[[k]]
      all_val_indices[[k]] <- val_idx
      
      train_test_data <- data
      # Masking Step
      train_test_data[val_idx, data_info$phase2_vars] <- NA
      
      # 1. Robust Quickpred
      pred_matrix <- tryCatch({
        mice::quickpred(train_test_data, mincor = 0.15, method = "pearson")
      }, error = function(e) NULL)
      
      if (is.null(pred_matrix)) return(list(bias = 1e9)) # Penalty for failure
      
      # 2. Robust MICE
      imputed_mids <- tryCatch({
        mice::mice(
          data = train_test_data, 
          m = m, 
          method = "pmm", 
          maxcor = 1.0001, ls.meth = "ridge",
          predictorMatrix = pred_matrix, 
          ridge = xs$ridge, 
          maxit = 25,
          printFlag = FALSE
        )
      }, error = function(e) NULL)
      
      if(is.null(imputed_mids)) return(list(bias = 1e9)) 
      
      for (i in 1:m) {
        completed_data <- complete(imputed_mids, action = i)
        fold_pieces[[i]][[k]] <- completed_data[val_idx, ]
      }
    }
    
    # --- Model Refitting & Coefficient Extraction ---
    flat_indices <- unlist(all_val_indices)
    coef_mat <- matrix(NA, nrow = m, ncol = length(true_coeffs))
    
    for (i in 1:m) {
      bound_df <- do.call(rbind, fold_pieces[[i]])
      reordered_df <- data 
      reordered_df[flat_indices, ] <- bound_df
      reordered_df <- reordered_df[order(flat_indices), ]
      reordered_df <- match_types(reordered_df, data_ori)
      
      # 3. Robust Model Fitting
      # We wrap the ENTIRE fitting process in tryCatch
      fit <- tryCatch({
        if (is_survey) {
          curr_design <- base_design
          curr_design$variables <- reordered_df # Inject imputed data into design
          
          if(is_cox) {
            svycoxph(model_formula, design = curr_design)
          } else {
            svyglm(model_formula, design = curr_design, family = model_family)
          }
        } else {
          if(is_cox) {
            coxph(model_formula, data = reordered_df)
          } else {
            glm(model_formula, data = reordered_df, family = model_family)
          }
        }
      }, error = function(e) return(NULL), warning = function(w) return(NULL)) 
      # Note: We treat warnings (like non-convergence) as failures to be safe
      
      if(is.null(fit)) {
        cat("   [!] Model Fit Failed (Singularity/Non-convergence)\n")
        return(list(bias = 1e9)) # Immediate exit with penalty
      }
      
      # 4. Safe Coefficient Alignment
      # Extract coefficients and force alignment with True Names
      fit_coefs <- coef(fit)
      aligned_coefs <- fit_coefs[true_names]
      
      # Check for NA (Singularity drops variables)
      if (any(is.na(aligned_coefs))) {
        cat("   [!] Singularity detected (dropped variables)\n")
        return(list(bias = 1e9))
      }
      
      coef_mat[i, ] <- aligned_coefs
    }
    
    # --- Bias Calculation ---
    diff_mat <- sweep(coef_mat, 2, true_coeffs, "-")
    avg_beta <- mean(abs(true_coeffs))
    
    # Relative Error Formula
    rel_err_mat <- sweep(abs(diff_mat), 2, abs(true_coeffs) + avg_beta, "/")
    weighted_err <- sweep(rel_err_mat, 2, weights, "*")
    
    final_bias <- mean(weighted_err[1], na.rm = TRUE)
    
    cat(sprintf("   Result Bias: %.4f\n", final_bias))
    cat("----------------------------------------------------------------\n")
    
    return (list(bias = final_bias))
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

samp_bal$H0_STAR <- nelsonaalen(samp_bal, T_I_STAR, EVENT_STAR)
samp_bal$H0_TRUE <- nelsonaalen(samp_bal, T_I, EVENT)

samp_ney$H0_STAR <- nelsonaalen(samp_ney, T_I_STAR, EVENT_STAR)
samp_ney$H0_TRUE <- nelsonaalen(samp_ney, T_I, EVENT)

mod_srs <- coxph(Surv(T_I, EVENT) ~
                   I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
                   I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
                   rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                   SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE,
                 data = samp_srs)

bal_design <- svydesign(ids = ~1, strata = ~STRATA, weights = ~W, 
                        data = samp_bal)
mod_bal <- svycoxph(Surv(T_I, EVENT) ~
                      I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
                      I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
                      rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                      SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE,
                    bal_design)

ney_design <- svydesign(ids = ~1, strata = ~STRATA, weights = ~W, 
                        data = samp_bal)
mod_ney <- svycoxph(Surv(T_I, EVENT) ~
                      I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
                      I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
                      rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                      SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE,
                    ney_design)

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
### PUC must be not tuned, we are having constant missing rates.

tune_srs <- tune_mice(samp_srs, data, data_info_srs, mod_srs, search_space,
                       best_config_path = "best_mice_config_srs.rds",
                       log_path = "mice_tuning_log_srs.csv",
                       n_evals = 40, m = 3, folds = 4, 
                      weights = c(100, rep(1, 7), rep(2, 4), 1, rep(3, 3)))
tune_bal <- tune_mice(samp_bal, data, data_info_balance, mod_bal, search_space,
                       best_config_path = "best_mice_config_bal.rds",
                       log_path = "mice_tuning_log_bal.csv",
                       n_evals = 40, m = 3, folds = 4, 
                      weights = c(100, rep(1, 7), rep(2, 4), 1, rep(3, 3)))
tune_ney <- tune_mice(samp_ney, data, data_info_neyman, mod_ney, search_space,
                       best_config_path = "best_mice_config_ney.rds",
                       log_path = "mice_tuning_log_ney.csv",
                       n_evals = 40, m = 3, folds = 4, 
                      weights = c(100, rep(1, 7), rep(2, 4), 1, rep(3, 3)))