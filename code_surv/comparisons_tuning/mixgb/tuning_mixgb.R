pacman::p_load(mlr3mbo, mlr3, bbotk, paradox, mlr3learners, survival, survey, torch, dplyr, data.table, DiceKriging, mixgb)
source("../../00_utils_functions.R")

tune_mixgb <- function(data, data_ori, data_info, target_model, search_space,
                       best_config_path = "best_mixgb_config.rds",
                       log_path = "mixgb_tuning_log.csv",
                       n_evals = 20, m = 5, folds = 4) {
  set.seed(42)
  N <- nrow(data)
  shuffled_ids <- sample(1:N)
  fixed_fold_indices <- split(shuffled_ids, rep(1:folds, length.out = N))
  
  true_coeffs <- coef(target_model)
  model_formula <- formula(target_model)
  
  is_survey <- inherits(target_model, c("svycoxph", "svyglm"))
  is_cox    <- inherits(target_model, c("svycoxph", "coxph"))
  
  base_design <- if(is_survey) {
    if(inherits(target_model, "svycoxph")) mod_bal$survey.design else target_model$survey.design
  } else NULL
  
  model_family <- if(inherits(target_model, c("glm", "svyglm"))) family(target_model) else NULL
  
  objective_fun = function(xs) {
    cat("\n[Opt] Testing Mixgb Config:", paste(names(xs), xs, sep = "=", collapse = ", "), "\n")
    mixgb_direct_names <- c("nrounds", "pmm.k", "initial.approx", "pmm.type")
    
    mixgb_args <- xs[names(xs) %in% mixgb_direct_names]
    xgb_params <- xs[!(names(xs) %in% mixgb_direct_names)]
    
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
    
    fold_pieces <- vector("list", m) 
    for(i in 1:m) fold_pieces[[i]] <- list()
    all_val_indices <- list()
    
    for (k in 1:folds) {
      val_idx <- fixed_fold_indices[[k]]
      all_val_indices[[k]] <- val_idx
      
      train_test_data <- data
      train_test_data[val_idx, data_info$phase2_vars] <- NA
      imputed_list <- do.call(mixgb, c(
        list(
          data = train_test_data, 
          m = m,
          xgb.params = xgb_params,
          verbose = FALSE
        ),
        mixgb_args
      ))
      
      for (i in 1:m) {
        fold_pieces[[i]][[k]] <- imputed_list[[i]][val_idx, ]
      }
    }
    
    coef_mat <- matrix(NA, nrow = m, ncol = length(true_coeffs))
    flat_indices <- unlist(all_val_indices)
    
    for (i in 1:m) {
      bound_df <- do.call(rbind, fold_pieces[[i]])
      reordered_df <- data 
      reordered_df[flat_indices, ] <- bound_df
      reordered_df <- reordered_df[order(flat_indices), ]
      reordered_df <- match_types(reordered_df, data_ori)
      if (is_survey) {
        curr_design <- base_design
        curr_design$variables <- reordered_df
        class(curr_design) <- class(base_design)
        fit <- if(is_cox) svycoxph(model_formula, design = curr_design) else svyglm(model_formula, design = curr_design, family = model_family)
      } else {
        fit <- if(is_cox) coxph(model_formula, data = reordered_df) else glm(model_formula, data = reordered_df, family = model_family)
      }
      coef_mat[i, ] <- coef(fit)
    }
    
    diff_mat <- sweep(coef_mat, 2, true_coeffs, "-")
    avg_beta <- mean(abs(true_coeffs))
    # Formula: |diff| / (|truth| + avg_beta)
    rel_err_mat <- sweep(abs(diff_mat), 2, abs(true_coeffs) + avg_beta, "/")
    return(list(bias = mean(rel_err_mat, na.rm = TRUE)))
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
  
  # Export results
  best_params <- opt_instance$result_x_domain
  if (!is.null(log_path)) write.csv(as.data.table(opt_instance$archive), log_path)
  if (!is.null(best_config_path)) saveRDS(best_params, best_config_path)
  
  return(best_params)
}


source("../../00_utils_functions.R")
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
  gamma             = p_dbl(lower = 0, upper = 1),
  min_child_weight  = p_int(lower = 1, upper = 7),
  eta               = p_dbl(lower = 0.01, upper = 0.2),
  subsample         = p_dbl(lower = 0.5, upper = 0.99),
  max_depth         = p_int(lower = 3, upper = 7),
  colsample_bytree  = p_dbl(lower = 0.5, upper = 1)
)

tune_srs <- tune_mixgb(samp_srs, data, data_info_srs, mod_srs, search_space,
                     best_config_path = "best_mixgb_config_srs.rds",
                     log_path = "mixgb_tuning_log_srs.csv",
                     n_evals = 60, m = 3, folds = 4)
tune_bal <- tune_mixgb(samp_bal, data, data_info_balance, mod_bal, search_space,
                     best_config_path = "best_mixgb_config_bal.rds",
                     log_path = "mixgb_tuning_log_bal.csv",
                     n_evals = 60, m = 3, folds = 4)
tune_ney <- tune_mixgb(samp_ney, data, data_info_neyman, mod_ney, search_space,
                     best_config_path = "best_mixgb_config_ney.rds",
                     log_path = "mixgb_tuning_log_ney.csv",
                     n_evals = 60, m = 3, folds = 4)


# X_srs <- samp_srs %>%
#   select(-data_info_srs$phase2_vars) %>%
#   subset(R == 1) %>%
#   select(-c("R", "W"))
# X_balance <- samp_balance %>%
#   select(-data_info_balance$phase2_vars) %>%
#   subset(R == 1) %>%
#   select(-c("R"))
# X_neyman <- samp_neyman %>%
#   select(-data_info_neyman$phase2_vars) %>%
#   subset(R == 1) %>%
#   select(-c("R"))
#
# Outcomes_srs <- samp_srs %>%
#   subset(R == 1) %>%
#   select(data_info_srs$phase2_vars) %>%
#   as.list()
# Outcomes_balance <- samp_balance %>%
#   subset(R == 1) %>%
#   select(data_info_balance$phase2_vars) %>%
#   as.list()
# Outcomes_neyman <- samp_neyman %>%
#   subset(R == 1) %>%
#   select(data_info_neyman$phase2_vars) %>%
#   as.list()
#
# types.srs <- ifelse(data_info_srs$phase2_vars %in% data_info_srs$num_vars, "reg:squarederror", "multi:softmax")
# types.srs[unlist(lapply(Outcomes_srs, FUN = function(i) length(unique(i)))) == 2] <- "binary:logistic"
# eval_metric.srs <- ifelse(data_info_srs$phase2_vars %in% data_info_srs$num_vars, "rmse", "mlogloss")
# eval_metric.srs[unlist(lapply(Outcomes_srs, FUN = function(i) length(unique(i)))) == 2] <- "logloss"
#
# types.balance <- ifelse(data_info_balance$phase2_vars %in% data_info_balance$num_vars, "reg:squarederror", "multi:softmax")
# types.balance[unlist(lapply(Outcomes_balance, FUN = function(i) length(unique(i)))) == 2] <- "binary:logistic"
# eval_metric.balance <- ifelse(data_info_balance$phase2_vars %in% data_info_balance$num_vars, "rmse", "mlogloss")
# eval_metric.balance[unlist(lapply(Outcomes_balance, FUN = function(i) length(unique(i)))) == 2] <- "logloss"
#
# types.neyman <- ifelse(data_info_neyman$phase2_vars %in% data_info_neyman$num_vars, "reg:squarederror", "multi:softmax")
# types.neyman[unlist(lapply(Outcomes_neyman, FUN = function(i) length(unique(i)))) == 2] <- "binary:logistic"
# eval_metric.neyman <- ifelse(data_info_neyman$phase2_vars %in% data_info_neyman$num_vars, "rmse", "mlogloss")
# eval_metric.neyman[unlist(lapply(Outcomes_neyman, FUN = function(i) length(unique(i)))) == 2] <- "logloss"
#
# grid <- tidyr::expand_grid(max_depth = 1:5,
#                            eta = seq(0.05, 1, by = 0.05),
#                            colsample_bytree = seq(0.1, 1, by = 0.1),
#                            subsample = seq(0.6, 0.9, by = 0.1))
#
# tuning <- function(X, outcomes, types, eval_metric, grid, nrounds = 500, k = 5, seed = 1){
#   set.seed(seed)
#   res_rows <- list()
#   models_per_setting <- list()
#
#   metric_col <- function(eval_metric) {
#     paste0("test_", eval_metric, "_mean")
#   }
#   X <- model.matrix(~ . - 1, data = X)
#   for (i in seq_len(nrow(grid))) {
#     cat("Current: ", i)
#     md <- grid$max_depth[i]
#     lr <- grid$eta[i]
#     cs <- grid$colsample_bytree[i]
#     ss <- grid$subsample[i]
#     per_outcome_scores <- numeric(length(outcomes))
#
#     for (j in seq_along(outcomes)) {
#       yj <- outcomes[[j]]
#       obj <- types[[j]]
#       metric <- eval_metric[[j]]
#       dtrain <- xgb.DMatrix(data = X, label = as.numeric(yj) - 1)
#
#       if (obj == "multi:softmax"){
#         num_class <- length(unique(yj))
#         params <- list(max_depth = md, eta = lr, objective = obj,
#                        subsample = ss, colsample_bytree = cs,
#                        num_class = num_class)
#       }else{
#         params <- list(max_depth = md, eta = lr, objective = obj,
#                        subsample = ss, colsample_bytree = cs)
#       }
#
#       cv <- xgb.cv(params = params, data = dtrain, nrounds = nrounds,
#                    nfold = k, metrics = metric, showsd = TRUE, verbose = 0)
#
#       mcol <- metric_col(metric)
#
#       metric_val <- tail(cv$evaluation_log[[mcol]], 1)
#       per_outcome_scores[j] <- metric_val
#     }
#
#     agg_loss <- sum(per_outcome_scores)
#
#     res_rows[[i]] <- data.frame(
#       max_depth = md,
#       eta = lr,
#       colsample_bytree = cs,
#       subsample = ss,
#       agg_loss = agg_loss,
#       t(per_outcome_scores)
#     )
#   }
#
#   res <- do.call(rbind, res_rows)
#   best_idx <- which.min(res$agg_loss)
#   best <- res[best_idx, c("max_depth", "eta", "colsample_bytree", "subsample", "agg_loss")]
#
#
#   list(best_params = best,
#        cv_table = res[order(res$agg_loss), ])
# }
#
# srs_tune <- tuning(X_srs, Outcomes_srs, types.srs, eval_metric.srs, grid, nrounds = 100, k = 5, seed = 1)
# save(srs_tune, file = "../../data/Params/mixgb/mixgb_srsParams.RData")
# balance_tune <- tuning(X_balance, Outcomes_balance, types.balance, eval_metric.balance, grid, nrounds = 100, k = 5, seed = 1)
# save(balance_tune, file = "../../data/Params/mixgb/mixgb_balanceParams.RData")
# neyman_tune <- tuning(X_neyman, Outcomes_neyman, types.neyman, eval_metric.neyman, grid, nrounds = 100, k = 5, seed = 1)
# save(neyman_tune, file = "../../data/Params/mixgb/mixgb_neymanParams.RData")
#
# ### With large amount of covariates, we use a less value of column subsample
# ### So we choose the parameter combinations with lower colsubsample rate while maintaining good total loss.
# set.seed(100)
# srs_chose <- as.list(srs_tune$cv_table["864", 1:4])
# balance_chose <- as.list(balance_tune$cv_table["867", 1:4])
# neyman_chose <- as.list(balance_tune$cv_table["868", 1:4])
