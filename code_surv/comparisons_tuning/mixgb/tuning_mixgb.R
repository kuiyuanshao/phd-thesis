lapply(c("xgboost", "dplyr"), require, character.only = T)
source("../../00_utils_functions.R")
samp_srs <- read.csv("../../data/Sample/SRS/0001.csv")
samp_balance <- read.csv("../../data/Sample/Balance/0001.csv")
samp_neyman <- read.csv("../../data/Sample/Neyman/0001.csv")

samp_srs <- match_types(samp_srs, data) %>% 
  mutate(across(all_of(data_info_srs$cat_vars), as.factor, .names = "{.col}"),
         across(all_of(data_info_srs$num_vars), as.numeric, .names = "{.col}"))
samp_balance <- match_types(samp_balance, data) %>% 
  mutate(across(all_of(data_info_balance$cat_vars), as.factor, .names = "{.col}"),
         across(all_of(data_info_balance$num_vars), as.numeric, .names = "{.col}"))
samp_neyman <- match_types(samp_neyman, data) %>% 
  mutate(across(all_of(data_info_neyman$cat_vars), as.factor, .names = "{.col}"),
         across(all_of(data_info_neyman$num_vars), as.numeric, .names = "{.col}"))

X_srs <- samp_srs %>%
  select(-data_info_srs$phase2_vars) %>% 
  subset(R == 1) %>%
  select(-c("R", "W"))
X_balance <- samp_balance %>%
  select(-data_info_balance$phase2_vars) %>% 
  subset(R == 1) %>%
  select(-c("R"))
X_neyman <- samp_neyman %>%
  select(-data_info_neyman$phase2_vars) %>% 
  subset(R == 1) %>%
  select(-c("R"))

Outcomes_srs <- samp_srs %>% 
  subset(R == 1) %>%
  select(data_info_srs$phase2_vars) %>%
  as.list()
Outcomes_balance <- samp_balance %>% 
  subset(R == 1) %>%
  select(data_info_balance$phase2_vars) %>%
  as.list()
Outcomes_neyman <- samp_neyman %>% 
  subset(R == 1) %>%
  select(data_info_neyman$phase2_vars) %>%
  as.list()

types.srs <- ifelse(data_info_srs$phase2_vars %in% data_info_srs$num_vars, "reg:squarederror", "multi:softmax")
types.srs[unlist(lapply(Outcomes_srs, FUN = function(i) length(unique(i)))) == 2] <- "binary:logistic"
eval_metric.srs <- ifelse(data_info_srs$phase2_vars %in% data_info_srs$num_vars, "rmse", "mlogloss")
eval_metric.srs[unlist(lapply(Outcomes_srs, FUN = function(i) length(unique(i)))) == 2] <- "logloss"

types.balance <- ifelse(data_info_balance$phase2_vars %in% data_info_balance$num_vars, "reg:squarederror", "multi:softmax")
types.balance[unlist(lapply(Outcomes_balance, FUN = function(i) length(unique(i)))) == 2] <- "binary:logistic"
eval_metric.balance <- ifelse(data_info_balance$phase2_vars %in% data_info_balance$num_vars, "rmse", "mlogloss")
eval_metric.balance[unlist(lapply(Outcomes_balance, FUN = function(i) length(unique(i)))) == 2] <- "logloss"

types.neyman <- ifelse(data_info_neyman$phase2_vars %in% data_info_neyman$num_vars, "reg:squarederror", "multi:softmax")
types.neyman[unlist(lapply(Outcomes_neyman, FUN = function(i) length(unique(i)))) == 2] <- "binary:logistic"
eval_metric.neyman <- ifelse(data_info_neyman$phase2_vars %in% data_info_neyman$num_vars, "rmse", "mlogloss")
eval_metric.neyman[unlist(lapply(Outcomes_neyman, FUN = function(i) length(unique(i)))) == 2] <- "logloss"

grid <- tidyr::expand_grid(max_depth = 1:5,
                           eta = seq(0.05, 1, by = 0.05),
                           colsample_bytree = seq(0.1, 1, by = 0.1),
                           subsample = seq(0.6, 0.9, by = 0.1))

tuning <- function(X, outcomes, types, eval_metric, grid, nrounds = 500, k = 5, seed = 1){
  set.seed(seed)
  res_rows <- list()
  models_per_setting <- list()
  
  metric_col <- function(eval_metric) {
    paste0("test_", eval_metric, "_mean")
  }
  X <- model.matrix(~ . - 1, data = X)
  for (i in seq_len(nrow(grid))) {
    cat("Current: ", i)
    md <- grid$max_depth[i]
    lr <- grid$eta[i]
    cs <- grid$colsample_bytree[i]
    ss <- grid$subsample[i]
    per_outcome_scores <- numeric(length(outcomes))
    
    for (j in seq_along(outcomes)) {
      yj <- outcomes[[j]]
      obj <- types[[j]]
      metric <- eval_metric[[j]]
      dtrain <- xgb.DMatrix(data = X, label = as.numeric(yj) - 1)
      
      if (obj == "multi:softmax"){
        num_class <- length(unique(yj))
        params <- list(max_depth = md, eta = lr, objective = obj,
                       subsample = ss, colsample_bytree = cs, 
                       num_class = num_class)
      }else{
        params <- list(max_depth = md, eta = lr, objective = obj,
                       subsample = ss, colsample_bytree = cs)
      }
      
      cv <- xgb.cv(params = params, data = dtrain, nrounds = nrounds, 
                   nfold = k, metrics = metric, showsd = TRUE, verbose = 0)
      
      mcol <- metric_col(metric)
      
      metric_val <- tail(cv$evaluation_log[[mcol]], 1)
      per_outcome_scores[j] <- metric_val
    }
    
    agg_loss <- sum(per_outcome_scores)
    
    res_rows[[i]] <- data.frame(
      max_depth = md,
      eta = lr,
      colsample_bytree = cs,
      subsample = ss,
      agg_loss = agg_loss,
      t(per_outcome_scores)
    )
  }
  
  res <- do.call(rbind, res_rows)
  best_idx <- which.min(res$agg_loss)
  best <- res[best_idx, c("max_depth", "eta", "colsample_bytree", "subsample", "agg_loss")]
  

  list(best_params = best,
       cv_table = res[order(res$agg_loss), ])
}

srs_tune <- tuning(X_srs, Outcomes_srs, types.srs, eval_metric.srs, grid, nrounds = 100, k = 5, seed = 1)
save(srs_tune, file = "../../data/Params/mixgb/mixgb_srsParams.RData")
balance_tune <- tuning(X_balance, Outcomes_balance, types.balance, eval_metric.balance, grid, nrounds = 100, k = 5, seed = 1)
save(balance_tune, file = "../../data/Params/mixgb/mixgb_balanceParams.RData")
neyman_tune <- tuning(X_neyman, Outcomes_neyman, types.neyman, eval_metric.neyman, grid, nrounds = 100, k = 5, seed = 1)
save(neyman_tune, file = "../../data/Params/mixgb/mixgb_neymanParams.RData")

### With large amount of covariates, we use a less value of column subsample
### So we choose the parameter combinations with lower colsubsample rate while maintaining good total loss.
set.seed(100)
srs_chose <- as.list(srs_tune$cv_table["864", 1:4])
balance_chose <- as.list(balance_tune$cv_table["867", 1:4])
neyman_chose <- as.list(balance_tune$cv_table["868", 1:4])
