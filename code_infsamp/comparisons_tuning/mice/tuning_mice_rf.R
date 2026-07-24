pacman::p_load(mlr3mbo, mlr3, bbotk, paradox, mlr3learners, Matrix, expm, dplyr, data.table, DiceKriging, mice, ranger, fields)
source("../../00_utils_functions.R")
options(ranger.num.threads = 8)
source("../../../utils/loss_calculator.R")

tune_info <- list(num_vars = c("Z1", "Z2", "X", "Xt", "Y"),
                  cat_vars = c("Z3"), phase1_vars = c("Xt"), phase2_vars = c("X"))

tune_rf <- function(data, data_info, search_space, log_path = "rf_tuning_results.csv",
                    n_evals = 30, m = 5, folds = 3) {
  for (c in data_info$num_vars) if (c %in% colnames(data)) data[[c]] <- as.numeric(as.character(data[[c]]))
  set.seed(42); N <- nrow(data); shuffled <- sample(1:N)
  fixed_fold_indices <- split(shuffled, rep(1:folds, length.out = N))
  loss_calc <- create_loss_calculator(data_info)
  hist_acc <- replicate(folds, numeric(0), simplify = FALSE); hist_dist <- replicate(folds, numeric(0), simplify = FALSE)

  objective_fun <- function(xs) {
    fold_mse <- fold_ce <- fold_ed <- numeric(folds)
    for (k in 1:folds) {
      val_idx <- fixed_fold_indices[[k]]; tt <- data; tt[val_idx, data_info$phase2_vars] <- NA
      mids <- tryCatch(mice(tt, m = m, method = "rf", ntree = xs$ntree, mtry = xs$mtry,
                            min.node.size = xs$nodesize, sample.fraction = xs$sampsize_ratio,
                            num.threads = 8, maxit = 5, printFlag = FALSE,
                            remove.collinear = FALSE, remove.constant = TRUE), error = function(e) NULL)
      if (is.null(mids)) return(list(acc_loss = 1e9, dist_loss = 1e9))
      m_mse <- m_ce <- m_ed <- numeric(m); true_val <- data[val_idx, ]
      for (i in 1:m) { fake <- complete(mids, i)[val_idx, ]; L <- loss_calc(true_val, fake)
        m_mse[i] <- L$weighted_mse; m_ce[i] <- L$weighted_ce; m_ed[i] <- L$weighted_ed }
      fold_mse[k] <- mean(m_mse); fold_ce[k] <- mean(m_ce); fold_ed[k] <- mean(m_ed)
      cur_acc <- fold_mse[k] + fold_ce[k]; cur_dist <- fold_ed[k]
      # stop early if both losses are in the worst quarter so far
      if (k < folds) { hist_acc[[k]] <<- c(hist_acc[[k]], cur_acc); hist_dist[[k]] <<- c(hist_dist[[k]], cur_dist)
        if (length(hist_acc[[k]]) >= max(10, floor(n_evals * 0.25)))
          if (cur_acc > quantile(hist_acc[[k]], .75, na.rm = TRUE) && cur_dist > quantile(hist_dist[[k]], .75, na.rm = TRUE))
            return(list(acc_loss = 1e9, dist_loss = 1e9)) }
    }
    acc <- mean(fold_mse) + mean(fold_ce); if (is.na(acc)) acc <- 1e9
    ed <- mean(fold_ed); if (is.na(ed)) ed <- 1e9; list(acc_loss = acc, dist_loss = ed)
  }
  obj <- ObjectiveRFun$new(fun = objective_fun, domain = search_space,
                           codomain = ps(acc_loss = p_dbl(tags = "minimize"), dist_loss = p_dbl(tags = "minimize")))
  inst <- OptimInstanceBatchMultiCrit$new(objective = obj, terminator = trm("evals", n_evals = n_evals))
  surr <- SurrogateLearner$new(lrn("regr.km", covtype = "matern5_2", control = list(trace = FALSE)))
  opt("mbo", loop_function = bayesopt_parego, surrogate = surr, acq_function = acqf("ei"))$optimize(inst)
  if (!is.null(log_path)) write.csv(as.data.table(inst$archive), log_path)
  inst$result_x_domain
}

load("../../data/True/0001.RData")
samp <- read.csv("../../data/Sample/InfSamp/0001.csv"); samp <- match_types(samp, as.data.frame(data))
samp <- samp[samp$R == 1, ]
samp <- samp %>% mutate(across(all_of(tune_info$cat_vars), as.factor),
                        across(all_of(tune_info$num_vars), ~ as.numeric(as.character(.))))

search_space <- ps(ntree = p_int(10, 200), mtry = p_int(round(ncol(samp) / 3), ncol(samp)),
                   nodesize = p_int(1, 20), sampsize_ratio = p_dbl(0.2, 0.9))
tm <- system.time({ tune_rf(samp, tune_info, search_space, "rf_tuning_results.csv", n_evals = 30, m = 5, folds = 3) })
save(tm, file = "rf_tuning_time.RData")

res <- read.csv("rf_tuning_results.csv")
res <- res %>% rowwise() %>%
  mutate(is_pareto = !any(res$acc_loss <= acc_loss & res$dist_loss <= dist_loss &
           (res$acc_loss < acc_loss | res$dist_loss < dist_loss))) %>% ungroup()
vals <- apply(filter(res, is_pareto)[, search_space$ids()], 2, median)
params <- list(); for (i in seq_along(vals)) params[[names(vals)[i]]] <-
  if (search_space$storage_type[i] == "integer") round(vals[i]) else round(vals[i], 4)
dir.create("../../data/Config", recursive = TRUE, showWarnings = FALSE)
saveRDS(params, file = "../../data/Config/best_config_rf.rds")
print(params)
