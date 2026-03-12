pacman::p_load(mlr3mbo, mlr3, bbotk, paradox, mlr3learners, Matrix, expm, dplyr, data.table, DiceKriging, mice, fields)
source("../../00_utils_functions.R") 

create_loss_calculator <- function(data_info) {
  target_num <- intersect(data_info$num_vars, data_info$phase2_vars)
  target_cat <- intersect(data_info$cat_vars, data_info$phase2_vars)

  calc <- function(true_data, fake_data) {
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

tune_rf <- function(data, data_info, search_space,
                      log_path = "rf_tuning_log_OE.csv",
                      n_evals = 20, m = 5, folds = 4) {

  num_vars <- data_info$num_vars
  for (c in num_vars) {
    if (c %in% colnames(data)) {
      parsed_col <- suppressWarnings(as.numeric(as.character(data[[c]])))
      if (any(is.na(parsed_col) & !is.na(data[[c]]))) {
        data[[c]] <- as.numeric(as.factor(data[[c]]))
      } else {
        data[[c]] <- parsed_col
      }
    }
  }

  set.seed(42)
  N <- nrow(data)
  shuffled_ids <- sample(1:N)
  fixed_fold_indices <- split(shuffled_ids, rep(1:folds, length.out = N))

  loss_calc <- create_loss_calculator(data_info)

  fold_histories_acc <- list()
  fold_histories_dist <- list()
  for(i in 1:folds) {
    fold_histories_acc[[i]] <- numeric(0)
    fold_histories_dist[[i]] <- numeric(0)
  }

  objective_fun = function(xs) {
    cat("\n----------------------------------------------------------------\n")
    cat("[Opt] Testing Config:\n")
    for (nm in names(xs)) {
      cat(sprintf("   %s: %s\n", nm, as.character(xs[[nm]])))
    }

    fold_mse <- numeric(folds)
    fold_ce <- numeric(folds)
    fold_ed <- numeric(folds)

    for (k in 1:folds) {
      val_idx <- fixed_fold_indices[[k]]
      train_test_data <- data
      train_test_data[val_idx, data_info$phase2_vars] <- NA

      current_sampsize <- ceiling(xs$sampsize_ratio * nrow(train_test_data))

      imputed_mids <- tryCatch({
        mice(
          data = train_test_data,
          m = m,
          method = "rf",
          ntree = xs$ntree,
          mtry = xs$mtry,
          nodesize = xs$nodesize,
          sampsize = current_sampsize,
          remove.collinear = FALSE,
          maxit = 10,
          printFlag = FALSE
        )
      }, error = function(e) return(NULL))
      
      if(is.null(imputed_mids)) return(list(acc_loss = 1e9, dist_loss = 1e9))

      m_mse <- numeric(m)
      m_ce <- numeric(m)
      m_ed <- numeric(m)
      true_val_data <- data[val_idx, ]

      for (i in 1:m) {
        completed_data <- complete(imputed_mids, action = i)
        fake_val_data <- completed_data[val_idx, ]

        loss_results <- loss_calc(true_val_data, fake_val_data)

        m_mse[i]   <- loss_results$weighted_mse
        m_ce[i]    <- loss_results$weighted_ce
        m_ed[i]    <- loss_results$weighted_ed
      }

      current_fold_acc <- mean(m_mse, na.rm=TRUE) + mean(m_ce, na.rm=TRUE)
      current_fold_dist <- mean(m_ed, na.rm=TRUE)

      fold_mse[k] <- mean(m_mse, na.rm=TRUE)
      fold_ce[k] <- mean(m_ce, na.rm=TRUE)
      fold_ed[k] <- current_fold_dist

      if (k < folds) {
        fold_histories_acc[[k]] <<- c(fold_histories_acc[[k]], current_fold_acc)
        fold_histories_dist[[k]] <<- c(fold_histories_dist[[k]], current_fold_dist)

        if (length(fold_histories_acc[[k]]) >= max(10, floor(n_evals * 0.25))) {
          threshold_acc <- quantile(fold_histories_acc[[k]], 0.75, na.rm=TRUE)
          threshold_dist <- quantile(fold_histories_dist[[k]], 0.75, na.rm=TRUE)

          if (current_fold_acc > threshold_acc && current_fold_dist > threshold_dist) {
            cat(sprintf("Trial pruned at fold %d. Both Acc (%.4f > %.4f) AND Dist (%.4f > %.4f) are in the worst 25 percent.\n",
                        k, current_fold_acc, threshold_acc, current_fold_dist, threshold_dist))
            return(list(acc_loss = 1e9, dist_loss = 1e9))
          }
        }
      }
    }

    avg_mse  <- mean(fold_mse, na.rm=TRUE)
    avg_ce   <- mean(fold_ce, na.rm=TRUE)
    avg_ed   <- mean(fold_ed, na.rm=TRUE)
    avg_acc  <- avg_mse + avg_ce

    if(is.na(avg_acc)) avg_acc <- 1e9
    if(is.na(avg_ed)) avg_ed <- 1e9

    cat(sprintf("   Trial Summary -> Loss Acc: %.4f (MSE: %.4f, CE: %.4f), Loss Dist (ED): %.4f\n",
                avg_acc, avg_mse, avg_ce, avg_ed))
    cat("----------------------------------------------------------------\n")

    return (list(acc_loss = avg_acc, dist_loss = avg_ed))
  }

  obj = ObjectiveRFun$new(
    fun = objective_fun,
    domain = search_space,
    codomain = ps(
      acc_loss = p_dbl(tags = "minimize"),
      dist_loss = p_dbl(tags = "minimize")
    )
  )

  opt_instance = OptimInstanceBatchMultiCrit$new(
    objective = obj,
    terminator = trm("evals", n_evals = n_evals)
  )

  learner_km = lrn("regr.km", covtype = "matern5_2", control = list(trace = FALSE))
  surrogate_object = SurrogateLearner$new(learner_km)

  optimizer = opt("mbo",
                  loop_function = bayesopt_parego,
                  surrogate = surrogate_object,
                  acq_function = acqf("ei")
  )

  optimizer$optimize(opt_instance)

  best_params <- opt_instance$result_x_domain
  if (!is.null(log_path)) write.csv(as.data.table(opt_instance$archive), log_path)

  return (best_params)
}

data <- read.csv("../../data/data.csv")
samp_srs_full <- read.csv(paste0("../../data/Sample/SRS/0001.csv"))
samp_srs <- match_types(samp_srs_full, data)
samp_srs <- samp_srs[samp_srs$R == 1,]

safe_numeric <- function(x) {
  num_x <- suppressWarnings(as.numeric(as.character(x)))
  if (any(is.na(num_x) & !is.na(x))) {
    return(as.numeric(factor(x)))
  }
  return(num_x)
}

samp <- samp_srs %>%
  mutate(across(all_of(data_info_srs$cat_vars), as.factor),
         across(all_of(data_info_srs$num_vars), safe_numeric))

search_space = ps(
  ntree = p_int(lower = 10, upper = 200),
  mtry = p_int(lower = round(ncol(samp) / 3), upper = ncol(samp)),
  nodesize = p_int(lower = 1, upper = 20),
  sampsize_ratio = p_dbl(lower = 0.2, upper = 0.9)
)
tm <- system.time({
  tune_srs <- tune_rf(
    data = samp,
    data_info = data_info_srs,
    search_space = search_space,
    log_path = "rf_tuning_results.csv",
    n_evals = 30, m = 5, folds = 3
  )
})
save(tm, file = "rf_tuning_time.RData")

rf_tuning_results <- read.csv("rf_tuning_results.csv")
rf_tuning_results <- rf_tuning_results %>%
  rowwise() %>%
  mutate(is_pareto = !any(
    rf_tuning_results$acc_loss <= acc_loss & 
      rf_tuning_results$dist_loss <= dist_loss & 
      (rf_tuning_results$acc_loss < acc_loss | rf_tuning_results$dist_loss < dist_loss)
  )) %>%
  ungroup()

pareto_front <- rf_tuning_results %>% filter(is_pareto == TRUE)
vals <- apply(pareto_front[, search_space$ids()], 2, median)
params <- list()
for (i in 1:length(vals)){
  if (search_space$storage_type[i] == "integer"){
    params[[names(vals)[i]]] <- round(vals[i])
    print(round(vals[i]))
  }else{
    params[[names(vals)[i]]] <- round(vals[i], 3)
    print(round(vals[i], 3))
  }
}

saveRDS(params, file = "../../data/Config/best_config_rf.rds")



