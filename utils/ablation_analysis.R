suppressWarnings(lapply(
  c("tidyr", "dplyr", "stringr", "ggplot2", "GGally", "ggrepel",
    "survival", "survey", "mice", "arrow", "mitools",
    "RColorBrewer", "ggh4x", "extrafont", "patchwork"),
  require, character.only = TRUE
))

source("../../code_surv/00_utils_functions.R")
options(survey.lonely.psu = "certainty")

ablation_vars_vec <- c(
  "HbA1c", "rs4506565 1", "rs4506565 2", "Age", "eGFR", "Insulin", "BMI",
  "Sex TRUE", "Insurance TRUE", "Race AFR", "Race AMR", "Race SAS", "Race EAS",
  "Smoke 2", "Smoke 3", "Age:Insulin"
)

ablation_truth <- c(
  1.25, 1.075, 1.15, 1.1, 0.9, 1.1, 1.1, 0.85, 0.9,
  0.90, 1, 0.9, 0.8, 0.85, 0.9, 1.1
)

ablation_range_coef <- list(
  Covariate == "HbA1c"          ~ scale_y_continuous(limits = c(-15, 15)),
  Covariate == "rs4506565 1"    ~ scale_y_continuous(limits = c(-25, 25)),
  Covariate == "rs4506565 2"    ~ scale_y_continuous(limits = c(-25, 25)),
  Covariate == "Age"            ~ scale_y_continuous(limits = c(-30, 30)),
  Covariate == "eGFR"           ~ scale_y_continuous(limits = c(-10, 10)),
  Covariate == "Insulin"        ~ scale_y_continuous(limits = c(-25, 25)),
  Covariate == "BMI"            ~ scale_y_continuous(limits = c(-25, 25)),
  Covariate == "Sex TRUE"       ~ scale_y_continuous(limits = c(-25, 25)),
  Covariate == "Insurance TRUE" ~ scale_y_continuous(limits = c(-25, 25)),
  Covariate == "Race AFR"       ~ scale_y_continuous(limits = c(-50, 50)),
  Covariate == "Race AMR"       ~ scale_y_continuous(limits = c(-50, 50)),
  Covariate == "Race SAS"       ~ scale_y_continuous(limits = c(-50, 50)),
  Covariate == "Race EAS"       ~ scale_y_continuous(limits = c(-50, 50)),
  Covariate == "Smoke 2"        ~ scale_y_continuous(limits = c(-50, 50)),
  Covariate == "Smoke 3"        ~ scale_y_continuous(limits = c(-50, 50)),
  Covariate == "Age:Insulin"    ~ scale_y_continuous(limits = c(-10, 10))
)

.cox_rhs_true <- paste(
  "I((HbA1c - 50) / 5) + rs4506565 + I((AGE - 60) / 5) + I((eGFR - 75) / 10) +",
  "I((Insulin - 15) / 2) + I((BMI - 28) / 2) + SEX + INSURANCE + RACE + SMOKE +",
  "I((AGE - 60) / 5):I((Insulin - 15) / 2)"
)
.cox_rhs_star <- paste(
  "I((HbA1c_STAR - 50) / 5) + rs4506565_STAR + I((AGE - 60) / 5) + I((eGFR_STAR - 75) / 10) +",
  "I((Insulin_STAR - 15) / 2) + I((BMI_STAR - 28) / 2) + SEX + INSURANCE + RACE + SMOKE_STAR +",
  "I((AGE - 60) / 5):I((Insulin_STAR - 15) / 2)"
)
cox_formula_true <- as.formula(paste("Surv(T_I, EVENT) ~", .cox_rhs_true))
cox_formula_star <- as.formula(paste("Surv(T_I, EVENT) ~", .cox_rhs_star))


kept_methods <- function(ablation) {
  kept <- Filter(function(s) isTRUE(s$outcome == "kept"), ablation)
  out  <- vapply(kept, function(s) as.character(s$method), character(1))
  out[!is.na(out)]
}

candidate_methods <- function(ablation) {
  m <- vapply(ablation, function(s) as.character(s$method), character(1))
  m[!is.na(m)]
}

ablation_table <- function(ablation) {
  do.call(rbind, lapply(ablation, function(s) data.frame(
    phase   = s$phase,
    group   = s$group,
    type    = if (isTRUE(s$parallel)) "compete" else "add",
    method  = if (is.null(s$method) || is.na(s$method)) "-" else s$method,
    outcome = s$outcome,
    note    = if (is.null(s$note)) "" else s$note,
    stringsAsFactors = FALSE
  )))
}

results_path <- function(method, sim_dir = "./simulations") {
  file.path(sim_dir, paste0("results_", toupper(method), ".RData"))
}
combined_path <- function(study, sim_dir = "./simulations") {
  file.path(sim_dir, paste0("results_", toupper(study), "_COMBINED.RData"))
}
summary_path <- function(study, sim_dir = "./simulations") {
  file.path(sim_dir, paste0("results_", toupper(study), "_SUMMARY.csv"))
}
boxplot_path <- function(study, sim_dir = "./simulations") {
  file.path(sim_dir, paste0("results_", toupper(study), "_BIAS_BOXPLOT.png"))
}


analyze_tuning <- function(file_prefix, cat_params = c(), cont_params = c()) {
  results_file    <- paste0(file_prefix, "_tuning_results.csv")
  importance_file <- paste0(file_prefix, "_tuning_results_importance.csv")

  tuning_results <- read.csv(results_file) %>%
    drop_na(avg_mse, avg_ce, avg_ed) %>%
    mutate(
      metric_1 = avg_mse + avg_ce,
      metric_2 = avg_ed,
      total_combined_loss = metric_1 + metric_2
    )

  importance <- read.csv(importance_file) %>%
    arrange(desc(importance))

  ordered_params <- trimws(importance$parameter)
  tuning_results_clean <- tuning_results %>%
    rowwise() %>%
    mutate(is_pareto = !any(
      tuning_results$metric_1 <= metric_1 &
        tuning_results$metric_2 <= metric_2 &
        (tuning_results$metric_1 < metric_1 | tuning_results$metric_2 < metric_2)
    )) %>%
    ungroup()

  pareto_front <- tuning_results_clean %>% filter(is_pareto == TRUE)

  min_m1 <- min(pareto_front$metric_1)
  max_m1 <- max(pareto_front$metric_1)
  min_m2 <- min(pareto_front$metric_2)
  max_m2 <- max(pareto_front$metric_2)

  tuning_results_clean <- tuning_results_clean %>%
    mutate(
      norm_m1 = (metric_1 - min_m1) / (max_m1 - min_m1 + 1e-9),
      norm_m2 = (metric_2 - min_m2) / (max_m2 - min_m2 + 1e-9),
      dist_to_utopia = sqrt((norm_m1 - 0)^2 + (norm_m2 - 0)^2)
    )

  top_models <- tuning_results_clean %>%
    arrange(dist_to_utopia) %>%
    head(5)

  cat("\n=== Parameter Summaries (Ordered by Importance):", file_prefix, "===\n")
  for (p in ordered_params) {
    if (p %in% colnames(top_models)) {
      if (p %in% cat_params) {
        cat("\nTable for", p, ":\n")
        print(table(top_models[[p]]))
      } else {
        cat("\nMedian", p, ":", median(top_models[[p]], na.rm = TRUE), "\n")
      }
    }
  }

  p1 <- ggplot(tuning_results_clean, aes(x = metric_1, y = metric_2)) +
    geom_point(aes(color = is_pareto, alpha = 0.7)) +
    geom_point(data = head(top_models, 5), color = "red", shape = 1, size = 6, stroke = 1.5) +
    scale_color_manual(values = c("FALSE" = "grey60", "TRUE" = "blue")) +
    labs(
      title = paste("Macro View:", file_prefix),
      subtitle = "Blue points form the Pareto Front. Red circles indicate the Top 5 optimal candidates.",
      x = "avg_mse + avg_ce (Minimize)",
      y = "avg_ed (Minimize)",
      color = "Is Pareto Optimal?"
    ) +
    theme_minimal()
  print(p1)

  top_5_long <- head(top_models, 5) %>%
    select(trial_number, avg_mse, avg_ce, avg_ed) %>%
    pivot_longer(cols = c(avg_mse, avg_ce, avg_ed),
                 names_to = "loss_component",
                 values_to = "value") %>%
    mutate(trial_number = as.factor(trial_number))

  p2 <- ggplot(top_5_long, aes(x = reorder(trial_number, value), y = value, fill = loss_component)) +
    geom_bar(stat = "identity", position = "dodge") +
    scale_fill_brewer(palette = "Set2", labels = c("Categorical Error", "Energy Distance", "MSE")) +
    labs(
      title = paste("Loss Breakdown of Top 5 Models:", file_prefix),
      x = "Trial Number",
      y = "Raw Loss Value",
      fill = "Loss Component"
    ) +
    theme_minimal()
  print(p2)

  for (p in cat_params) {
    if (p %in% colnames(tuning_results)) {
      p_box <- ggplot(tuning_results) +
        geom_boxplot(aes(x = as.factor(.data[[p]]), y = total_combined_loss)) +
        labs(x = p, title = paste("Boxplot of Combined Loss by", p)) +
        theme_minimal()
      print(p_box)
    }
  }

  for (p in cont_params) {
    if (p %in% colnames(tuning_results)) {
      p_smooth <- ggplot(tuning_results, aes(x = .data[[p]], y = total_combined_loss)) +
        geom_point() +
        geom_smooth() +
        labs(x = p, title = paste("Trend of Combined Loss vs", p)) +
        theme_minimal()
      print(p_smooth)
    }
  }

  invisible(list(clean_data = tuning_results_clean, top_models = top_models))
}

analyze_tuning_set <- function(components) {
  lapply(components, function(cmp) {
    analyze_tuning(cmp$prefix, cat_params = cmp$cat, cont_params = cmp$cont)
  })
}


retrieveEst <- function(method,
                        track_bm  = FALSE,
                        overwrite = FALSE,
                        sim_dir   = "./simulations",
                        true_dir  = "../../code_surv/data/True",
                        cc_dir    = "../../code_surv/data/SampleE/SRS",
                        n_rep     = 500) {
  out_file <- results_path(method, sim_dir)
  if (!overwrite && file.exists(out_file)) {
    cat(sprintf("[skip] %s already retrieved (%s)\n", method, out_file))
    return(invisible(out_file))
  }

  n_var <- length(ablation_vars_vec)
  resultCoeff <- resultStdError <- resultCI <- resultBm <- NULL

  for (i in seq_len(n_rep)) {
    digit <- str_pad(i, 4, pad = "0")
    true_file <- file.path(true_dir, paste0(digit, ".RData"))
    if (!file.exists(true_file)) next
    cat("Current:", digit, "\n")
    load(true_file)

    if (method == "VAL") {
      cox.mod <- coxph(cox_formula_true, data = data)
    } else if (method == "UNVAL") {
      cox.mod <- coxph(cox_formula_star, data = data)
    } else if (method == "CC") {
      cc_file <- file.path(cc_dir, paste0(digit, ".csv"))
      if (!file.exists(cc_file)) next
      samp <- match_types(read.csv(cc_file), data)
      cox.mod <- coxph(cox_formula_true, data = samp)
    } else {
      parquet_file <- file.path(sim_dir, method, paste0(digit, ".parquet"))
      if (!file.exists(parquet_file)) next

      multi_impset <- read_parquet(parquet_file) %>%
        group_split(imp_id) %>%
        lapply(function(d) d %>% select(-imp_id)) %>%
        lapply(function(dat) match_types(dat, data))
      imp.mids <- imputationList(multi_impset)
      # Explicit data= so the shared formula's vars resolve in each imputed set,
      # not in the global env where the formula was built.
      cox.mod <- lapply(imp.mids$imputations, function(d) coxph(cox_formula_true, data = d))
      pooled <- MIcombine(cox.mod)
      capture.output(sumry <- summary(pooled), file = nullfile())

      resultCoeff    <- rbind(resultCoeff,    c(exp(sumry$results), toupper(method), digit))
      resultStdError <- rbind(resultStdError, c(sumry$se,           toupper(method), digit))
      resultCI       <- rbind(resultCI,       c(exp(sumry$`(lower`), exp(sumry$`upper)`), toupper(method), digit))
      if (track_bm) {
        betas <- do.call(rbind, MIextract(cox.mod, fun = coef))
        resultBm <- rbind(resultBm, c(apply(betas, 2, var), toupper(method), digit))
      }
      next
    }

    resultCoeff    <- rbind(resultCoeff,    c(exp(coef(cox.mod)),            toupper(method), digit))
    resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))),     toupper(method), digit))
    resultCI       <- rbind(resultCI,       c(exp(confint(cox.mod)[, 1]), exp(confint(cox.mod)[, 2]), toupper(method), digit))
    if (track_bm) {
      resultBm <- rbind(resultBm, c(rep(NA, n_var), toupper(method), digit))
    }
  }

  if (is.null(resultCoeff)) {
    cat(sprintf("[warn] no replicates found for '%s'; nothing saved.\n", method))
    return(invisible(NULL))
  }

  resultCoeff    <- setNames(as.data.frame(resultCoeff),    c(ablation_vars_vec, "Method", "ID"))
  resultStdError <- setNames(as.data.frame(resultStdError), c(ablation_vars_vec, "Method", "ID"))
  resultCI       <- setNames(as.data.frame(resultCI),
                             c(paste0(ablation_vars_vec, ".lower"),
                               paste0(ablation_vars_vec, ".upper"), "Method", "ID"))

  if (track_bm) {
    resultBm <- setNames(as.data.frame(resultBm), c(ablation_vars_vec, "Method", "ID"))
    save(resultCoeff, resultStdError, resultCI, resultBm, file = out_file)
  } else {
    save(resultCoeff, resultStdError, resultCI, file = out_file)
  }
  invisible(out_file)
}

combine_results <- function(methods, out_file,
                            track_bm = FALSE,
                            sim_dir  = "./simulations") {
  filenames <- results_path(methods, sim_dir)
  list_coeff <- list_ci <- list_se <- list_bm <- list()
  for (f in filenames) {
    if (!file.exists(f)) next
    e <- new.env()
    load(f, envir = e)
    list_coeff[[f]] <- e$resultCoeff
    list_ci[[f]]    <- e$resultCI
    list_se[[f]]    <- e$resultStdError
    if (track_bm) list_bm[[f]] <- e$resultBm
  }

  combined_resultCoeff    <- `rownames<-`(do.call(rbind, list_coeff), NULL)
  combined_resultCI       <- `rownames<-`(do.call(rbind, list_ci),    NULL)
  combined_resultStdError <- `rownames<-`(do.call(rbind, list_se),    NULL)

  if (track_bm) {
    combined_resultBm <- `rownames<-`(do.call(rbind, list_bm), NULL)
    save(combined_resultCoeff, combined_resultCI, combined_resultStdError,
         combined_resultBm, file = out_file)
  } else {
    save(combined_resultCoeff, combined_resultCI, combined_resultStdError,
         file = out_file)
  }
  invisible(out_file)
}

run_ablation_study <- function(study, ablation,
                               baselines = c("VAL", "UNVAL", "CC"),
                               track_bm  = FALSE,
                               overwrite = FALSE,
                               sim_dir   = "./simulations") {
  active   <- c(baselines, candidate_methods(ablation))
  out_file <- combined_path(study, sim_dir)
  cat(sprintf("\n=== %s study: %d methods -> %s ===\n",
              toupper(study), length(active), out_file))
  cat("    ", paste(active, collapse = ", "), "\n", sep = "")

  for (m in active) {
    retrieveEst(m, track_bm = track_bm, overwrite = overwrite, sim_dir = sim_dir)
  }
  combine_results(active, out_file = out_file, track_bm = track_bm, sim_dir = sim_dir)

  list(active = active, combined_file = out_file)
}


compute_summary <- function(combined_file, methods, ablation = NULL,
                            vars_vec = ablation_vars_vec, truth = ablation_truth) {
  env <- new.env()
  load(combined_file, envir = env)
  coeff <- env$combined_resultCoeff
  se    <- env$combined_resultStdError
  ci    <- env$combined_resultCI

  z <- qnorm(0.975)
  keep_levels <- setdiff(toupper(methods), "VAL")
  truth_df <- data.frame(Covariate = vars_vec, Truth = truth, stringsAsFactors = FALSE)

  long_coeff <- coeff %>%
    filter(Method != "VAL") %>%
    pivot_longer(all_of(vars_vec), names_to = "Covariate", values_to = "Coefficient") %>%
    mutate(Coefficient = as.numeric(Coefficient)) %>%
    left_join(truth_df, by = "Covariate") %>%
    mutate(Percent_Bias = (Coefficient - Truth) / Truth * 100)

  long_se <- se %>%
    filter(Method != "VAL") %>%
    pivot_longer(all_of(vars_vec), names_to = "Covariate", values_to = "ModelSE") %>%
    mutate(ModelSE = as.numeric(ModelSE)) %>%
    select(Method, ID, Covariate, ModelSE)

  long_ci <- ci %>%
    filter(Method != "VAL") %>%
    pivot_longer(-c(Method, ID), names_to = "key", values_to = "value") %>%
    mutate(bound = ifelse(grepl("\\.lower$", key), "Lower", "Upper"),
           Covariate = sub("\\.(lower|upper)$", "", key),
           value = as.numeric(value)) %>%
    select(Method, ID, Covariate, bound, value) %>%
    pivot_wider(names_from = bound, values_from = value)

  per_rep <- long_coeff %>%
    select(Method, ID, Covariate, Coefficient, Truth, Percent_Bias) %>%
    left_join(long_se, by = c("Method", "ID", "Covariate")) %>%
    left_join(long_ci, by = c("Method", "ID", "Covariate")) %>%
    group_by(Method, Covariate) %>%
    mutate(EmpSE         = sd(Coefficient, na.rm = TRUE),
           mean_bias_log = mean(log(Coefficient), na.rm = TRUE) - log(Truth)) %>%
    ungroup()

  summary <- per_rep %>%
    group_by(Method, Covariate) %>%
    summarise(
      Mean_Percent_Bias        = mean(Percent_Bias, na.rm = TRUE),
      Empirical_SE             = first(EmpSE),
      RMSE                     = sqrt(mean((Coefficient - Truth)^2, na.rm = TRUE)),
      Nominal_Coverage         = mean(Truth >= Lower & Truth <= Upper, na.rm = TRUE),
      Oracle_Coverage_corrSE   = mean(Truth >= Coefficient - z * EmpSE &
                                      Truth <= Coefficient + z * EmpSE, na.rm = TRUE),
      Oracle_Coverage_corrBias = mean(log(Truth) >= (log(Coefficient) - mean_bias_log) - z * ModelSE &
                                      log(Truth) <= (log(Coefficient) - mean_bias_log) + z * ModelSE, na.rm = TRUE),
      .groups = "drop"
    )

  if (!is.null(ablation)) {
    outc <- c()
    for (s in ablation) {
      if (!is.null(s$method) && !is.na(s$method)) outc[toupper(as.character(s$method))] <- s$outcome
    }
    summary <- summary %>%
      mutate(Outcome = ifelse(Method %in% names(outc), unname(outc[Method]),
                       ifelse(Method %in% c("UNVAL", "CC"), "baseline", NA))) %>%
      relocate(Outcome, .after = Method)
  }

  summary <- summary %>%
    mutate(Method    = factor(Method, levels = keep_levels),
           Covariate = factor(Covariate, levels = vars_vec)) %>%
    arrange(Method, Covariate)

  coeff_long <- long_coeff %>%
    mutate(Method    = factor(Method, levels = keep_levels),
           Covariate = factor(Covariate, levels = vars_vec))

  list(summary = summary, coeff_long = coeff_long)
}

summarise_study <- function(study, combined_file, methods, ablation = NULL,
                            width = 18, height = 12, sim_dir = "./simulations") {
  res <- compute_summary(combined_file, methods, ablation)

  csv <- summary_path(study, sim_dir)
  write.csv(res$summary, csv, row.names = FALSE)

  png <- boxplot_path(study, sim_dir)
  ggsave(png, plot_bias_box(res$coeff_long), width = width, height = height, limitsize = FALSE)

  cat(sprintf("Summary table -> %s\nBias boxplot  -> %s\n", csv, png))
  invisible(res)
}

theme_ablation <- function(base_size = 14) {
  theme_bw(base_size = base_size, base_family = "Times New Roman") +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "grey95")
    )
}

plot_bias_box <- function(coeff_long) {
  ggplot(coeff_long) +
    geom_boxplot(aes(x = Method, y = Percent_Bias, fill = Method), alpha = 0.6, outlier.size = 0.8) +
    geom_hline(yintercept = 0, lty = 2, color = "red", linewidth = 0.8) +
    facet_wrap(~ Covariate, scales = "free", ncol = 4) +
    scale_fill_viridis_d(option = "cividis") +
    labs(x = NULL, y = "Percentage Bias (%)") +
    theme_ablation() +
    facetted_pos_scales(y = ablation_range_coef)
}
