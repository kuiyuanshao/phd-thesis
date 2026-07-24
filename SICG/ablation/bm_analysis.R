source("../../utils/ablation_analysis.R")

bm_methods       <- c("None", "Bootstrap", "MC-Dropout", "SWAG")
surv_method_map  <- c(UNPACK = "None", BOOTSTRAP = "Bootstrap", DROPOUT = "MC-Dropout", SWAG = "SWAG")
nutri_method_map <- c(NONE = "None", BOOTSTRAP = "Bootstrap", DROPOUT = "MC-Dropout", SWAG = "SWAG")
bm_models        <- c("Cox", "SBP (Gaussian)", "HTN (Binomial)")

surv_var_class <- c(
  "HbA1c" = "continuous", "rs4506565 1" = "categorical", "rs4506565 2" = "categorical",
  "Age" = "observed", "eGFR" = "continuous", "Insulin" = "continuous", "BMI" = "continuous",
  "Sex TRUE" = "observed", "Insurance TRUE" = "observed",
  "Race AFR" = "observed", "Race AMR" = "observed", "Race SAS" = "observed", "Race EAS" = "observed",
  "Smoke 2" = "categorical", "Smoke 3" = "categorical", "Age:Insulin" = "continuous"
)

nutri_vars_vec <- c("Intercept", "ln_na_true", "c_age", "c_bmi", "high_chol",
                    "usborn", "female", "bkg_pr", "bkg_o", "ln_na_true:c_age")
nutri_var_class <- c(
  "Intercept" = "observed", "ln_na_true" = "continuous", "c_age" = "observed",
  "c_bmi" = "observed", "high_chol" = "observed", "usborn" = "observed",
  "female" = "observed", "bkg_pr" = "observed", "bkg_o" = "observed",
  "ln_na_true:c_age" = "continuous"
)

bm_path <- function(kind, ext, sim_dir = "./simulations") {
  file.path(sim_dir, sprintf("results_SICG_BM_%s.%s", kind, ext))
}


f_sbp_bm <- as.formula("sbp ~ ln_na_true * c_age + c_bmi + high_chol + usborn + female + bkg_pr + bkg_o")
f_htn_bm <- as.formula("hypertension ~ ln_na_true * c_age + c_bmi + high_chol + usborn + female + bkg_pr + bkg_o")

retrieve_bm_nutri <- function(method_dir,
                              sim_dir  = "./simulations/NUTRI",
                              true_dir = "../../code_nutri/data/True",
                              n_rep    = 500) {
  res_sbp <- res_htn <- NULL

  for (i in seq_len(n_rep)) {
    digit <- str_pad(i, 4, pad = "0")
    parquet_file <- file.path(sim_dir, method_dir, paste0(digit, ".parquet"))
    if (!file.exists(parquet_file)) next
    true_file <- file.path(true_dir, paste0(digit, ".RData"))
    if (!file.exists(true_file)) next
    cat("Current: nutri", method_dir, digit, "\n")

    tryCatch({
      e <- new.env(); load(true_file, envir = e); data <- e$data

      multi_impset <- read_parquet(parquet_file) %>%
        group_split(imp_id) %>%
        lapply(function(d) d %>% select(-imp_id)) %>%
        lapply(function(dat) match_types(as.data.frame(dat), data))

      mod_sbp <- lapply(multi_impset, function(d) lm(f_sbp_bm, data = d))
      mod_htn <- lapply(multi_impset, function(d) {
        if (sum(unique(d$hypertension)) == 3) d$hypertension <- d$hypertension - 1
        glm(f_htn_bm, data = d, family = binomial)
      })

      betas_sbp <- do.call(rbind, lapply(mod_sbp, coef))
      betas_htn <- do.call(rbind, lapply(mod_htn, coef))
      res_sbp <- rbind(res_sbp, c(apply(betas_sbp, 2, var), method_dir, digit))
      res_htn <- rbind(res_htn, c(apply(betas_htn, 2, var), method_dir, digit))
    }, error = function(err)
      cat("  [skip] nutri", method_dir, digit, "-", conditionMessage(err), "\n"))
  }

  name_df <- function(x) if (is.null(x)) NULL else
    setNames(as.data.frame(x, stringsAsFactors = FALSE), c(nutri_vars_vec, "Method", "ID"))
  list(SBP = name_df(res_sbp), HTN = name_df(res_htn))
}

retrieve_bm_surv <- function(method,
                             overwrite = FALSE,
                             sim_dir   = "./simulations",
                             true_dir  = "../../code_surv/data/True",
                             n_rep     = 500) {
  out_file <- bm_path(paste0("SURV_", toupper(method)), "RData")
  if (!overwrite && file.exists(out_file)) {
    cat(sprintf("[skip] surv B %s (cached %s)\n", method, out_file))
    e <- new.env(); load(out_file, envir = e); return(e$resultBm)
  }

  resultBm <- NULL
  for (i in seq_len(n_rep)) {
    digit <- str_pad(i, 4, pad = "0")
    true_file <- file.path(true_dir, paste0(digit, ".RData"))
    if (!file.exists(true_file)) next
    parquet_file <- file.path(sim_dir, toupper(method), paste0(digit, ".parquet"))
    if (!file.exists(parquet_file)) next
    cat("Current: surv", method, digit, "\n")

    tryCatch({
      e <- new.env(); load(true_file, envir = e); data <- e$data
      multi_impset <- read_parquet(parquet_file) %>%
        group_split(imp_id) %>%
        lapply(function(d) d %>% select(-imp_id)) %>%
        lapply(function(dat) match_types(as.data.frame(dat), data))
      betas <- do.call(rbind, lapply(multi_impset, function(d) coef(coxph(cox_formula_true, data = d))))
      resultBm <- rbind(resultBm, c(apply(betas, 2, var), toupper(method), digit))
    }, error = function(err)
      cat("  [skip] surv", method, digit, "-", conditionMessage(err), "\n"))
  }

  if (is.null(resultBm)) { cat(sprintf("[warn] no surv reps for %s\n", method)); return(NULL) }
  resultBm <- setNames(as.data.frame(resultBm, stringsAsFactors = FALSE),
                       c(ablation_vars_vec, "Method", "ID"))
  save(resultBm, file = out_file)
  cat(sprintf("surv B %s -> %s (%d reps)\n", method, out_file, nrow(resultBm)))
  resultBm
}


tidy_bm_long <- function(bm_df, model_label, method_map, var_class, vars_vec) {
  if (is.null(bm_df)) return(NULL)
  bm_df %>%
    filter(Method %in% names(method_map)) %>%
    mutate(across(all_of(vars_vec), as.numeric)) %>%
    pivot_longer(all_of(vars_vec), names_to = "Covariate", values_to = "Bm") %>%
    mutate(Model    = factor(model_label, levels = bm_models),
           Method   = factor(unname(method_map[as.character(Method)]), levels = bm_methods),
           VarType  = unname(var_class[Covariate])) %>%
    filter(is.finite(Bm))
}


cat("\n=== surv Cox B_m: recomputing from simulations/<METHOD> ===\n")
surv_methods <- c("UNPACK", "BOOTSTRAP", "DROPOUT", "SWAG")
combined_resultBm <- `rownames<-`(do.call(rbind, lapply(surv_methods, retrieve_bm_surv)), NULL)
surv_long <- tidy_bm_long(combined_resultBm, "Cox",
                          surv_method_map, surv_var_class, ablation_vars_vec)

cat("\n=== nutri SBP / HTN B_m: retrieving from simulations/NUTRI ===\n")
nutri_dirs <- c("NONE", "BOOTSTRAP", "DROPOUT", "SWAG")
nutri_bm   <- lapply(nutri_dirs, retrieve_bm_nutri)

combined_bm_SBP <- `rownames<-`(do.call(rbind, lapply(nutri_bm, `[[`, "SBP")), NULL)
combined_bm_HTN <- `rownames<-`(do.call(rbind, lapply(nutri_bm, `[[`, "HTN")), NULL)
save(combined_bm_SBP, combined_bm_HTN, file = bm_path("NUTRI", "RData"))

sbp_long <- tidy_bm_long(combined_bm_SBP, "SBP (Gaussian)", nutri_method_map, nutri_var_class, nutri_vars_vec)
htn_long <- tidy_bm_long(combined_bm_HTN, "HTN (Binomial)", nutri_method_map, nutri_var_class, nutri_vars_vec)

bm_long <- bind_rows(surv_long, sbp_long, htn_long)
save(bm_long, file = bm_path("COMBINED", "RData"))

imputed_long <- bm_long %>% filter(VarType %in% c("continuous", "categorical"))

bm_summary <- imputed_long %>%
  group_by(Model, VarType, Method) %>%
  summarise(n_cov     = n_distinct(Covariate),
            n_obs     = n(),
            Mean_Bm   = mean(Bm, na.rm = TRUE),
            Median_Bm = median(Bm, na.rm = TRUE),
            SD_Bm     = sd(Bm, na.rm = TRUE),
            .groups   = "drop") %>%
  group_by(Model, VarType) %>%
  mutate(Ratio_vs_None = Mean_Bm / Mean_Bm[Method == "None"]) %>%
  ungroup() %>%
  arrange(Model, VarType, Method)
write.csv(bm_summary, bm_path("SUMMARY", "csv"), row.names = FALSE)

bm_by_cov <- imputed_long %>%
  group_by(Model, VarType, Covariate, Method) %>%
  summarise(n_rep     = n(),
            Mean_Bm   = mean(Bm, na.rm = TRUE),
            Median_Bm = median(Bm, na.rm = TRUE),
            .groups   = "drop") %>%
  arrange(Model, VarType, Covariate, Method)
write.csv(bm_by_cov, bm_path("BY_COVARIATE", "csv"), row.names = FALSE)

plot_bm_box <- function(d) {
  ggplot(d, aes(x = Method, y = Bm, fill = Method)) +
    geom_boxplot(alpha = 0.6, outlier.size = 0.6) +
    geom_hline(data = d %>% group_by(Model, VarType) %>%
                 summarise(ref = median(Bm[Method == "None"], na.rm = TRUE), .groups = "drop"),
               aes(yintercept = ref), lty = 2, color = "red", linewidth = 0.6) +
    facet_wrap(~ Model + VarType, scales = "free_y", ncol = 2) +
    scale_y_log10() +
    scale_fill_viridis_d(option = "cividis") +
    labs(x = NULL, y = expression(B[m] ~ "(between-imputation variance, log scale)")) +
    theme_ablation()
}
ggsave(bm_path("BOXPLOT", "png"), plot_bm_box(imputed_long %>% filter(Bm > 0)),
       width = 14, height = 11, limitsize = FALSE)

cat(sprintf("\nB_m summary       -> %s\nB_m per covariate -> %s\nB_m boxplot       -> %s\n",
            bm_path("SUMMARY", "csv"), bm_path("BY_COVARIATE", "csv"), bm_path("BOXPLOT", "png")))
print(as.data.frame(bm_summary), digits = 4)
