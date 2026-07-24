lapply(c("dplyr", "tidyr", "stringr", "ggplot2"), require, character.only = TRUE)

is_nesi <- grepl("nesi.org.nz", Sys.info()["nodename"]) || dir.exists("/opt/nesi")
if (is_nesi) {
  sim_root <- path.expand("~/00_nesi_projects/uoa03789_nobackup/simulations/code_surv")
} else {
  sim_root <- "./simulations"
}

vars_vec <- c("HbA1c", "rs4506565 1", "rs4506565 2", "Age",
              "eGFR", "Insulin", "BMI", "Sex TRUE", "Insurance TRUE",
              "Race AFR", "Race AMR", "Race SAS", "Race EAS", "Smoke 2", "Smoke 3",
              "Age:Insulin")
sample_types <- c("SampleOE", "SampleE")
z <- qnorm(0.975)

combined_file <- file.path(sim_root, "results", "results_COMBINED.RData")
if (!file.exists(combined_file)) stop("missing combined file: ", combined_file)
combined_env <- new.env(); load(combined_file, envir = combined_env)

summarise_one <- function(sample_type) {
  coeff <- combined_env[[paste0("combined_resultCoeff_", sample_type)]]
  se    <- combined_env[[paste0("combined_resultStdError_", sample_type)]]
  ci    <- combined_env[[paste0("combined_resultCI_", sample_type)]]
  if (is.null(coeff)) { warning("no combined objects for ", sample_type); return(NULL) }

  long_coeff <- coeff %>%
    pivot_longer(all_of(vars_vec), names_to = "Covariate", values_to = "Coefficient") %>%
    mutate(Coefficient = as.numeric(Coefficient))

  # Truth = mean of the validated-data (VAL) estimate across replicates.
  truth <- long_coeff %>%
    filter(Method == "VAL") %>%
    group_by(Covariate) %>%
    summarise(Truth = mean(Coefficient, na.rm = TRUE), .groups = "drop")

  long_se <- se %>%
    pivot_longer(all_of(vars_vec), names_to = "Covariate", values_to = "ModelSE") %>%
    mutate(ModelSE = as.numeric(ModelSE)) %>%
    select(Method, ID, Covariate, ModelSE)

  long_ci <- ci %>%
    pivot_longer(-c(Design, Method, ID), names_to = "key", values_to = "value") %>%
    mutate(bound = ifelse(grepl("\\.lower$", key), "Lower", "Upper"),
           Covariate = sub("\\.(lower|upper)$", "", key),
           value = as.numeric(value)) %>%
    select(Method, ID, Covariate, bound, value) %>%
    pivot_wider(names_from = bound, values_from = value)

  per_rep <- long_coeff %>%
    filter(Method != "VAL") %>%
    select(Method, ID, Covariate, Coefficient) %>%
    left_join(truth,   by = "Covariate") %>%
    left_join(long_se, by = c("Method", "ID", "Covariate")) %>%
    left_join(long_ci, by = c("Method", "ID", "Covariate")) %>%
    group_by(Method, Covariate) %>%
    mutate(EmpSE = sd(Coefficient, na.rm = TRUE),
           mean_bias_log = mean(log(Coefficient), na.rm = TRUE) - log(Truth)) %>%
    ungroup()

  per_rep %>%
    group_by(Method, Covariate) %>%
    summarise(
      Mean_Percent_Bias    = mean((Coefficient - Truth) / Truth * 100, na.rm = TRUE),
      Empirical_SE         = first(EmpSE),
      Mean_SE              = mean(ModelSE, na.rm = TRUE),   # mean model-based SE (Mean SE)
      RMSE                 = sqrt(mean((Coefficient - Truth)^2, na.rm = TRUE)),
      Nominal_Coverage     = mean(Truth >= Lower & Truth <= Upper, na.rm = TRUE),
      Oracle_Coverage_SE   = mean(Truth >= Coefficient - z * EmpSE &
                                  Truth <= Coefficient + z * EmpSE, na.rm = TRUE),
      Oracle_Coverage_Bias = mean(log(Truth) >= (log(Coefficient) - mean_bias_log) - z * ModelSE &
                                  log(Truth) <= (log(Coefficient) - mean_bias_log) + z * ModelSE, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(SampleType = sample_type)
}

summary_table <- bind_rows(lapply(sample_types, summarise_one)) %>%
  mutate(Covariate = factor(Covariate, levels = vars_vec)) %>%
  select(SampleType, Method, Covariate, everything()) %>%
  arrange(SampleType, Method, Covariate)

results_dir <- file.path(sim_root, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
print(as.data.frame(summary_table))
write.csv(summary_table, file = file.path(results_dir, "Summary_Table.csv"), row.names = FALSE)
cat(sprintf("Summary table -> %s\n", file.path(results_dir, "Summary_Table.csv")))
