lapply(c("ggplot2", "dplyr", "tidyr", "stringr"), require, character.only = TRUE)
source("00_utils_functions.R")

methods <- c("VAL", "UNVAL", "CC", "MICE-PMM", "MICE-RF", "MIXGB", "RAKING", "SICG", "SIRD", "TABCSDI", "GAIN")
designs_order <- toupper(c("SRS", "RS", "WRS", "SFS", "ODS_TAIL", "Neyman_ODS", "Neyman_INF"))
vars_vec <- c("Intercept", "ln_na_true", "c_age", "c_bmi", "high_chol", "usborn", "female", "bkg_pr", "bkg_o", "ln_na_true:c_age")

load("./simulations/results/results_COMBINED.RData")

dgp_htn <- data.frame(
  Covariate = vars_vec,
  Truth = c(-1.91298009, 1.43901944, 0.08715117, 0.05645171, 0.37938242, -0.21958482, -0.11976800, -0.17504836, -0.84484199, 0.005)
)

dgp_sbp <- data.frame(
  Covariate = vars_vec,
  Truth = c(126.2498900, 27.4725275, 0.4060319, 0.1703107, 1.6108133, -0.5636760, -8.9318846, -1.6697425, -2.3219322, 0.15)
)

process_metrics <- function(coeff_df, se_df, ci_df, truth_long) {
  
  coeff_long <- coeff_df %>%
    filter(!Method %in% c("VAL")) %>%
    mutate(Design = toupper(as.character(Design)),
           Method = toupper(as.character(Method))) %>%
    pivot_longer(cols = all_of(vars_vec), names_to = "Covariate", values_to = "Estimate") %>%
    mutate(Estimate = as.numeric(Estimate),
           Covariate = as.character(Covariate)) %>%
    left_join(truth_long, by = "Covariate") %>%
    mutate(Raw_Bias = Estimate - Truth,
           Percent_Bias = (Estimate - Truth) / Truth * 100) %>%
    mutate(Method = factor(Method, levels = methods),
           Design = factor(Design, levels = designs_order),
           Covariate = factor(Covariate, levels = vars_vec))
  
  rmse_res <- coeff_long %>%
    group_by(Design, Method, Covariate) %>%
    summarise(RMSE = sqrt(mean(Raw_Bias^2, na.rm = TRUE)),
              # Rel_RMSE is scale-free (% of true coef), so averageable across covariates.
              Rel_RMSE = sqrt(mean(Raw_Bias^2, na.rm = TRUE)) / abs(first(Truth)) * 100,
              .groups = "drop")
  
  ci_long_lower <- ci_df %>%
    filter(!Method %in% c("VAL")) %>%
    mutate(Design = toupper(as.character(Design)),
           Method = toupper(as.character(Method))) %>%
    select(Design, Method, ID, ends_with(".lower")) %>%
    pivot_longer(cols = ends_with(".lower"), names_to = "Covariate", values_to = "Lower") %>%
    mutate(Covariate = str_remove(as.character(Covariate), "\\.lower$"),
           Lower = as.numeric(Lower))
  
  ci_long_upper <- ci_df %>%
    filter(!Method %in% c("VAL")) %>%
    mutate(Design = toupper(as.character(Design)),
           Method = toupper(as.character(Method))) %>%
    select(Design, Method, ID, ends_with(".upper")) %>%
    pivot_longer(cols = ends_with(".upper"), names_to = "Covariate", values_to = "Upper") %>%
    mutate(Covariate = str_remove(as.character(Covariate), "\\.upper$"),
           Upper = as.numeric(Upper))
  
  nominal_coverage <- ci_long_lower %>%
    left_join(ci_long_upper, by = c("Design", "Method", "ID", "Covariate")) %>%
    left_join(truth_long, by = "Covariate") %>%
    mutate(Covered = ifelse(Truth >= Lower & Truth <= Upper, 1, 0)) %>%
    group_by(Design, Method, Covariate) %>%
    summarise(Nominal_Coverage = mean(Covered, na.rm = TRUE), .groups = "drop") %>%
    mutate(Method = factor(Method, levels = methods),
           Design = factor(Design, levels = designs_order),
           Covariate = factor(Covariate, levels = vars_vec))
  
  emp_se <- coeff_long %>%
    group_by(Design, Method, Covariate) %>%
    summarise(EmpSE = sd(Estimate, na.rm = TRUE), .groups = "drop")
  
  oracle_coverage <- coeff_long %>%
    left_join(emp_se, by = c("Design", "Method", "Covariate")) %>%
    mutate(Oracle_Lower = Estimate - qnorm(0.975) * EmpSE,
           Oracle_Upper = Estimate + qnorm(0.975) * EmpSE,
           Oracle_Covered = ifelse(Truth >= Oracle_Lower & Truth <= Oracle_Upper, 1, 0)) %>%
    group_by(Design, Method, Covariate) %>%
    summarise(Oracle_Coverage = mean(Oracle_Covered, na.rm = TRUE), .groups = "drop") %>%
    mutate(Method = factor(Method, levels = methods),
           Design = factor(Design, levels = designs_order),
           Covariate = factor(Covariate, levels = vars_vec))

  se_long <- se_df %>%
    filter(!Method %in% c("VAL")) %>%
    mutate(Design = toupper(as.character(Design)),
           Method = toupper(as.character(Method))) %>%
    pivot_longer(cols = all_of(vars_vec), names_to = "Covariate", values_to = "ModelSE") %>%
    mutate(ModelSE = as.numeric(ModelSE), Covariate = as.character(Covariate))

  mean_se <- se_long %>%                                   # mean model-based SE (Mean SE)
    group_by(Design, Method, Covariate) %>%
    summarise(Mean_SE = mean(ModelSE, na.rm = TRUE), .groups = "drop") %>%
    mutate(Method = factor(Method, levels = methods),
           Design = factor(Design, levels = designs_order),
           Covariate = factor(Covariate, levels = vars_vec))

  oracle_coverage_bias <- coeff_long %>%
    mutate(Method = as.character(Method), Design = as.character(Design),
           Covariate = as.character(Covariate)) %>%
    left_join(se_long, by = c("Design", "Method", "ID", "Covariate")) %>%
    group_by(Design, Method, Covariate) %>%
    mutate(Mean_Raw_Bias = mean(Raw_Bias, na.rm = TRUE),
           BC_Lower = (Estimate - Mean_Raw_Bias) - qnorm(0.975) * ModelSE,
           BC_Upper = (Estimate - Mean_Raw_Bias) + qnorm(0.975) * ModelSE,
           BC_Covered = ifelse(Truth >= BC_Lower & Truth <= BC_Upper, 1, 0)) %>%
    summarise(Oracle_Coverage_Bias = mean(BC_Covered, na.rm = TRUE), .groups = "drop") %>%
    mutate(Method = factor(Method, levels = methods),
           Design = factor(Design, levels = designs_order),
           Covariate = factor(Covariate, levels = vars_vec))

  list(coeff_long = coeff_long, rmse_res = rmse_res, nominal_coverage = nominal_coverage,
       oracle_coverage = oracle_coverage, oracle_coverage_bias = oracle_coverage_bias, mean_se = mean_se)
}

res_sbp <- process_metrics(combined_resultCoeff_SBP, combined_resultStdError_SBP, combined_resultCI_SBP, dgp_sbp)
res_htn <- process_metrics(combined_resultCoeff_HTN, combined_resultStdError_HTN, combined_resultCI_HTN, dgp_htn)


build_summary <- function(res, outcome) {
  res$coeff_long %>%
    group_by(Design, Method, Covariate) %>%
    summarise(Mean_Percent_Bias = mean(Percent_Bias, na.rm = TRUE),
              Empirical_SE = sd(Estimate, na.rm = TRUE), .groups = "drop") %>%
    left_join(res$rmse_res,             by = c("Design", "Method", "Covariate")) %>%
    left_join(res$nominal_coverage,     by = c("Design", "Method", "Covariate")) %>%
    left_join(res$oracle_coverage,      by = c("Design", "Method", "Covariate")) %>%
    left_join(res$oracle_coverage_bias, by = c("Design", "Method", "Covariate")) %>%
    left_join(res$mean_se,              by = c("Design", "Method", "Covariate")) %>%
    mutate(Outcome = outcome) %>%
    select(Outcome, Design, Method, Covariate, everything()) %>%
    arrange(Outcome, Design, Method, Covariate)
}

summary_table <- bind_rows(
  build_summary(res_sbp, "SBP"),
  build_summary(res_htn, "HTN")
)

cov_groups <- c(c_age = "Numeric covariates", c_bmi = "Numeric covariates",
                high_chol = "Binary & categorical covariates",
                usborn    = "Binary & categorical covariates",
                female    = "Binary & categorical covariates",
                bkg_pr    = "Binary & categorical covariates",
                bkg_o     = "Binary & categorical covariates")

group_summary <- summary_table %>%
  mutate(Covariate = as.character(Covariate)) %>%
  filter(Covariate %in% names(cov_groups)) %>%
  mutate(Covariate = unname(cov_groups[Covariate])) %>%
  group_by(Outcome, Design, Method, Covariate) %>%
  summarise(Mean_Percent_Bias    = mean(Mean_Percent_Bias,    na.rm = TRUE),
            Empirical_SE         = mean(Empirical_SE,         na.rm = TRUE),
            Mean_SE              = mean(Mean_SE,              na.rm = TRUE),
            RMSE                 = mean(RMSE,                 na.rm = TRUE),
            Rel_RMSE             = mean(Rel_RMSE,             na.rm = TRUE),
            Nominal_Coverage     = mean(Nominal_Coverage,     na.rm = TRUE),
            Oracle_Coverage      = mean(Oracle_Coverage,      na.rm = TRUE),
            Oracle_Coverage_Bias = mean(Oracle_Coverage_Bias, na.rm = TRUE),
            .groups = "drop")

summary_table <- summary_table %>%
  mutate(Covariate = as.character(Covariate)) %>%
  bind_rows(group_summary)

print(as.data.frame(summary_table))
write.csv(summary_table, file = "./simulations/results/Summary_Table.csv", row.names = FALSE)
