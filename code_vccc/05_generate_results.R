lapply(c("dplyr", "tidyr", "ggplot2"), require, character.only = TRUE)

sim_root <- "./simulations"
out_dir  <- file.path(sim_root, "results")
z <- qnorm(0.975)

load(file.path(out_dir, "results_COMBINED.RData"))   # combined_long
combined_long <- combined_long %>%
  mutate(across(c(Coeff, SE, Lower, Upper), as.numeric))

# UNVAL stores naive (attenuated) coefficients under the *_nv names; map them onto
# the validated names so the truth join below does not drop them to NA.
combined_long <- combined_long %>%
  mutate(Covariate = ifelse(Method == "UNVAL", sub("_nv$", "_v", Covariate), Covariate))

truth <- combined_long %>%
  filter(Method == "VAL") %>%
  group_by(Model, Covariate) %>%
  summarise(Truth = mean(Coeff, na.rm = TRUE), .groups = "drop")

dat <- combined_long %>%
  filter(Method != "VAL") %>%
  left_join(truth, by = c("Model", "Covariate")) %>%
  group_by(Model, Design, Method, Covariate) %>%
  mutate(EmpSE         = sd(Coeff, na.rm = TRUE),
         mean_bias     = mean(Coeff, na.rm = TRUE) - Truth,
         mean_log_bias = mean(log(Coeff), na.rm = TRUE) - log(Truth)) %>%
  ungroup()

summary_table <- dat %>%
  group_by(Model, Design, Method, Covariate) %>%
  summarise(
    Mean_Percent_Bias    = mean((Coeff - Truth) / Truth * 100, na.rm = TRUE),
    Empirical_SE         = first(EmpSE),
    Mean_SE              = mean(SE, na.rm = TRUE),   # mean model-based SE (Mean SE)
    RMSE                 = sqrt(mean((Coeff - Truth)^2, na.rm = TRUE)),
    Rel_RMSE             = sqrt(mean((Coeff - Truth)^2, na.rm = TRUE)) / abs(first(Truth)) * 100,
    Nominal_Coverage     = mean(Truth >= Lower & Truth <= Upper, na.rm = TRUE),
    Oracle_Coverage_SE   = mean(Truth >= Coeff - z * EmpSE &
                                Truth <= Coeff + z * EmpSE, na.rm = TRUE),
    # COX bias-corrected coverage is on the log scale (HR ratio), so the jump factor cancels.
    Oracle_Coverage_Bias = if (first(Model) == "COX")
        mean(log(Truth) >= (log(Coeff) - mean_log_bias) - z * SE &
             log(Truth) <= (log(Coeff) - mean_log_bias) + z * SE, na.rm = TRUE)
      else
        mean(Truth >= (Coeff - mean_bias) - z * SE &
             Truth <= (Coeff - mean_bias) + z * SE, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(Model, Design, Method, Covariate)

print(as.data.frame(summary_table))
write.csv(summary_table, file = file.path(out_dir, "Summary_Table.csv"), row.names = FALSE)
cat(sprintf("Summary table -> %s\n", file.path(out_dir, "Summary_Table.csv")))
