suppressMessages(lapply(c("dplyr"), require, character.only = TRUE))
load("./data/params/truth.RData"); truthX <- truth$beta_X
load("./simulations/results/results_COMBINED.RData")
if ("Coef" %in% names(combined)) combined <- combined[combined$Coef == "X", ]  # focal exposure summary

method_order <- c("VAL", "UNVAL", "CC", "CC-W", "NAIVE", "NAIVE+W", "MICE-PMM", "MICE-RF", "MIXGB",
                  "GAIN", "TABCSDI", "SICG", "SIRD", "RAKING", "WL", "SISL")
summ <- combined %>%
  mutate(Method = factor(Method, levels = method_order)) %>%
  group_by(Method) %>%
  summarise(
    n_rep = dplyr::n(),
    Truth = truthX,
    Mean_Est = mean(Estimate),
    Bias = mean(Estimate) - truthX,
    Pct_Bias = 100 * (mean(Estimate) - truthX) / truthX,
    Emp_SE = sd(Estimate),
    Mean_SE = mean(SE),
    RMSE = sqrt(mean((Estimate - truthX)^2)),
    Coverage = 100 * mean(CI_lo <= truthX & truthX <= CI_hi),
    .groups = "drop"
  ) %>% arrange(Method)

dir.create("./simulations/results", recursive = TRUE, showWarnings = FALSE)
write.csv(summ, "./simulations/results/Summary_Table.csv", row.names = FALSE)
print(as.data.frame(summ), digits = 3)
