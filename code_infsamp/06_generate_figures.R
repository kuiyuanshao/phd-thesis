suppressMessages(lapply(c("ggplot2", "dplyr"), require, character.only = TRUE))
load("./data/params/truth.RData"); truthX <- truth$beta_X
load("./simulations/results/results_COMBINED.RData")
if ("Coef" %in% names(combined)) combined <- combined[combined$Coef == "X", ]  # exposure figure

method_order <- c("VAL", "UNVAL", "CC", "NAIVE", "NAIVE+W", "MICE-PMM", "MICE-RF", "MIXGB",
                  "GAIN", "TABCSDI", "RAKING", "WL", "SISL")
combined <- combined %>%
  mutate(Pct_Bias = 100 * (Estimate - truthX) / truthX,
         Method = factor(Method, levels = method_order))

p <- ggplot(combined, aes(Method, Pct_Bias)) +
  geom_hline(yintercept = 0, linetype = 2, colour = "grey50") +
  geom_boxplot(outlier.size = 0.4, fill = "#A6CEE3", width = 0.6) +
  labs(x = NULL, y = expression("Percent bias in " * beta[X] * " (%)"),
       title = "Simulation Design 0 (SISL): percent bias by estimator") +
  theme_bw(base_size = 12)

out <- "./simulations/results"; dir.create(out, recursive = TRUE, showWarnings = FALSE)
ggsave(file.path(out, "design0_pctbias.pdf"), p, width = 7, height = 4.5)
ggsave(file.path(out, "design0_pctbias.png"), p, width = 7, height = 4.5, dpi = 200)
cat("Wrote design0_pctbias.{pdf,png} to", out, "\n")
