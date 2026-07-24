source("../../../utils/tuning_analysis.R")

search_space <- c("lr", "alpha", "beta", "hint_rate", "batch_size")
importance <- read.csv("gain_tuning_results_importance.csv")

vals <- pareto_medians("gain_tuning_results.csv", search_space, digits = 3)
vals
importance

gain_epoch <- read_tuning("gain_epoch_tuning_results.csv")

library(ggplot2)
ggplot(gain_epoch) +
  geom_line(aes(x = epochs, y = avg_ed))
round(apply(gain_epoch[gain_epoch$is_pareto, "epochs", drop = FALSE], 2, median), 3)
