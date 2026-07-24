source("../../../utils/tuning_analysis.R")

search_space <- c("lr", "alpha", "beta", "hint_rate", "batch_size")
importance <- read.csv("gain_tuning_results_importance.csv")

gain_res <- read_tuning("gain_tuning_results.csv")
pareto_front <- gain_res[gain_res$is_pareto, , drop = FALSE]
vals <- round(apply(pareto_front[1, search_space], 2, mean), 3)
vals
importance

gain_epoch <- read_tuning("gain_epoch_tuning_results.csv")
round(apply(gain_epoch[gain_epoch$is_pareto, "epochs", drop = FALSE], 2, median), 3)
