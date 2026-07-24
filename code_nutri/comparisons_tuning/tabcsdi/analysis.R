source("../../../utils/tuning_analysis.R")

search_space <- c("lr", "batch_size", "featureemb")
importance <- read.csv("tabcsdi_tuning_results_importance.csv")

vals <- pareto_medians("tabcsdi_tuning_results.csv", search_space, digits = 5)
vals
importance
