source("../../../utils/tuning_analysis.R")

search_space <- c("lr", "channels", "nheads", "layers", "batch_size",
                  "num_steps", "diffusion_embedding_dim", "featureemb")
importance <- read.csv("tabcsdi_tuning_results_importance.csv")

vals <- pareto_medians("tabcsdi_tuning_results.csv", search_space, digits = 5)
vals
importance

tabcsdi_epoch <- read_tuning("tabcsdi_epoch_tuning_results.csv")

library(ggplot2)
ggplot(tabcsdi_epoch) +
  geom_line(aes(x = epochs, y = avg_ed))
round(apply(tabcsdi_epoch[tabcsdi_epoch$is_pareto, "epochs", drop = FALSE], 2, median), 3)
