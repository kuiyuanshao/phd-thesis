pacman::p_load(dplyr, tidyr)

search_space <- c("lr", "alpha", "beta", "hint_rate", "batch_size")
importance <- read.csv("gain_tuning_results_importance.csv")

gain_tuning_results <- read.csv("gain_tuning_results.csv")
gain_tuning_results_clean <- gain_tuning_results %>%
  mutate(acc_loss = avg_mse + avg_ce,
         dist_loss = avg_ed) %>%
  drop_na()

gain_tuning_results_final <- gain_tuning_results_clean %>%
  rowwise() %>%
  mutate(is_pareto = !any(
    gain_tuning_results_clean$acc_loss <= acc_loss &
      gain_tuning_results_clean$dist_loss <= dist_loss &
      (gain_tuning_results_clean$acc_loss < acc_loss | gain_tuning_results_clean$dist_loss < dist_loss)
  )) %>%
  ungroup()

pareto_front <- gain_tuning_results_final %>% 
  filter(is_pareto == TRUE)
vals <- round(apply(pareto_front[1, search_space], 2, mean), 3)

vals
importance

gain_epoch_tuning_results <- read.csv("gain_epoch_tuning_results.csv")
gain_epoch_tuning_results <- gain_epoch_tuning_results %>%
  mutate(acc_loss = avg_mse + avg_ce,
         dist_loss = avg_ed) %>%
  drop_na()

gain_epoch_tuning_results <- gain_epoch_tuning_results %>%
  rowwise() %>%
  mutate(is_pareto = !any(
    gain_epoch_tuning_results$acc_loss <= acc_loss &
      gain_epoch_tuning_results$dist_loss <= dist_loss &
      (gain_epoch_tuning_results$acc_loss < acc_loss | gain_epoch_tuning_results$dist_loss < dist_loss)
  )) %>%
  ungroup()

pareto_front <- gain_epoch_tuning_results %>% 
  filter(is_pareto == TRUE)

round(apply(pareto_front[, "epochs"], 2, median), 3)

