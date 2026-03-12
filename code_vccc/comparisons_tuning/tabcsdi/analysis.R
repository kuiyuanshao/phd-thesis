pacman::p_load(dplyr, tidyr)

search_space <- c("lr", "channels", "nheads", "layers", "batch_size",
                  "num_steps", "diffusion_embedding_dim", "featureemb",
                  "token_emb_dim")
importance <- read.csv("tabcsdi_tuning_results_importance.csv")

tabcsdi_tuning_results <- read.csv("tabcsdi_tuning_results.csv")
tabcsdi_tuning_results_clean <- tabcsdi_tuning_results %>%
  mutate(acc_loss = avg_mse + avg_ce,
         dist_loss = avg_ed) %>%
  drop_na()

tabcsdi_tuning_results_final <- tabcsdi_tuning_results_clean %>%
  rowwise() %>%
  mutate(is_pareto = !any(
    tabcsdi_tuning_results_clean$acc_loss <= acc_loss &
      tabcsdi_tuning_results_clean$dist_loss <= dist_loss &
      (tabcsdi_tuning_results_clean$acc_loss < acc_loss | tabcsdi_tuning_results_clean$dist_loss < dist_loss)
  )) %>%
  ungroup()

pareto_front <- tabcsdi_tuning_results_final %>% 
  filter(is_pareto == TRUE)
vals <- round(apply(pareto_front[, search_space], 2, median), 3)

vals
importance
