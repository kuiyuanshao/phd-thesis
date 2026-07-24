source("../../utils/ablation_analysis.R")

sicg_ablation <- list(
  list(phase = 1, group = "Generator + loss (parallel)", parallel = TRUE,  method = "UNPACK",                outcome = "kept",    note = "winner: Unpack WGAN"),
  list(phase = 1, group = "Generator + loss (parallel)", parallel = TRUE,  method = "PACK",                  outcome = "dropped", note = "PacWGAN"),
  list(phase = 1, group = "Generator + loss (parallel)", parallel = TRUE,  method = "UNPACK_INFO",           outcome = "dropped", note = "Unpack + information loss"),
  list(phase = 1, group = "Generator + loss (parallel)", parallel = TRUE,  method = "PACK_MML",              outcome = "dropped", note = "Pac + PCA moment-matching"),
  list(phase = 2, group = "+ CE loss",                   parallel = FALSE, method = "UNPACK_CE",             outcome = "dropped"),
  list(phase = 3, group = "+ semi-supervised",           parallel = FALSE, method = "UNPACK_SEMISUPERVISED", outcome = "kept",    note = "current best"),
  list(phase = 4, group = "+ residual modelling",           parallel = FALSE, method = "UNPACK_RESIDUALMODELING", outcome = "kept",    note = "500/500 reps; best RMSE & OracleBias, higher point bias"),
  list(phase = 5, group = "+ MI approximations (parallel)", parallel = TRUE,  method = "BOOTSTRAP", outcome = "kept",    note = "MI approx on UNPACK base (Adam); highest nominal coverage 0.83, lower efficiency"),
  list(phase = 5, group = "+ MI approximations (parallel)", parallel = TRUE,  method = "DROPOUT",   outcome = "kept",    note = "MC-Dropout on UNPACK base (Adam); best MI approx -- lowest bias/RMSE, coverage 0.82"),
  list(phase = 5, group = "+ MI approximations (parallel)", parallel = TRUE,  method = "SWAG",      outcome = "dropped", note = "SWAG on UNPACK base (SGD); weakest -- coverage only 0.62 (449/500 reps)")
)

cat("\n=== SICG ablation chain ===\n")
print(ablation_table(sicg_ablation), row.names = FALSE)

sicg_tuning <- list(
  unpack = list(
    prefix = "./tuning_results/unpack",
    cat  = c("hidden_dim", "layers", "batch_size", "scale_layer",
             "scale_hidden_dim", "scale_lr", "discriminator_steps"),
    cont = c("lr", "weight_decay", "dropout", "tau")
  ),
  pack = list(
    prefix = "./tuning_results/pack",
    cat  = c("hidden_dim", "layers", "batch_size", "scale_layer",
             "scale_hidden_dim", "scale_lr", "discriminator_steps", "pack"),
    cont = c("lr", "weight_decay", "dropout", "tau")
  ),
  unpack_epochs = list(prefix = "./tuning_results/unpack_epochs", cat = c("epochs"), cont = c()),
  pack_epochs   = list(prefix = "./tuning_results/pack_epochs",   cat = c("epochs"), cont = c()),
  swag          = list(prefix = "./tuning_results/swag",          cat = c(),         cont = c("lr"))
)

tuning_results <- analyze_tuning_set(sicg_tuning)

study <- run_ablation_study("SICG", sicg_ablation, track_bm = FALSE)

res <- summarise_study("SICG", study$combined_file, study$active, ablation = sicg_ablation)
print(as.data.frame(res$summary))
