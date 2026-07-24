source("../../utils/ablation_analysis.R")

# Method names are UPPER CASE to match on-disk simulation folders; same `group` = parallel competitors.
sird_ablation <- list(
  list(phase = 1, group = "Diffusion backbone (parallel)", parallel = TRUE,  method = "MULTINOMIAL",     outcome = "kept",    note = "winner"),
  list(phase = 1, group = "Diffusion backbone (parallel)", parallel = TRUE,  method = "ANALOGBITS",      outcome = "dropped"),
  list(phase = 2, group = "+ CFG",                         parallel = FALSE, method = "MULTINOMIAL_CFG", outcome = "dropped", note = "classifier-free guidance"),
  list(phase = 3, group = "MI approximations (parallel)",  parallel = TRUE,  method = "BOOTSTRAP",       outcome = "kept"),
  list(phase = 3, group = "MI approximations (parallel)",  parallel = TRUE,  method = "DROPOUT",         outcome = "kept",    note = "MC-Dropout"),
  list(phase = 3, group = "MI approximations (parallel)",  parallel = TRUE,  method = "SWAG",            outcome = "kept")
)

cat("\n=== SIRD ablation chain ===\n")
print(ablation_table(sird_ablation), row.names = FALSE)

sird_tuning <- list(
  multinomial = list(
    prefix = "./tuning_results/multinomial",
    cat  = c("batch_size", "layers", "channels", "diffusion_embedding_dim"),
    cont = c("lr", "loss_num", "loss_cat", "num_steps", "sum_scale", "weight_decay")
  ),
  multinomial_cfg = list(
    prefix = "./tuning_results/multinomial_cfg",
    cat  = NULL,
    cont = c("cfg_scale_num", "cfg_scale_cat", "cond_drop_prob")
  ),
  multinomial_swag = list(
    prefix = "./tuning_results/multinomial_swag",
    cat  = NULL, cont = c("lr")
  ),
  multinomial_swag_epochs = list(
    prefix = "./tuning_results/multinomial_swag_epochs",
    cat  = c("epochs"), cont = NULL
  )
)

tuning_results <- analyze_tuning_set(sird_tuning)

study <- run_ablation_study("SIRD", sird_ablation, track_bm = TRUE)

res <- summarise_study("SIRD", study$combined_file, study$active, ablation = sird_ablation)
print(as.data.frame(res$summary))
