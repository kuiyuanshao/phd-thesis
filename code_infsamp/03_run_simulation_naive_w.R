suppressMessages(lapply(c("dplyr", "stringr"), require, character.only = TRUE))
source("00_utils_functions.R")
BASE_IMP_PREDICTORS <- IMP_PREDICTORS

method <- "naive_w"; m <- 20L; sim_root <- "./simulations"
args <- commandArgs(trailingOnly = TRUE); task_env <- Sys.getenv("SLURM_ARRAY_TASK_ID")
task_id <- if (length(args) >= 1) as.integer(args[1]) else if (nzchar(task_env)) as.integer(task_env) else 1L
replicate <- 500L; n_chunks <- 20L; chunk_size <- ceiling(replicate / n_chunks)
run_all <- (length(args) == 0 && !nzchar(task_env))
first_rep <- if (run_all) 1L else (task_id - 1) * chunk_size + 1
last_rep <- if (run_all) replicate else min(task_id * chunk_size, replicate)
seed_file <- "./data/params/simulation_seed.RData"
if (file.exists(seed_file)) load(seed_file) else { sim_seed <- sample(1:1e6, replicate); save(sim_seed, file = seed_file) }

for (design in DESIGNS) dir.create(file.path(sim_root, design, method), recursive = TRUE, showWarnings = FALSE)
for (i in first_rep:last_rep) {
  digit <- str_pad(i, 4, pad = "0")
  for (design in DESIGNS) {
    out_file <- file.path(sim_root, design, method, paste0(digit, ".RData"))
    if (file.exists(out_file)) next
    set.seed(sim_seed[i])
    samp <- read.csv(paste0("./data/Sample/", design, "/", digit, ".csv"))
    Wmat <- model.matrix(~ factor(W), samp)[, -1, drop = FALSE]
    colnames(Wmat) <- paste0("W_factor_", seq_len(ncol(Wmat)))
    samp <- cbind(samp, as.data.frame(Wmat))
    IMP_PREDICTORS <- c(BASE_IMP_PREDICTORS, colnames(Wmat))
    imp_list <- imp_gaussian(samp, m, weighted = FALSE)
    save(imp_list, file = file.path(sim_root, design, method, paste0(digit, ".RData")), compress = "xz")
  }
}
