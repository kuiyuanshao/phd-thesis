suppressMessages(lapply(c("mice", "ranger", "dplyr", "stringr"), require, character.only = TRUE))
source("00_utils_functions.R")

cfg_file <- "./data/Config/best_config_rf.rds"
cfg <- if (file.exists(cfg_file)) {
  readRDS(cfg_file)
} else {
  list(ntree = 50L, mtry = 3L, nodesize = 5L, sampsize_ratio = 0.7, maxit = 5L, threads = 4L)
}
if (is.null(cfg$threads)) cfg$threads <- 4L; if (is.null(cfg$maxit)) cfg$maxit <- 5L

do_rf <- function(samp, m, cfg) {
  di <- DATA_INFOS[["infsamp"]]
  dat <- samp
  for (v in di$cat_vars) dat[[v]] <- as.factor(dat[[v]])
  for (v in di$num_vars) dat[[v]] <- as.numeric(dat[[v]])
  imp <- mice::mice(dat, m = m, method = "rf", maxit = cfg$maxit,
                    ntree = cfg$ntree, mtry = cfg$mtry, min.node.size = cfg$nodesize,
                    sample.fraction = cfg$sampsize_ratio, num.threads = cfg$threads,
                    printFlag = FALSE, remove.collinear = FALSE)
  lapply(mice::complete(imp, "all"), function(d) { s <- samp; s$X <- d$X; s })
}

method <- "rf"; m <- 20L; sim_root <- "./simulations"
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
    imp_list <- do_rf(samp, m, cfg)
    save(imp_list, file = file.path(sim_root, design, method, paste0(digit, ".RData")), compress = "xz")
  }
}
