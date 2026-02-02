lapply(c("mice", "dplyr", "stringr"), require, character.only = T)
source("00_utils_functions.R")

if(!dir.exists('./simulations')){dir.create('./simulations')}
if(!dir.exists('./simulations/SRS')){dir.create('./simulations/SRS')}
if(!dir.exists('./simulations/Balance')){dir.create('./simulations/Balance')}
if(!dir.exists('./simulations/Neyman')){dir.create('./simulations/Neyman')}

if(!dir.exists('./simulations/SRS/mice')){dir.create('./simulations/SRS/mice')}
if(!dir.exists('./simulations/Balance/mice')){dir.create('./simulations/Balance/mice')}
if(!dir.exists('./simulations/Neyman/mice')){dir.create('./simulations/Neyman/mice')}

args <- commandArgs(trailingOnly = TRUE)
task_id <- as.integer(ifelse(length(args) >= 1,
                             args[1],
                             Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))
sampling_design <- ifelse(length(args) >= 2, 
                          args[2], Sys.getenv("SAMP", "All"))

replicate <- 500
n_chunks <- 20
chunk_size <- ceiling(replicate / n_chunks)
first_rep <- (task_id - 1) * chunk_size + 1
last_rep <- min(task_id * chunk_size, replicate)

cat("Task", task_id,
    "handling replicates", first_rep, "to", last_rep,
    "for sampling design:", sampling_design, "\n")

## ---------- helper: run & save one mice call ----------------
do_mice <- function(dat, nm, digit) {
  mincor <- 0.35
  repeat {
    cat(sprintf("      trying mincor = %.2f\n", mincor))
    tm <- system.time({
      mice_imp <- tryCatch(
        mice(dat, m = 20, print = T, maxit = 25,
             maxcor = 1.0001, ls.meth = "ridge", ridge = 0.1,
             predictorMatrix = quickpred(dat, mincor = mincor)),
        error = identity
      )
    })
    
    if (!inherits(mice_imp, "error")) {
      cat(sprintf("[system.time] user=%.3fs sys=%.3fs elapsed=%.3fs\n",
                  tm[["user.self"]], tm[["sys.self"]], tm[["elapsed"]]))
      
      save(mice_imp, tm, file = file.path("simulations", nm, "mice",
                                      paste0(digit, ".RData")))
      break
    }
    message(sprintf("      mice() failed (%s)", mice_imp$message))
    
    mincor <- mincor + 0.05
  }
}

## ---------- main loop ---------------------------------------
for (i in first_rep:last_rep) {
  digit <- str_pad(i, 4, pad = "0")
  cat("  replicate", digit, "\n")
  
  load(file.path("./data/Complete", paste0(digit, ".RData")))
  
  if (sampling_design %in% c("SRS", "All")) {
    if (!file.exists(file.path("simulations", "SRS", "mice",
                               paste0(digit, ".RData")))){
      samp <- read.csv(file.path("./data/Sample/SRS", paste0(digit, ".csv"))) %>%
        match_types(data) %>%
        mutate(across(all_of(data_info_srs$cat_vars), as.factor),
               across(all_of(data_info_srs$num_vars), as.numeric))
      do_mice(samp, "SRS", digit)
    }
  }
  
  if (sampling_design %in% c("Balance", "All")) {
    if (!file.exists(file.path("simulations", "Balance", "mice",
                               paste0(digit, ".RData")))){
      samp <- read.csv(file.path("./data/Sample/Balance",
                                 paste0(digit, ".csv"))) %>%
        match_types(data) %>%
        mutate(across(all_of(data_info_balance$cat_vars), as.factor),
               across(all_of(data_info_balance$num_vars), as.numeric))
      do_mice(samp, "Balance", digit)
    }
  }
  
  if (sampling_design %in% c("Neyman", "All")) {
    if (!file.exists(file.path("simulations", "Neyman", "mice",
                               paste0(digit, ".RData")))){
      samp <- read.csv(file.path("./data/Sample/Neyman",
                                 paste0(digit, ".csv"))) %>%
        match_types(data) %>%
        mutate(across(all_of(data_info_neyman$cat_vars), as.factor),
               across(all_of(data_info_neyman$num_vars), as.numeric))
      do_mice(samp, "Neyman", digit)
    }
  }
}


