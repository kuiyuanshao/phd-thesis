lapply(c("mice", "dplyr", "stringr"), require, character.only = T)
source("00_utils_functions.R")

if(!dir.exists('./simulations/SampleOE')){dir.create('./simulations/SampleOE')}
if(!dir.exists('./simulations/SampleOE/SRS')){dir.create('./simulations/SampleOE/SRS')}
if(!dir.exists('./simulations/SampleOE/Balance')){dir.create('./simulations/SampleOE/Balance')}
if(!dir.exists('./simulations/SampleOE/Neyman')){dir.create('./simulations/SampleOE/Neyman')}

if(!dir.exists('./simulations/SampleOE/SRS/mice')){dir.create('./simulations/SampleOE/SRS/mice')}
if(!dir.exists('./simulations/SampleOE/Balance/mice')){dir.create('./simulations/SampleOE/Balance/mice')}
if(!dir.exists('./simulations/SampleOE/Neyman/mice')){dir.create('./simulations/SampleOE/Neyman/mice')}

# args <- commandArgs(trailingOnly = TRUE)
# task_id <- as.integer(ifelse(length(args) >= 1,
#                              args[1],
#                              Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))
# sampling_design <- ifelse(length(args) >= 2, 
#                           args[2], Sys.getenv("SAMP", "All"))
# 
# replicate <- 500
# n_chunks <- 20
# chunk_size <- ceiling(replicate / n_chunks)
# first_rep <- (task_id - 1) * chunk_size + 1
# last_rep <- min(task_id * chunk_size, replicate)

cat("Task", task_id,
    "handling replicates", first_rep, "to", last_rep,
    "for sampling design:", sampling_design, "\n")

best_config_srs <- readRDS("./comparisons_tuning/mice/best_mice_config_srs.rds")
best_config_bal <- readRDS("./comparisons_tuning/mice/best_mice_config_bal.rds")
best_config_ney <- readRDS("./comparisons_tuning/mice/best_mice_config_ney.rds")
do_mice <- function(dat, nm, digit, best_config) {
  dat$H0_STAR <- nelsonaalen(dat, T_I_STAR, EVENT_STAR)
  dat$H0_TRUE <- nelsonaalen(dat, T_I, EVENT)
  mincor <- best_config$mincor
  repeat {
    cat(sprintf("      trying mincor = %.2f\n", mincor))
    tm <- system.time({
      mice_imp <- tryCatch(
        mice(dat, m = 20, print = T, maxit = 25,
             maxcor = 1.0001, ls.meth = "ridge", ridge = best_config$ridge,
             predictorMatrix = quickpred(dat, mincor = mincor)),
        error = identity
      )
    })
    
    if (!inherits(mice_imp, "error")) {
      cat(sprintf("[system.time] user=%.3fs sys=%.3fs elapsed=%.3fs\n",
                  tm[["user.self"]], tm[["sys.self"]], tm[["elapsed"]]))
      
      save(mice_imp, tm, file = file.path("simulations/SampleOE", nm, "mice",
                                      paste0(digit, ".RData")))
      break
    }
    message(sprintf("      mice() failed (%s)", mice_imp$message))
    
    mincor <- mincor + 0.05
  }
}

## ---------- main loop ---------------------------------------
for (i in 1:10) {
  digit <- str_pad(i, 4, pad = "0")
  cat("  replicate", digit, "\n")
  
  load(file.path("./data/Complete", paste0(digit, ".RData")))
  
  if (sampling_design %in% c("SRS", "All")) {
    if (!file.exists(file.path("simulations/SampleOE", "SRS", "mice",
                               paste0(digit, ".RData")))){
      samp <- read.csv(file.path("./data/SampleOE/SRS", paste0(digit, ".csv"))) %>%
        match_types(data) %>%
        mutate(across(all_of(data_info_srs$cat_vars), as.factor),
               across(all_of(data_info_srs$num_vars), as.numeric))
      do_mice(samp, "SRS", digit, best_config_srs)
    }
  }
  
  if (sampling_design %in% c("Balance", "All")) {
    if (!file.exists(file.path("simulations/SampleOE", "Balance", "mice",
                               paste0(digit, ".RData")))){
      samp <- read.csv(file.path("./data/SampleOE/Balance",
                                 paste0(digit, ".csv"))) %>%
        match_types(data) %>%
        mutate(across(all_of(data_info_balance$cat_vars), as.factor),
               across(all_of(data_info_balance$num_vars), as.numeric))
      do_mice(samp, "Balance", digit, best_config_bal)
    }
  }
  
  if (sampling_design %in% c("Neyman", "All")) {
    if (!file.exists(file.path("simulations/SampleOE", "Neyman", "mice",
                               paste0(digit, ".RData")))){
      samp <- read.csv(file.path("./data/SampleOE/Neyman",
                                 paste0(digit, ".csv"))) %>%
        match_types(data) %>%
        mutate(across(all_of(data_info_neyman$cat_vars), as.factor),
               across(all_of(data_info_neyman$num_vars), as.numeric))
      do_mice(samp, "Neyman", digit, best_config_ney)
    }
  }
}


