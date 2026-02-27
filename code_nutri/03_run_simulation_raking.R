lapply(c("survey", "svyVGAM", "dplyr", "stringr"), require, character.only = T)
lapply(paste0("./comparisons/raking/", list.files("./comparisons/raking/")), source)
source("00_utils_functions.R")
options(survey.lonely.psu = "certainty")
if(!dir.exists('./simulations/SampleOE')){dir.create('./simulations/SampleOE')}
if(!dir.exists('./simulations/SampleOE/SRS')){dir.create('./simulations/SampleOE/SRS')}
if(!dir.exists('./simulations/SampleOE/Balance')){dir.create('./simulations/SampleOE/Balance')}
if(!dir.exists('./simulations/SampleOE/Neyman')){dir.create('./simulations/SampleOE/Neyman')}

if(!dir.exists('./simulations/SampleOE/SRS/raking')){dir.create('./simulations/SampleOE/SRS/raking')}
if(!dir.exists('./simulations/SampleOE/Balance/raking')){dir.create('./simulations/SampleOE/Balance/raking')}
if(!dir.exists('./simulations/SampleOE/Neyman/raking')){dir.create('./simulations/SampleOE/Neyman/raking')}

# replicate <- 500
# n_chunks <- 20
# chunk_size <- ceiling(replicate / n_chunks)
# first_rep <- (task_id - 1) * chunk_size + 1
# last_rep <- min(task_id * chunk_size, replicate)

for (i in 1:500){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  load(paste0("./data/Complete/", digit, ".RData"))
  samp_srs <- read.csv(paste0("./data/SampleOE/SRS/", digit, ".csv"))
  samp_balance <- read.csv(paste0("./data/SampleOE/Balance/", digit, ".csv"))
  samp_neyman <- read.csv(paste0("./data/SampleOE/Neyman/", digit, ".csv"))
  
  samp_srs$W <- 20
  samp_srs <- match_types(samp_srs, data)
  samp_balance <- match_types(samp_balance, data)
  samp_neyman <- match_types(samp_neyman, data)
  if (!file.exists(paste0("./simulations/SampleOE/SRS/raking/", digit, ".RData"))){
    rakingest <- calibrateFun(samp_srs, design = "simulations/SRS")
    save(rakingest, file = paste0("./simulations/SampleOE/SRS/raking/", digit, ".RData"))
  }
  if (!file.exists(paste0("./simulations/SampleOE/Balance/raking/", digit, ".RData"))){
    rakingest <- calibrateFun(samp_balance)
    save(rakingest, file = paste0("./simulations/SampleOE/Balance/raking/", digit, ".RData"))
  }
  if (!file.exists(paste0("./simulations/SampleOE/Neyman/raking/", digit, ".RData"))){
    rakingest <- calibrateFun(samp_neyman)
    save(rakingest, file = paste0("./simulations/SampleOE/Neyman/raking/", digit, ".RData"))
  }
}




