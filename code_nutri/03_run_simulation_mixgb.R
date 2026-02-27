lapply(c("mixgb", "dplyr", "stringr"), require, character.only = T)
#lapply(paste0("./comparisons/mixgb/", list.files("./comparisons/mixgb/")), source)
source("00_utils_functions.R")

if(!dir.exists('./simulations/SampleOE')){dir.create('./simulations/SampleOE')}
if(!dir.exists('./simulations/SampleOE/SRS')){dir.create('./simulations/SampleOE/SRS')}
if(!dir.exists('./simulations/SampleOE/Balance')){dir.create('./simulations/SampleOE/Balance')}
if(!dir.exists('./simulations/SampleOE/Neyman')){dir.create('./simulations/SampleOE/Neyman')}

if(!dir.exists('./simulations/SampleOE/SRS/mixgb')){dir.create('./simulations/SampleOE/SRS/mixgb')}
if(!dir.exists('./simulations/SampleOE/Balance/mixgb')){dir.create('./simulations/SampleOE/Balance/mixgb')}
if(!dir.exists('./simulations/SampleOE/Neyman/mixgb')){dir.create('./simulations/SampleOE/Neyman/mixgb')}


args <- commandArgs(trailingOnly = TRUE)
task_id <- as.integer(ifelse(length(args) >= 1,
                             args[1],
                             Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))

replicate <- 500
n_chunks <- 20
chunk_size <- ceiling(replicate / n_chunks)
first_rep <- (task_id - 1) * chunk_size + 1
last_rep <- min(task_id * chunk_size, replicate)

load("./data/mixgb_srsParams.RData")
load("./data/mixgb_balanceParams.RData")
load("./data/mixgb_neymanParams.RData")
params_srs <- as.list(srs_tune$cv_table["864", 1:4])
params_balance <- as.list(balance_tune$best_params[-5])
params_neyman <- as.list(neyman_tune$best_params[-5])

do_mixgb <- function(samp, params, nm, digit) {
  cv_results <- mixgb_cv(samp, xgb.params = params, verbose = F, nrounds = 100)
  tm <- system.time({
    mixgb_imp <- mixgb(samp, m = 20, xgb.params = params,
                       initial.fac = "sample", nrounds = cv_results$best.nrounds)
  })
  mixgb_imp <- lapply(mixgb_imp, function(dat){
    match_types(as.data.frame(dat), data)
  })
  imp.mids <- as.mids(mixgb_imp)
  cox.fit <- with(data = imp.mids, 
                  exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) +
                                rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                                SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE))
  pooled <- mice::pool(cox.fit)
  sumry <- summary(pooled, conf.int = TRUE)
  cat("Bias: \n")
  cat(exp(sumry$estimate) - exp(coef(cox.true)), "\n")
  cat("Variance: \n")
  cat(apply(bind_rows(lapply(cox.fit$analyses, function(i){exp(coef(i))})), 2, var), "\n")
  
  save(mixgb_imp, tm, cv_results, file = file.path("simulations/SampleOE", nm, "mixgb",
                                                                paste0(digit, ".RData")))
}
for (i in 1:500){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  load(paste0("./data/Complete/", digit, ".RData"))
  samp_srs <- read.csv(paste0("./data/SampleOE/SRS/", digit, ".csv"))
  samp_balance <- read.csv(paste0("./data/SampleOE/Balance/", digit, ".csv"))
  samp_neyman <- read.csv(paste0("./data/SampleOE/Neyman/", digit, ".csv"))
  
  samp_srs <- match_types(samp_srs, data) %>% 
    mutate(across(all_of(data_info_srs$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info_srs$num_vars), as.numeric, .names = "{.col}"))
  samp_balance <- match_types(samp_balance, data) %>% 
    mutate(across(all_of(data_info_balance$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info_balance$num_vars), as.numeric, .names = "{.col}"))
  samp_neyman <- match_types(samp_neyman, data) %>% 
    mutate(across(all_of(data_info_neyman$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info_neyman$num_vars), as.numeric, .names = "{.col}"))
  
  if (!file.exists(paste0("./simulations/SampleOE/SRS/mixgb/", digit, ".RData"))){
    do_mixgb(samp_srs, params_srs, "simulations/SRS", digit)
  }
  if (!file.exists(paste0("./simulations/SampleOE/Balance/mixgb/", digit, ".RData"))){
    do_mixgb(samp_balance, params_balance, "Balance", digit)
  }
  if (!file.exists(paste0("./simulations/SampleOE/Neyman/mixgb/", digit, ".RData"))){
    do_mixgb(samp_neyman, params_neyman, "Neyman", digit)
  }
}



