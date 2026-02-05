lapply(c("dplyr", "stringr", "torch", "survival", "mitools"), require, character.only = T)
files <- list.files("../tpvmi_gans", full.names = TRUE, recursive = FALSE)
files <- files[!grepl("tests", files)]
lapply(files, source)
source("00_utils_functions.R")
if(!dir.exists('./simulations')){dir.create('./simulations')}
if(!dir.exists('./simulations/SRS')){dir.create('./simulations/SRS')}
if(!dir.exists('./simulations/Balance')){dir.create('./simulations/Balance')}
if(!dir.exists('./simulations/Neyman')){dir.create('./simulations/Neyman')}

if(!dir.exists('./simulations/SRS/tpvmi_gans')){dir.create('./simulations/SRS/tpvmi_gans')}
if(!dir.exists('./simulations/Balance/tpvmi_gans')){dir.create('./simulations/Balance/tpvmi_gans')}
if(!dir.exists('./simulations/Neyman/tpvmi_gans')){dir.create('./simulations/Neyman/tpvmi_gans')}

# args <- commandArgs(trailingOnly = TRUE)
# task_id <- as.integer(ifelse(length(args) >= 1,
#                              args[1],
#                              Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))
# sampling_design <- ifelse(length(args) >= 2, 
#                           args[2], Sys.getenv("SAMP", "All"))
# start_rep <- 1
# end_rep   <- 500
# n_chunks  <- 20
# task_id   <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))
# 
# n_in_window <- end_rep - start_rep + 1L
# chunk_size  <- ceiling(n_in_window / n_chunks)
# 
# first_rep <- start_rep + (task_id - 1L) * chunk_size
# last_rep  <- min(start_rep + task_id * chunk_size - 1L, end_rep)

do_tpvmi_gans <- function(samp, info, nm, digit) {
  tm <- system.time({
    tpvmi_gans_imp <- tpvmi_gans(samp, m = 5, epochs = 2000,
                                 data_info = info, params = list(mi_approx = "dropout"), 
                                 device = "cuda")
  })
  tpvmi_gans_imp$imputation <- lapply(tpvmi_gans_imp$imputation, function(dat){
    match_types(dat, data)
  })
  imp.mids <- imputationList(tpvmi_gans_imp$imputation)
  cox.mod <- with(data = imp.mids, 
                  exp = coxph(Surv(T_I, EVENT) ~
                                poly(I((HbA1c - 50) / 15), 2, raw = TRUE) + 
                                I((eGFR - 60) / 20) + 
                                I((BMI - 30) / 5) + rs4506565 + 
                                I((AGE - 60) / 15) + SEX +
                                INSURANCE + RACE + SMOKE +
                                I((HbA1c - 50) / 15):I((AGE - 60) / 15)))
  pooled <- MIcombine(cox.mod)
  cat("Bias: \n")
  cat(exp(pooled$coefficients) - exp(coef(cox.true)), "\n")
  #save(tpvmi_gans_imp, tm, file = file.path("simulations", nm, "tpvmi_gans",
  #                                        paste0(digit, ".RData")))
}

for (i in 1:500){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  load(paste0("./data/True/", digit, ".RData"))
  samp_srs <- read.csv(paste0("./data/Sample/SRS/", digit, ".csv"))
  samp_balance <- read.csv(paste0("./data/Sample/Balance/", digit, ".csv"))
  samp_neyman <- read.csv(paste0("./data/Sample/Neyman/", digit, ".csv"))
  
  samp_srs$W <- 20
  samp_srs <- match_types(samp_srs, data) %>% 
    mutate(across(all_of(data_info_srs$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info_srs$num_vars), as.numeric, .names = "{.col}"))
  samp_balance <- match_types(samp_balance, data) %>% 
    mutate(across(all_of(data_info_balance$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info_balance$num_vars), as.numeric, .names = "{.col}"))
  samp_neyman <- match_types(samp_neyman, data) %>% 
    mutate(across(all_of(data_info_neyman$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info_neyman$num_vars), as.numeric, .names = "{.col}"))
  
  cox.true <- coxph(Surv(T_I, EVENT) ~ poly(I((HbA1c - 50) / 15), 2, raw = TRUE) + I((eGFR - 60) / 20) + 
                      I((BMI - 30) / 5) + rs4506565 + I((AGE - 60) / 15) + SEX +
                      INSURANCE + RACE + SMOKE +
                      I((HbA1c - 50) / 15):I((AGE - 60) / 15), data = data)
  
  # if (!file.exists(paste0("./simulations/SRS/tpvmi_gans/", digit, ".RData"))){
  #   do_tpvmi_gans(samp_srs, data_info_srs, "SRS", digit)
  # }
  # if (!file.exists(paste0("./simulations/Balance/tpvmi_gans/", digit, ".RData"))){
  #   do_tpvmi_gans(samp_balance, data_info_balance, "Balance", digit)
  # }
  # if (!file.exists(paste0("./simulations/Neyman/tpvmi_gans/", digit, ".RData"))){
  #   do_tpvmi_gans(samp_neyman, data_info_neyman, "Neyman", digit)
  # }
  do_tpvmi_gans(samp_srs, data_info_srs, "SRS", digit)
}

