#################################### Generate Sample ####################################
lapply(c("dplyr", "survival"), require, character.only = T)
source("00_utils_functions.R")
generateSample <- function(data, proportion, seed){
  set.seed(seed)
  nRow <- N <- nrow(data)
  n_phase2 <- n <- round(nRow * proportion) 
  p2vars <- c("rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499", 
              "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039", 
              "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806",
              "HbA1c", "Creatinine", "eGFR", "WEIGHT", "BMI", 
              "SMOKE", "INCOME", "ALC", "EXER", "EDU", "SBP", "Triglyceride", 
              "Glucose", "F_Glucose", "Insulin", "Na_INTAKE", "K_INTAKE", 
              "KCAL_INTAKE", "PROTEIN_INTAKE", "HYPERTENSION",
              "C", "EVENT", "T_I")
  # Simple Random Sampling
  srs_ind <- sample(nRow, n_phase2)
  samp_srs <- data %>%
    dplyr::mutate(R = ifelse(1:nRow %in% srs_ind, 1, 0),
                  W = 1,
                  across(all_of(p2vars), ~ replace(., R == 0, NA)))
  # Balanced Sampling
  time_cut <- as.numeric(cut(data$T_I_STAR, breaks = c(-Inf, 6, 12, 18, Inf), 
                             labels = 1:4))
  hba1c_cut <- as.numeric(cut(data$HbA1c_STAR, breaks = c(-Inf, 55, 70, Inf), 
                   labels = 1:3))
  strata <- interaction(data$EVENT_STAR, time_cut, hba1c_cut, drop = TRUE)
  data$STRATA <- strata
  k <- nlevels(strata)
  per_strat <- floor(n_phase2 / k)
  ids_by_str <- split(seq_len(nRow), strata)
  balanced_ind <- unlist(lapply(names(ids_by_str), function(i){
    if (table(strata)[i] < per_strat){
      return (ids_by_str[[i]]) # Sample everyone if insufficient 
    }else{
      return (sample(ids_by_str[[i]], per_strat))
    }
  }))
  openStrata <- names(table(strata)[table(strata) > per_strat])
  remaining_per_strat <- ceiling((n_phase2 - length(balanced_ind)) / length(openStrata))
  remaining_ind <- unlist(lapply(openStrata, function(i){
    sample(ids_by_str[[i]][!(ids_by_str[[i]] %in% balanced_ind)], remaining_per_strat)
  }))[1:(n_phase2 - length(balanced_ind))]
  balanced_ind <- c(balanced_ind, remaining_ind)
  balanced_weights <- table(strata) / table(strata[balanced_ind])
  samp_balance <- data %>%
    dplyr::mutate(R = ifelse(1:nRow %in% balanced_ind, 1, 0), 
                  W = case_when(!!!lapply(names(balanced_weights), function(value){
                    expr(STRATA == !!value ~ !!balanced_weights[[value]])
                  })),
                  across(all_of(p2vars), ~ replace(., R == 0, NA)))
  # Stratified Sampling with Neyman Allocation
  ### Getting Influence Function by auxiliary variables

  mod.aux <- coxph(Surv(T_I_STAR, EVENT_STAR) ~
                     I((HbA1c_STAR - 50) / 5) + I(I((HbA1c_STAR - 50) / 5)^2) +
                     I((HbA1c_STAR - 50) / 5):I((AGE - 50) / 5) +
                     rs4506565_STAR + I((AGE - 50) / 5) + I((eGFR_STAR - 90) / 10) +
                     SEX + INSURANCE + RACE + I(BMI_STAR / 5) + SMOKE_STAR,
                   data = data, x = T, y = T)
  inf <- residuals(mod.aux, type = "dfbeta")[, 1]
  data$inf <- inf
  neyman_alloc <- exactAllocation(data, stratum_variable = "STRATA", 
                                  target_variable = "inf", 
                                  sample_size = n_phase2)
  neyman_ind <- unlist(lapply(names(ids_by_str), function(i){
      sample(ids_by_str[[i]], neyman_alloc[i])
  }))
  neyman_weights <- table(strata) / neyman_alloc
  samp_neyman <- data %>%
    dplyr::mutate(R = ifelse(1:nRow %in% neyman_ind, 1, 0), 
                  W = case_when(!!!lapply(names(neyman_weights), function(value){
                    expr(STRATA == !!value ~ !!neyman_weights[[value]])
                  })),
                  across(all_of(p2vars), ~ replace(., R == 0, NA))) %>%
    select(-inf)
  
  return (list(samp_srs = samp_srs, data,
               samp_balance = samp_balance, data,
               samp_neyman = samp_neyman, data))
}

####### STARTING SIMULATION.  SAVING FILES ########
if(!dir.exists('./data/Sample')){dir.create('./data/Sample')}
if(!dir.exists('./data/Sample/SRS')){dir.create('./data/Sample/SRS')}
if(!dir.exists('./data/Sample/Balance')){dir.create('./data/Sample/Balance')}
if(!dir.exists('./data/Sample/Neyman')){dir.create('./data/Sample/Neyman')}
replicate <- 500
if (file.exists("./data/data_sampling_seed.RData")){
  load("./data/data_sampling_seed.RData")
}else{
  seed <- sample(1:100000, 500)
  save(seed, file = "./data/data_sampling_seed.RData")
}
for (i in 1:replicate){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  load(paste0("./data/True/", digit, ".RData"))
  samp_result <- generateSample(data, 0.05, seed[i])
  write.csv(samp_result$samp_srs, 
            file = paste0("./data/Sample/SRS/", digit, ".csv"))
  write.csv(samp_result$samp_balance, 
            file = paste0("./data/Sample/Balance/", digit, ".csv"))
  write.csv(samp_result$samp_neyman, 
            file = paste0("./data/Sample/Neyman/", digit, ".csv"))
}
