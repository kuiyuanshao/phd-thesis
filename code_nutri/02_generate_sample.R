lapply(c("stringr", "dplyr", "data.table"), require, character.only = T)
source("./00_utils_functions.R")

generateSamples <- function(data, proportion, seed, digit){
  set.seed(seed)
  nrow <- dim(data)[1]
  n_phase2 <- nrow * proportion
  covars <- c("c_age", "c_bmi", "female", "usborn", "high_chol", 
              "bkg_pr", "bkg_o")
  aux_vars <- c("ln_na_avg", "ln_k_avg", "ln_kcal_avg", "ln_protein_avg")
  target_vars <- c("ln_na_true", "ln_k_true", "ln_kcal_true", "ln_protein_true")
  model_formula <- "sbp ~ c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr"
  
  mainDir <- "./data"
  dir.create(file.path(mainDir, "Sample"), showWarnings = FALSE)
  dir.create(file.path(mainDir, paste0("Sample", "/SRS")), showWarnings = FALSE)
  dir.create(file.path(mainDir, paste0("Sample", "/RS")), showWarnings = FALSE)
  dir.create(file.path(mainDir, paste0("Sample", "/WRS")), showWarnings = FALSE)
  dir.create(file.path(mainDir, paste0("Sample", "/SFS")), showWarnings = FALSE)
  dir.create(file.path(mainDir, paste0("Sample", "/ODS_TAIL")), showWarnings = FALSE)
  dir.create(file.path(mainDir, paste0("Sample", "/Neyman_INF")), showWarnings = FALSE)
  dir.create(file.path(mainDir, paste0("Sample", "/Neyman_ODS")), showWarnings = FALSE)
  
  # SRS
  id_phase2 <- sample(nrow, n_phase2)
  data_srs <- data %>%
    dplyr::mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
                  W = 1,
                  across(all_of(target_vars), ~ replace(., R == 0, NA)))
  write.csv(data_srs, file = paste0("./data/Sample/SRS/", digit, ".csv"))
  
  # ODS with extreme tails
  order_outcome <- order(data[["sbp"]])
  id_phase2 <- c(order_outcome[1:(n_phase2 %/% 2)], order_outcome[(nrow - n_phase2 %/% 2 + 1):nrow])
  data_ods <- data %>%
    dplyr::mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
                  W = 1,
                  across(all_of(target_vars), ~ replace(., R == 0, NA)))
  write.csv(data_ods, file = paste0("./data/Sample/ODS_TAIL/", digit, ".csv"))
  
  # RS
  modphase1 <- lm(as.formula(paste0(model_formula, " + ", paste0(aux_vars, collapse = " + "))), data = data)
  rs <- residuals(modphase1)
  order_residual <- order(rs)
  id_phase2 <- c(order_residual[1:(n_phase2 %/% 2)], order_residual[(nrow - n_phase2 %/% 2 + 1):nrow])
  data_rs <- data
  data_rs$rs <- rs
  data_rs <- data_rs %>%
    dplyr::mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
                  W = 1,
                  across(all_of(target_vars), ~ replace(., R == 0, NA)))
  write.csv(data_rs, file = paste0("./data/Sample/RS/", digit, ".csv"))
  
  # WRS
  pilot_prop <- 0.1
  n_pilot <- round(n_phase2 * pilot_prop)
  pilot_indices <- sample(1:nrow(data), n_pilot)
  pilot_data <- data[pilot_indices, ]
  
  remaining_indices <- setdiff(1:nrow(data), pilot_indices)
  
  var_formula <- as.formula(paste(target_vars[1], "~", paste(c(aux_vars, covars), collapse = "+")))
  pilot_model <- lm(var_formula, data = pilot_data)
  outcome_model <- lm(model_formula, data = data[-pilot_indices, ])
  sd_pilot <- sd(resid(pilot_model))
  
  resid_outcome <- resid(outcome_model)
  wrs <- order(resid_outcome * sd_pilot)
  n_rest <- n_phase2 - n_pilot
  idx <- c(wrs[1:(n_rest %/% 2)], wrs[(nrow - n_pilot - n_rest %/% 2 + 1):(nrow - n_pilot)])
  id_phase2 <- c(pilot_indices, remaining_indices[idx])
  
  data_wrs <- data
  data_wrs$wrs <- 0
  data_wrs$wrs[pilot_indices] <- resid(pilot_model)
  data_wrs$wrs[-pilot_indices] <- resid_outcome * sd_pilot
  data_wrs <- data_wrs %>%
    dplyr::mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
                  W = 1,
                  across(all_of(target_vars), ~ replace(., R == 0, NA)))
  write.csv(data_wrs, file = paste0("./data/Sample/WRS/", digit, ".csv"))
  
  # SFS
  modphase1 <- lm(as.formula(model_formula), data = data)
  score <- residuals(modphase1) * data[[aux_vars[1]]]
  order_score <- order(score)
  id_phase2 <- c(order_score[1:(n_phase2 %/% 2)], order_score[(nrow - n_phase2 %/% 2 + 1):nrow])
  data_sfs <- data
  data_sfs$score <- score
  data_sfs <- data_sfs %>%
    dplyr::mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
                  W = 1,
                  across(all_of(target_vars), ~ replace(., R == 0, NA)))
  write.csv(data_sfs, file = paste0("./data/Sample/SFS/", digit, ".csv"))
  
  # ODS with exact allocation
  quantile_split <- c(0.19, 0.81)
  outcome <- cut(data[["sbp"]], breaks = c(-Inf, quantile(data[["sbp"]], probs = quantile_split), Inf), 
                 labels = paste(1:(length(quantile_split) + 1), sep=','))
  data_ods_exactAlloc <- data
  data_ods_exactAlloc$outcome_strata <- as.numeric(outcome)
  alloc <- exactAllocation(data = data_ods_exactAlloc, stratum_variable = "outcome_strata", 
                           target_variable = target_vars[1], sample_size = n_phase2)
  weights <- table(data_ods_exactAlloc[["outcome_strata"]]) / alloc
  id_phase2 <- lapply(1:length(table(data_ods_exactAlloc[["outcome_strata"]])), 
                      function(j){sample((1:nrow)[data_ods_exactAlloc[["outcome_strata"]] == j], alloc[j])})
  id_phase2 <- unlist(id_phase2)
  data_ods_exactAlloc <- data_ods_exactAlloc %>%
    dplyr::mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
                  W = case_when(!!!lapply(names(weights), function(value){
                                expr(.data[["outcome_strata"]] == !!value ~ !!weights[[value]])
                                })),
           across(all_of(target_vars), ~ replace(., R == 0, NA)))
  write.csv(data_ods_exactAlloc, file = paste0("./data/Sample/Neyman_ODS/", digit, ".csv"))
  
  # Neyman Allocation By Influence Function.
  data_inf_exactAlloc <- data
  infl = dfbeta(lm(as.formula(paste0(model_formula, " + ", 
                                     paste0(c(aux_vars[1], paste0(aux_vars[1], ":c_age")), collapse = " + "))), 
                   data = data))[, 2]
  data_inf_exactAlloc$outcome_strata <- as.numeric(outcome)
  data_inf_exactAlloc$inf <- infl
  neyman_alloc <- exactAllocation(data_inf_exactAlloc, stratum_variable = "outcome_strata", 
                                  target_variable = "inf", 
                                  sample_size = n_phase2)
  neyman_weights <- table(data_inf_exactAlloc[["outcome_strata"]]) / neyman_alloc
  id_phase2 <- lapply(1:length(table(data_inf_exactAlloc[["outcome_strata"]])), 
                      function(j){sample((1:nrow)[data_inf_exactAlloc[["outcome_strata"]] == j], alloc[j])})
  id_phase2 <- unlist(id_phase2)
  data_inf_exactAlloc <- data_inf_exactAlloc %>%
    dplyr::mutate(R = ifelse(1:nrow %in% id_phase2, 1, 0),
                  W = case_when(!!!lapply(names(weights), function(value){
                    expr(.data[["outcome_strata"]] == !!value ~ !!weights[[value]])
                  })),
                  across(all_of(target_vars), ~ replace(., R == 0, NA)))
  write.csv(data_inf_exactAlloc, file = paste0("./data/Sample/Neyman_INF/", digit, ".csv"))
}

if (file.exists("./data/params/data_sampling_seed.RData")){
  load("./data/params/data_sampling_seed.RData")
}else{
  seed <- sample(1:100000, 500)
  save(seed, file = "./data/params/data_sampling_seed.RData")
}

n <- 500
pb <- txtProgressBar(min = 0, max = n, initial = 0) 
for (i in 1:n){
  set.seed(i)
  setTxtProgressBar(pb, i)
  digit <- str_pad(i, nchar(4444), pad=0)
  load(paste0("./data/True/", digit, ".RData"))
  generateSamples(as.data.frame(data), 0.05, seed[i], digit)
}
close(pb)

