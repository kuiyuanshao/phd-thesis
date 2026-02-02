pacman::p_load(mlr3mbo, mlr3, paradox, mlr3learners, survival, survey, torch, dplyr, data.table)

files <- list.files("../../../tpvmi_gans", full.names = TRUE, recursive = FALSE)
files <- files[!grepl("tests", files)]
lapply(files, source)
source("../../00_utils_functions.R")
load(paste0("../../data/True/0001.RData"))

samp_srs <- read.csv(paste0("../../data/Sample/SRS/0001.csv"))
samp_srs <- match_types(samp_srs, data) 

samp_bal <- read.csv(paste0("../../data/Sample/Balance/0001.csv"))
samp_bal <- match_types(samp_bal, data) 

samp_ney <- read.csv(paste0("../../data/Sample/Neyman/0001.csv"))
samp_ney <- match_types(samp_ney, data)

samp_srs <- samp_srs[samp_srs$R == 1,]
samp_bal <- samp_bal[samp_bal$R == 1,]
samp_ney <- samp_ney[samp_ney$R == 1,]

mod_srs <- coxph(Surv(T_I, EVENT) ~
                   I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
                   I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
                   rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                   SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE,
                 data = samp_srs)

bal_design <- svydesign(ids = ~1, strata = ~STRATA, weights = ~W, 
                        data = samp_bal)
mod_bal <- svycoxph(Surv(T_I, EVENT) ~
                        I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
                        I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
                        rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                        SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE,
                    bal_design)

ney_design <- svydesign(ids = ~1, strata = ~STRATA, weights = ~W, 
                        data = samp_bal)
mod_ney <- svycoxph(Surv(T_I, EVENT) ~
                      I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
                      I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
                      rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                      SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE,
                    ney_design)

samp_srs <- samp_srs %>% 
  mutate(across(all_of(data_info_srs$cat_vars), as.factor, .names = "{.col}"),
         across(all_of(data_info_srs$num_vars), as.numeric, .names = "{.col}"))
samp_bal <- samp_bal %>% 
  mutate(across(all_of(data_info_balance$cat_vars), as.factor, .names = "{.col}"),
         across(all_of(data_info_balance$num_vars), as.numeric, .names = "{.col}"))
samp_ney <- samp_ney %>% 
  mutate(across(all_of(data_info_neyman$cat_vars), as.factor, .names = "{.col}"),
         across(all_of(data_info_neyman$num_vars), as.numeric, .names = "{.col}"))

search_space = ps(
  lr_d       = p_dbl(lower = 1e-5, upper = 5e-4),
  g_d_ratio  = p_dbl(lower = 0.2, upper = 1),
  pac        = p_int(lower = 1, upper = 100),
  common_width = p_int(lower = 128, upper = 512),
  common_depth = p_int(lower = 2, upper = 4),
  weight_decay = p_dbl(lower = 1e-6, upper = 1e-3)
)

tune_srs <- tune_gan(samp_srs, data, data_info_srs, mod_srs, search_space,
                     best_config_path = "best_gan_config_srs.rds",
                     log_path = "gan_tuning_log_srs.csv",
                     n_evals = 60, 
                     m = 3, epochs = 2000, device = "cuda", folds = 4)
tune_bal <- tune_gan(samp_bal, data, data_info_balance, mod_bal, search_space,
                     best_config_path = "best_gan_config_bal.rds",
                     log_path = "gan_tuning_log_bal.csv",
                     n_evals = 60, 
                     m = 3, epochs = 2000, device = "cuda", folds = 4)
tune_ney <- tune_gan(samp_ney, data, data_info_neyman, mod_ney, search_space,
                     best_config_path = "best_gan_config_ney.rds",
                     log_path = "gan_tuning_log_ney.csv",
                     n_evals = 60, 
                     m = 3, epochs = 2000, device = "cuda", folds = 4)