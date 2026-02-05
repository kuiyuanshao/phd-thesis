lapply(c("survival", "dplyr", "stringr", "survey", "mice", "arrow", "mitools"), require, character.only = T)
source("00_utils_functions.R")

i <- 1
digit <- stringr::str_pad(i, 4, pad = 0)
cat("Current:", digit, "\n")
load(paste0("./data/True/", digit, ".RData"))
cox.fit <- coxph(Surv(T_I, EVENT) ~
                   I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
                   I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
                   rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                   SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE,
                 data = data)
multi_impset <- read_parquet(paste0("./simulations/SRS/tpvmi_rddm/", digit, ".parquet"))
multi_impset <- multi_impset %>% group_split(imp_id)
multi_impset <- lapply(multi_impset, function(d) d %>% select(-imp_id))
multi_impset <- lapply(multi_impset, function(dat){
  match_types(dat, data)
})
imp.mids <- imputationList(multi_impset)
cox.mod <- with(data = imp.mids, 
                exp = coxph(Surv(T_I, EVENT) ~
                              I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
                              I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
                              rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                              SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE))
pooled <- MIcombine(cox.mod)
exp(coef(cox.fit)) - exp(pooled$coefficients)


log_param <- read.csv("./comparisons_tuning/mice/mice_tuning_log_srs.csv")
log_param_fd <- read.csv("./comparisons_tuning/mice/mice_tuning_log_fd_srs.csv")
pairs(log_param_fd[, 2:4])
log_param$type <- "coef"
log_param_fd$type <- "fd"

log_param <- rbind(log_param, log_param_fd)
ggplot(log_param_fd) + 
  geom_point(aes(x = ridge,
                 y = bias, colour = type)) + 
  ylim(0, 50)
log_param$mincor
# library(mice)
# library(mitools)
# i <- 1
# digit <- stringr::str_pad(i, 4, pad = 0)
# best_config <- readRDS("./comparisons_tuning/mice/best_mice_config_srs.rds")
# load(paste0("./data/True/", digit, ".RData"))
# cox.fit <- coxph(Surv(T_I, EVENT) ~
#                    I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
#                    I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
#                    rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
#                    SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE,
#                  data = data)
# samp <- read.csv(paste0("./data/Sample/SRS/", digit, ".csv"))
# samp <- match_types(samp, data)
# 
# pred_matrix <- quickpred(samp, mincor = 0.35, method = "pearson")
# 
# mice_imp <- mice(samp, m = 5, print = T, maxit = 50,
#                  maxcor = 1.0001, ls.meth = "ridge", ridge = 0.5,
#                  predictorMatrix = pred_matrix)
# multi_impset <- mice::complete(mice_imp, "all")
# multi_impset <- lapply(multi_impset, function(dat){
#   match_types(dat, data)
# })
# imp.mids <- imputationList(multi_impset)
# cox.mod <- with(data = imp.mids, 
#                 exp = coxph(Surv(T_I, EVENT) ~
#                               I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
#                               I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
#                               rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
#                               SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE))
# pooled <- MIcombine(cox.mod)
# exp(coef(cox.fit)) - exp(pooled$coefficients)
# 
library(ggplot2)

ggplot(data %>% filter(EVENT == 1)) +
  geom_density(aes(x = HbA1c), colour = "red") +
  geom_density(aes(x = HbA1c_STAR), colour = "black") +
  geom_density(data = multi_impset[[1]] %>% filter(EVENT == 1),
               aes(x = HbA1c), colour = "blue")

ggplot(data) +
  geom_density(aes(x = T_I), colour = "red") +
  geom_density(aes(x = T_I_STAR), colour = "black") +
  geom_density(data = multi_impset[[1]], aes(x = T_I), colour = "blue")

ggplot(data) +
  geom_density(aes(x = HbA1c), colour = "red") +
  geom_density(aes(x = HbA1c_STAR), colour = "black") +
  geom_density(data = multi_impset[[1]], aes(x = HbA1c), colour = "blue")

ggplot(data) +
  geom_density(aes(x = eGFR), colour = "red") +
  geom_density(aes(x = eGFR_STAR), colour = "black") +
  geom_density(data = multi_impset[[1]], aes(x = eGFR), colour = "blue")

ggplot(data) +
  geom_density(aes(x = BMI), colour = "red") +
  geom_density(aes(x = BMI_STAR), colour = "black") +
  geom_density(data = multi_impset[[1]], aes(x = BMI), colour = "blue")

# 
# i <- 2
# digit <- stringr::str_pad(i, 4, pad = 0)
# lapply(c("dplyr", "stringr", "torch", "coro", "survival", "mclust", "mitools"), require, character.only = T)
# files <- list.files("../tpvmi_gans", full.names = TRUE, recursive = FALSE)
# files <- files[!grepl("tests", files)]
# lapply(files, source)
# source("00_utils_functions.R")
# load(paste0("./data/True/", digit, ".RData"))
# cox.fit <- coxph(Surv(T_I, EVENT) ~
#                    I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
#                    I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
#                    rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
#                    SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE,
#                  data = data)
# samp <- read.csv(paste0("./data/Sample/SRS/", digit, ".csv"))
# samp <- match_types(samp, data) %>% 
#   mutate(across(all_of(data_info_srs$cat_vars), as.factor, .names = "{.col}"),
#          across(all_of(data_info_srs$num_vars), as.numeric, .names = "{.col}"))
# tpvmi_gans_imp <- tpvmi_gans(samp, m = 5, epochs = 2000,
#                              data_info = data_info_srs, 
#                              params = list(mi_approx = "dropout", lr_d = 2e-4,
#                                            g_dropout = 0.5, pac = 50, lambda = 10),
#                              device = "cuda")
# multi_impset <- lapply(tpvmi_gans_imp$imputation, function(dat){
#   match_types(dat, data)
# })
# imp.mids <- imputationList(multi_impset)
# cox.mod <- with(data = imp.mids, 
#                 exp = coxph(Surv(T_I, EVENT) ~
#                               I((HbA1c - 50) / 5) + I(I((HbA1c - 50) / 5)^2) +
#                               I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
#                               rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
#                               SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE))
# pooled <- MIcombine(cox.mod)
# cat("Bias: \n")
# cat(exp(pooled$coefficients) - exp(coef(cox.fit)), "\n")
# 
# 
# 
