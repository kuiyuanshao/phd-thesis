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
cox.star <- coxph(Surv(T_I_STAR, EVENT_STAR) ~
                   I((HbA1c_STAR - 50) / 5) + I(I((HbA1c_STAR - 50) / 5)^2) +
                   I((HbA1c_STAR - 50) / 5):I((AGE - 50) / 5) +
                   rs4506565_STAR + I((AGE - 50) / 5) + I((eGFR_STAR - 90) / 10) +
                   SEX + INSURANCE + RACE + I(BMI_STAR / 5) + SMOKE_STAR,
                 data = data)
multi_impset <- read_parquet(paste0("./simulations/SRS/sicg/", digit, ".parquet"))
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
round(exp(coef(cox.fit)) - exp(pooled$coefficients), 4)

proportions(table(data$EVENT, multi_impset[[1]]$EVENT))


library(ggplot2)


library(survival)
library(survminer)

diag_median <- function(df) {
  fit <- survfit(Surv(T_I, EVENT) ~ SMOKE, data = df)
  return(surv_median(fit))
}

real_median <- diag_median(multi_impset[[1]])
fake_median <- diag_median(data)




ggplot(data) + 
  geom_density(aes(x = HbA1c)) +
  facet_wrap(~SMOKE)
ggplot(multi_impset[[1]]) + 
  geom_density(aes(x = HbA1c)) +
  facet_wrap(~SMOKE)

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

ggplot(data) + geom_point(aes(x = HbA1c, y = multi_impset[[1]]$HbA1c)) + geom_abline()
ggplot(data) + geom_point(aes(x = T_I, y = multi_impset[[1]]$T_I)) + geom_abline()
# log_param <- read.csv("./comparisons_tuning/mice/mice_tuning_log_srs.csv")
# log_param_fd <- read.csv("./comparisons_tuning/mice/mice_tuning_log_fd_srs.csv")
# pairs(log_param_fd[, 2:4])
# log_param$type <- "coef"
# log_param_fd$type <- "fd"
# 
# log_param <- rbind(log_param, log_param_fd)
# ggplot(log_param_fd) + 
#   geom_point(aes(x = ridge,
#                  y = bias, colour = type)) + 
#   ylim(0, 50)
# log_param$mincor

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

