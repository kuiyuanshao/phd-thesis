lapply(c("survival", "dplyr", "stringr", "survey", "mice", "arrow", "mitools", 'ggplot2'), require, character.only = T)
source("00_utils_functions.R")

i <- 3
digit <- stringr::str_pad(i, 4, pad = 0)
cat("Current:", digit, "\n")
load(paste0("./data/True/", digit, ".RData"))
cox.fit <- coxph(Surv(T_I, EVENT) ~
                   I((HbA1c - 50) / 5) +
                   rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                   SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE,
                 data = data)
multi_impset <- read_parquet(paste0("./simulations/SRS/sird/", digit, ".parquet"))
multi_impset <- multi_impset %>% group_split(imp_id)
multi_impset <- lapply(multi_impset, function(d) d %>% select(-imp_id))
multi_impset <- lapply(multi_impset, function(dat){
  match_types(dat, data)
})
imp.mids <- imputationList(multi_impset)
cox.mod <- with(data = imp.mids,
                exp = coxph(Surv(T_I, EVENT) ~
                              I((HbA1c - 50) / 5) +
                              rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                              SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE))
pooled <- MIcombine(cox.mod)
round(exp(coef(cox.fit)) - exp(pooled$coefficients), 4)

proportions(table(data$EVENT, multi_impset[[1]]$EVENT))



glm(data$HbA1c - multi_impset[[1]]$HbA1c ~ data$SEX)
summary(glm(data$HbA1c - multi_impset[[1]]$HbA1c ~ data$RACE))
summary(glm(data$eGFR - multi_impset[[1]]$eGFR ~ data$SEX))
glm(data$eGFR - multi_impset[[1]]$eGFR ~ data$INSURANCE)

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
#                    I((HbA1c - 50) / 5)  +
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
#                               I((HbA1c - 50) / 5)  +
#                               I((HbA1c - 50) / 5):I((AGE - 50) / 5) +
#                               rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
#                               SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE))
# pooled <- MIcombine(cox.mod)
# exp(coef(cox.fit)) - exp(pooled$coefficients)
# 

