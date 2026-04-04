lapply(c("survival", "dplyr", "stringr", "survey", "mice", "arrow", "mitools", 'ggplot2'), require, character.only = T)
source("00_utils_functions.R")

i <- 2
digit <- stringr::str_pad(i, 4, pad = 0)
cat("Current:", digit, "\n")
load(paste0("./data/True/", digit, ".RData"))
cox.fit <- coxph(Surv(T_I, EVENT) ~
                   I((HbA1c - 50) / 5) + rs4506565 + I((AGE - 60) / 5) + I((eGFR - 75) / 10) +
                   I((Insulin - 15) / 2) + I((BMI - 28) / 2) + SEX + INSURANCE + RACE + 
                   SMOKE + I((AGE - 60) / 5):I((Insulin - 15) / 2), data = data)
temp_env <- new.env()

# load(paste0("./simulations/SampleOE/SRS/rf/", digit, ".RData"), env = temp_env)
# multi_impset <- temp_env[[ls(temp_env)[1]]]
# multi_impset <- mice::complete(multi_impset, "all")

multi_impset <- read_parquet(paste0("./simulations/SampleOE/SRS/sird/", digit, ".parquet"))
# multi_impset <- read_parquet(paste0("../SIRD/ablation/simulations/Multinomial/", digit, ".parquet"))

multi_impset <- multi_impset %>% group_split(imp_id)
multi_impset <- lapply(multi_impset, function(d) d %>% select(-imp_id))

multi_impset <- lapply(multi_impset, function(dat) {
  match_types(dat, data)
})
imp.mids <- imputationList(multi_impset)
cox.mod <- with(data = imp.mids,
                exp = coxph(Surv(T_I, EVENT) ~
                              I((HbA1c - 50) / 5) + rs4506565 + I((AGE - 60) / 5) + I((eGFR - 75) / 10) +
                              I((Insulin - 15) / 2) + I((BMI - 28) / 2) + SEX + INSURANCE + RACE + 
                              SMOKE + I((AGE - 60) / 5):I((Insulin - 15) / 2)))
pooled <- MIcombine(cox.mod)
round(exp(coef(cox.fit)) - exp(pooled$coefficients), 4)





k <- 4
diff_list <- list()

for (i in 1:k) {
  digit <- stringr::str_pad(i, 4, pad = "0")
  cat("Current:", digit, "\n")
  
  load(paste0("./data/True/", digit, ".RData"))
  cox.fit <- coxph(Surv(T_I, EVENT) ~
                     I((HbA1c - 50) / 5) + rs4506565 + I((AGE - 60) / 5) + I((eGFR - 75) / 10) +
                     I((Insulin - 15) / 2) + I((BMI - 28) / 2) + SEX + INSURANCE + RACE + 
                     SMOKE + I((AGE - 60) / 5):I((Insulin - 15) / 2), data = data)
  
  file_path <- paste0("./simulations/SampleOE/SRS/sird/", digit, ".parquet")
  
  if (!file.exists(file_path)) {
    cat("File not found, skipping:", file_path, "\n")
    next
  }
  
  multi_impset <- read_parquet(file_path)
  
  multi_impset <- multi_impset %>% group_split(imp_id)
  multi_impset <- lapply(multi_impset, function(d) d %>% select(-imp_id))
  
  multi_impset <- lapply(multi_impset, function(dat) {
    match_types(dat, data)
  })
  imp.mids <- imputationList(multi_impset)
  
  cox.mod <- with(data = imp.mids,
                  exp = coxph(Surv(T_I, EVENT) ~
                                I((HbA1c - 50) / 5) + rs4506565 + I((AGE - 60) / 5) + I((eGFR - 75) / 10) +
                                I((Insulin - 15) / 2) + I((BMI - 28) / 2) + SEX + INSURANCE + RACE + 
                                SMOKE + I((AGE - 60) / 5):I((Insulin - 15) / 2)))
  pooled <- MIcombine(cox.mod)
  
  diffs <- exp(coef(cox.fit)) - exp(pooled$coefficients)
  diff_list[[i]] <- diffs
}

diff_matrix <- do.call(rbind, diff_list)
rmse_diffs <- sqrt(apply(diff_matrix^2, 2, mean, na.rm = TRUE))
median_diffs <- apply(diff_matrix, 2, median, na.rm = TRUE)
round(rmse_diffs, 4)
abs(round(median_diffs, 4))

mean(round(rmse_diffs, 4) < abs(round(sqrt(apply(diffCoeff[diffCoeff$Method == "MULTINOMIAL", 2:17]^2, 2, mean)), 4)))
round(rmse_diffs, 4) - abs(round(sqrt(apply(diffCoeff[diffCoeff$Method == "MULTINOMIAL", 2:17]^2, 2, mean)), 4))

mean(abs(round(median_diffs, 4)) < abs(round(apply(diffCoeff[diffCoeff$Method == "MULTINOMIAL", 2:17], 2, median), 4)))
abs(round(median_diffs, 4)) - abs(round(apply(diffCoeff[diffCoeff$Method == "MULTINOMIAL", 2:17], 2, median), 4))

proportions(table(data$EVENT, multi_impset[[1]]$EVENT))
proportions(table(data$SMOKE, multi_impset[[1]]$SMOKE))

sum(diag(proportions(table(data$SMOKE, data$SMOKE_STAR))))
sum(diag(proportions(table(data$SMOKE, multi_impset[[1]]$SMOKE))))
table(data$SMOKE)
table(multi_impset[[1]]$SMOKE)

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
  geom_density(aes(x = log(HbA1c)), colour = "red") +
  geom_density(aes(x = log(HbA1c_STAR)), colour = "black") +
  geom_density(data = multi_impset[[1]], aes(x = log(HbA1c)), colour = "blue")

ggplot(data) +
  geom_density(aes(x = eGFR), colour = "red") +
  geom_density(aes(x = eGFR_STAR), colour = "black") +
  geom_density(data = multi_impset[[1]], aes(x = eGFR), colour = "blue")

ggplot(data) +
  geom_density(aes(x = BMI), colour = "red") +
  geom_density(aes(x = BMI_STAR), colour = "black") +
  geom_density(data = multi_impset[[1]], aes(x = BMI), colour = "blue")

ggplot(data) + 
  geom_point(aes(x = HbA1c, y = multi_impset[[1]]$HbA1c)) + 
  geom_abline()
ggplot(data) + 
  geom_point(aes(x = T_I, y = multi_impset[[1]]$T_I)) + 
  geom_abline() + ylim(0, 25)



lapply(c("survival", "dplyr", "stringr", "survey", "mice", "arrow", "mitools", 'ggplot2'), require, character.only = T)
source("00_utils_functions.R")

i <- 1
digit <- stringr::str_pad(i, 4, pad = 0)
cat("Current:", digit, "\n")
load(paste0("./data/True/", digit, ".RData"))
cox.fit <- coxph(Surv(T_I, EVENT) ~
                   I((HbA1c - 50) / 5) + rs4506565 + I((AGE - 60) / 5) + I((eGFR - 75) / 10) +
                   I((Insulin - 15) / 2) + I((BMI - 28) / 2) + SEX + INSURANCE + RACE + 
                   SMOKE + I((AGE - 60) / 5):I((Insulin - 15) / 2), data = data)
fit.STAR <- coxph(Surv(T_I, EVENT) ~
                    I((HbA1c_STAR - 50) / 5) + rs4506565_STAR + I((AGE - 60) / 5) + I((eGFR_STAR - 75) / 10) +
                    I((Insulin_STAR - 15) / 2) + I((BMI_STAR - 28) / 2) + SEX + INSURANCE + RACE +
                    SMOKE_STAR + I((AGE - 60) / 5):I((Insulin_STAR - 15) / 2), data = data)
multi_impset <- read_parquet(paste0("~/00_nesi_projects/uoa03789_nobackup/simulations/Multinomial/", digit, ".parquet"))
multi_impset <- multi_impset %>% group_split(imp_id)
multi_impset <- lapply(multi_impset, function(d) d %>% select(-imp_id))
multi_impset <- lapply(multi_impset, function(dat){
  match_types(dat, data)
})
imp.mids <- imputationList(multi_impset)
cox.mod <- with(data = imp.mids,
                exp = coxph(Surv(T_I, EVENT) ~
                              I((HbA1c - 50) / 5) + rs4506565 + I((AGE - 60) / 5) + I((eGFR - 75) / 10) +
                              I((Insulin - 15) / 2) + I((BMI - 28) / 2) + SEX + INSURANCE + RACE + 
                              SMOKE + I((AGE - 60) / 5):I((Insulin - 15) / 2)))
pooled <- MIcombine(cox.mod)
round(exp(coef(cox.fit)) - exp(pooled$coefficients), 4) 
round(exp(coef(cox.fit)) - exp(coef(fit.STAR)), 4)


