lapply(c("survival", "dplyr", "stringr", "survey", "mice", "arrow", "mitools", 'ggplot2'), require, character.only = T)
source("00_utils_functions.R")

i <- 1
digit <- stringr::str_pad(i, 4, pad = 0)
data <- read.csv("./data/data.csv")
multi_impset <- read_parquet(paste0("./simulations/SRS/sird/", digit, ".parquet"))
multi_impset <- multi_impset %>% group_split(imp_id)
multi_impset <- lapply(multi_impset, function(d) d %>% select(-imp_id))
multi_impset <- lapply(multi_impset, function(dat){
  match_types(dat, data)
})
imp.mids <- imputationList(multi_impset)

fit_gauss_true <- lm(CD4_COUNT_1Y_sqrt_v ~ VL_COUNT_BSL_LOG_v + CD4_COUNT_BSL_sqrt_v + SEX +
                       AGE_AT_MED_START_v, 
                     data = data)

fit_gauss_imp <- with(data = imp.mids,
                      exp = lm(CD4_COUNT_1Y_sqrt_v ~ VL_COUNT_BSL_LOG_v + CD4_COUNT_BSL_sqrt_v + SEX +
                                 AGE_AT_MED_START_v))
pooled.lm <- MIcombine(fit_gauss_imp)
# Extract full coefficients
coef_gauss_true <- coef(fit_gauss_true)
coef_gauss_imp  <- pooled.lm$coefficient
# Calculate and output Relative Bias
rel_bias_gauss <- (coef_gauss_imp - coef_gauss_true) / coef_gauss_true
cat("\n--- Gaussian Model Relative Bias ---\n")
print(rel_bias_gauss)


#### 2. Binomial Model (Logistic Regression) ##############################
fit_bin_true <- glm(ANY_OI_v ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_v + YEAR_OF_ENROLLMENT, 
                    family = binomial, 
                    data = data)
bn.mod <- with(data = imp.mids,
                exp = glm(ANY_OI_v ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_v + YEAR_OF_ENROLLMENT, 
                          family = binomial))
pooled.bn <- MIcombine(bn.mod)
coef_bin_true <- coef(fit_bin_true)
coef_bin_imp  <- pooled.bn$coefficient

# Calculate and output Relative Bias
rel_bias_bin <- (coef_bin_imp - coef_bin_true) / coef_bin_true
cat("\n--- Binomial Model Relative Bias ---\n")
print(rel_bias_bin)


#### 3. Cox Model (Proportional Hazards) ##################################
fit_cox_true <- coxph(Surv(TIME_v, ANY_OI_v) ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_v + YEAR_OF_ENROLLMENT, 
                      data = data)
coef_cox_true <- coef(fit_cox_true)
cox.mod <- with(data = imp.mids,
                exp = coxph(Surv(TIME_v, ANY_OI_v) ~ CD4_COUNT_BSL_sqrt_v + SEX + AGE_AT_MED_START_v + YEAR_OF_ENROLLMENT))
pooled <- MIcombine(cox.mod)
coef_cox_imp <- pooled$coefficient
# Calculate and output Relative Bias
rel_bias_cox <- (exp(coef_cox_imp) - exp(coef_cox_true)) / exp(coef_cox_true)
cat("\n--- Cox Model Relative Bias ---\n")
print(rel_bias_cox)

jump_distance <- sqrt(200) - sqrt(100) # approx 4.14
# Calculate the scaled Hazard Ratios
hr_true <- exp(coef(fit_cox_true)["CD4_COUNT_BSL_sqrt_v"] * jump_distance)
hr_imp <- exp(pooled$coefficient["CD4_COUNT_BSL_sqrt_v"] * jump_distance)
cat("True HR (100-cell jump):", hr_true, "\n")
cat("Imputation HR (100-cell jump):", hr_imp, "\n")
