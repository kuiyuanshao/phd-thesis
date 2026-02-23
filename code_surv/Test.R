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
multi_impset <- read_parquet(paste0("./simulations/SampleOE/SRS/sicg/", digit, ".parquet"))
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

proportions(table(data$EVENT, multi_impset[[1]]$EVENT))
proportions(table(data$SMOKE, multi_impset[[1]]$SMOKE))
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



# library(survival)
# library(ggplot2)
# 
# # 1. Fit the Model with the exact transformation
# # We use '*' to include main effects + interaction. 
# # If you ONLY want the interaction term, use ':' instead.
# fit <- coxph(Surv(T_I, EVENT) ~ I(Insulin - 14):I((AGE - 60) / 5), data = multi_impset[[1]])
# 
# # 2. Create a Grid of the SCALED variables
# # We define the range for the Scaled values (e.g., -4 to +4 sigmas)
# scaled_grid <- expand.grid(
#   scaled_age   = seq(-3, 3, length.out = 100),  
#   scaled_hba1c = seq(-3, 3, length.out = 100)
# )
# 
# # 3. Predict Risk (Manually)
# # Since 'predict' usually expects raw data, we calculate the Linear Predictor (LP) manually
# # using the model coefficients to ensure we stay strictly on the scaled metric.
# coefs <- coef(fit)
# 
# # Extract coefficients (names might vary, checking standard output)
# # usually: "I((HbA1c - 50)/5)", "I((AGE - 50)/5)", interaction
# b_hba1c <- coefs[grep("Insulin", names(coefs), fixed=TRUE)][1] # Main effect HbA1c
# b_age   <- coefs[grep("AGE", names(coefs), fixed=TRUE)][1]   # Main effect Age
# b_int   <- coefs[grep(":", names(coefs), fixed=TRUE)]        # Interaction
# 
# # Calculate Linear Predictor (Log Hazard)
# # Formula: Beta1*A + Beta2*H + Beta3*(A*H)
# scaled_grid$lp <- (b_age * scaled_grid$scaled_age) + 
#   (b_hba1c * scaled_grid$scaled_hba1c) + 
#   (b_int * scaled_grid$scaled_age * scaled_grid$scaled_hba1c)
# 
# # Convert to Hazard Ratio (Risk relative to the center point 50/50)
# scaled_grid$HR <- exp(scaled_grid$lp)
# 
# # 4. Plot the Contour Map
# ggplot(scaled_grid, aes(x = scaled_age, y = scaled_hba1c, z = HR)) +
#   # Create the heat map background
#   geom_tile(aes(fill = HR)) + 
#   # Add contour lines to show the "shape"
#   geom_contour(color = "white", alpha = 0.5) +
#   # Use a color scale that highlights high vs low risk (Blue -> Red)
#   scale_fill_distiller(palette = "Spectral", direction = -1, name = "Hazard Ratio") +
#   labs(
#     title = "Interaction Landscape (Scaled)",
#     subtitle = "Center (0,0) = Age 50, HbA1c 50. Color = Relative Risk.",
#     x = "Scaled Age  [(Age - 50) / 5]",
#     y = "Scaled HbA1c  [(HbA1c - 50) / 5]"
#   ) +
#   theme_minimal() +
#   coord_cartesian(expand = FALSE)

