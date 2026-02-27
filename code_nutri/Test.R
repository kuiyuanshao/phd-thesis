lapply(c("survival", "dplyr", "stringr", "survey", "mice", "arrow", "mitools", 'ggplot2'), require, character.only = T)
source("00_utils_functions.R")

i <- 1
digit <- stringr::str_pad(i, 4, pad = 0)
cat("Current:", digit, "\n")
load(paste0("./data/True/", digit, ".RData"))
cox.fit <- glm(sbp ~ ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr +
                 ln_na_true:c_age, data = data)
cox.star <- glm(sbp ~ ln_k_avg + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr +
                  ln_na_true:c_age, data = data)
multi_impset <- read_parquet(paste0("./simulations/SRS/tabcsdi/", digit, ".parquet"))
multi_impset <- multi_impset %>% group_split(imp_id)
multi_impset <- lapply(multi_impset, function(d) d %>% select(-imp_id))
multi_impset <- lapply(multi_impset, function(dat){
  match_types(dat, data)
})
imp.mids <- imputationList(multi_impset)
cox.mod <- with(data = imp.mids,
                exp = glm(sbp ~ ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr +
                            ln_na_true:c_age))
pooled <- MIcombine(cox.mod)
round(coef(cox.fit) - pooled$coefficients, 4)

R <- multi_impset[[1]]$R == 0
mse <- mean((multi_impset[[1]]$ln_na_true[R] - data$ln_na_true[R])^2)
mse

ggplot(data) +
  geom_density(aes(x = ln_na_true), colour = "red") +
  geom_density(aes(x = ln_na_avg), colour = "black") +
  geom_density(data = multi_impset[[1]], aes(x = ln_na_true), colour = "blue")
ggplot(data) + 
  geom_point(aes(x = multi_impset[[1]]$ln_na_true, y = sbp)) +
  geom_point(aes(x = ln_na_true, y = sbp), alpha = 0.2, colour = "red")



i <- 1
digit <- stringr::str_pad(i, 4, pad = 0)
cat("Current:", digit, "\n")
load(paste0("./data/True/", digit, ".RData"))
cox.fit <- glm(sbp ~ ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr +
                 ln_na_true:c_age, data = data)
multi_impset <- read_parquet(paste0("./simulations/SRS/gain/", digit, ".parquet"))
multi_impset <- match_types(multi_impset, data)
cox.mod <- glm(sbp ~ ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr +
                 ln_na_true:c_age, data = multi_impset)
round(coef(cox.fit) - coef(cox.mod), 4)
