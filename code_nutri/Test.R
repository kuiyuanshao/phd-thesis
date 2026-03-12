lapply(c("survival", "dplyr", "stringr", "survey", "mice", "arrow", "mitools", 'ggplot2'), require, character.only = T)
source("00_utils_functions.R")

i <- 1
digit <- stringr::str_pad(i, 4, pad = 0)
cat("Current:", digit, "\n")
load(paste0("./data/True/", digit, ".RData"))
lm.fit <- glm(sbp ~ ln_na_true * c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, data = data)
temp_env <- new.env()

load(paste0("./simulations/SRS/rf/", digit, ".RData"), env = temp_env)
multi_impset <- temp_env[[ls(temp_env)[1]]]
multi_impset <- mice::complete(multi_impset, "all")

multi_impset <- lapply(multi_impset, function(dat) {
  match_types(dat, data)
})
imp.mids <- imputationList(multi_impset)


lm.mod <- with(data = imp.mids,
                exp = glm(sbp ~ ln_na_true * c_age + c_bmi + high_chol + 
                            usborn + female + bkg_o + bkg_pr))

pooled <- MIcombine(lm.mod)
round(coef(lm.fit) - pooled$coefficients, 4)


ggplot(data) +
  geom_density(aes(x = ln_na_true), colour = "black") +
  geom_density(data = multi_impset[[1]], aes(x = ln_na_true), colour = "blue")



