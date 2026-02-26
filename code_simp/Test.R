lapply(c("survival", "dplyr", "stringr", "survey", "mice", "arrow", "mitools", 'ggplot2'), require, character.only = T)
source("00_utils_functions.R")

i <- 1
digit <- stringr::str_pad(i, 4, pad = 0)
cat("Current:", digit, "\n")
load(paste0("./data/True/", digit, ".RData"))
cox.fit <- glm(Y ~ X + Z, data = data)
multi_impset <- read_parquet(paste0("./simulations/SRS/sird/", digit, ".parquet"))
multi_impset <- multi_impset %>% group_split(imp_id)
multi_impset <- lapply(multi_impset, function(d) d %>% select(-imp_id))
multi_impset <- lapply(multi_impset, function(dat){
  match_types(dat, data)
})
imp.mids <- imputationList(multi_impset)
cox.mod <- with(data = imp.mids,
                exp = glm(Y ~ X + Z))
pooled <- MIcombine(cox.mod)
round(coef(cox.fit) - pooled$coefficients, 4)

R <- multi_impset[[1]]$R == 0
mse <- 0
for (i in 1:5){
  mse <- mse + mean((multi_impset[[i]]$X[R] - data$X[R])^2)
}
mse



ggplot(data) +
  geom_density(aes(x = X), colour = "red") +
  geom_density(aes(x = X_star), colour = "black") +
  geom_density(data = multi_impset[[1]], aes(x = X), colour = "blue")
ggplot(data) + 
  geom_point(aes(x = X, y = multi_impset[[1]]$X)) + 
  geom_abline() +
  geom_point(aes(x = X, y = X_star), alpha = 0.2, colour = "red")
ggplot(data) + 
  geom_point(aes(x = multi_impset[[1]]$X, y = Y)) + 
  geom_smooth(aes(x = X, y = Y),
              method = "glm", method.args = list(family = "gaussian")) +
  geom_point(aes(x = X, y = Y), alpha = 0.2, colour = "red")

