generateData <- function(digit, seed){
  set.seed(seed)
  n = 1e4
  beta <- c(1, 1, 1)
  e_U <- c(sqrt(.5), sqrt(.25))
  Z <- rbinom(n, 1, .5)
  X <- (1-Z)*rnorm(n, 0, 1) + Z*rnorm(n, 0.5, 1)
  epsilon <- rnorm(n, 0, 1)
  Y <- beta[1] + beta[2]*X + beta[3]*Z + epsilon
  X_star <- X + rnorm(n, 0, e_U[1]*(Z==0) + e_U[2]*(Z==1))
  data <- data.frame(X_star=X_star, Y=Y, X=X, Z=Z)
  if(!dir.exists('./data/True')){system('mkdir ./data/True')}
  save(data, file=paste0('./data/True/', digit, '.RData'),compress = 'xz')
}


if(!dir.exists('./data')){dir.create("./data")}
if(!dir.exists('./data/True')){dir.create("./data/True")}
if(!dir.exists('./data/params')){dir.create("./data/params")}
replicate <- 500
if (file.exists("./data/params/data_generation_seed.RData")){
  load("./data/params/data_generation_seed.RData")
}else{
  seed <- sample(1:100000, 500)
  save(seed, file = "./data/params/data_generation_seed.RData")
}

for (i in 1:10){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  suppressMessages({generateData(digit, seed[i])})
}


