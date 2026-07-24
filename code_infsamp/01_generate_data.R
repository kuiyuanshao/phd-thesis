suppressMessages(lapply(c("data.table", "stringr"), require, character.only = TRUE))
source("00_utils_functions.R")

N <- 10000L
a1 <- c(-0.45, 0, 0.60)
a2 <- c(-0.35, 0, 0.45)
s0 <- 0.30; s1 <- 0.58
gam <- c(0.18, 0.12, 0.14)
alp <- 1.0; del <- c(0.40, 0, 0.50); sigt <- 1.90
bX <- 0.48; bet <- c(0.30, 0.20, 0.30)
sy0 <- 0.30; sy1 <- 0.70
cellmean <- outer(a1, a2, `+`)
cmin <- min(cellmean)
cellsd <- s0 + s1 * (cellmean - cmin)
cellsdY <- sy0 + sy1 * (cellmean - cmin)

generateData <- function(seed) {
  set.seed(seed)
  S1 <- sample.int(3, N, TRUE)
  S2 <- sample.int(3, N, TRUE)
  Z1 <- rnorm(N); Z2 <- rnorm(N); Z3 <- rbinom(N, 1, 0.5)
  cm <- a1[S1] + a2[S2]
  csd <- cellsd[cbind(S1, S2)]
  csdY <- cellsdY[cbind(S1, S2)]
  X <- cm + gam[1] * Z1 + gam[2] * Z2 + gam[3] * Z3 + csd * rnorm(N)
  Xt <- alp * X + del[1] * Z1 + del[2] * Z2 + del[3] * Z3 + sigt * rnorm(N)
  Y <- bX * X + bet[1] * Z1 + bet[2] * Z2 + bet[3] * Z3 + csdY * rnorm(N)
  data.table(id = 1:N, S1, S2, Z1, Z2, Z3, X, Xt, Y)
}

replicate <- 500L
for (d in c("./data", "./data/True", "./data/params"))
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)

truth <- list(beta_X = bX, term = "X")
save(truth, file = "./data/params/truth.RData")

seed_file <- "./data/params/data_generation_seed.RData"
if (file.exists(seed_file)) load(seed_file) else { seed <- sample(1:1e6, replicate); save(seed, file = seed_file) }

for (i in 1:replicate) {
  digit <- str_pad(i, 4, pad = "0")
  cat("Generating population", digit, "\n")
  data <- generateData(seed[i])
  save(data, file = paste0("./data/True/", digit, ".RData"), compress = "xz")
}
