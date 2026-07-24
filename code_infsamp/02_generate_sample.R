# Draw phase-two samples and release only the weight, not the strata.
# Joint (S1,S2,Y) design (2026-07-13): Y is a full stratifier — global Y-tertiles are crossed
# with the 9 suppressed (S1,S2) cells to form 27 joint strata, and the phase-two sample is
# Neyman-allocated on Y across the joint strata. W = N_h/n_h is released for every row;
# S1, S2 and all stratum labels stay suppressed.
suppressMessages(lapply(c("data.table", "dplyr", "stringr"), require, character.only = TRUE))
source("00_utils_functions.R")

generateSamples <- function(data, n_phase2, seed, digit, n_ybin = 3L) {
  set.seed(seed)
  data <- as.data.frame(data); nN <- nrow(data)
  data$cell <- (data$S1 - 1L) * 3L + data$S2

  dir.create("./data/Sample/InfSamp", recursive = TRUE, showWarnings = FALSE)
  file_infsamp <- paste0("./data/Sample/InfSamp/", digit, ".csv")
  if (file.exists(file_infsamp)) return(invisible())

  qs   <- quantile(data$Y, seq(0, 1, length.out = n_ybin + 1))
  ybin <- cut(data$Y, qs, include.lowest = TRUE, labels = FALSE)
  jstr <- (data$cell - 1L) * n_ybin + ybin                        # cell x Y-bin joint stratum
  dd    <- data.frame(jstr = factor(jstr), Y = data$Y)
  alloc <- exactAllocation(dd, "jstr", "Y", n_phase2)             # Neyman on Y across joint strata
  id2 <- integer(0); Wvec <- rep(NA_real_, nN)
  for (h in names(alloc)) {
    idx <- which(jstr == as.integer(h)); nh <- alloc[[h]]
    sel <- if (length(idx) <= nh) idx else sample(idx, nh)
    id2 <- c(id2, sel); Wvec[idx] <- length(idx) / nh
  }

  samp <- data %>%
    mutate(R = as.integer(1:nN %in% id2),
           W = Wvec,
           pi = 1 / W,
           X = replace(X, R == 0, NA)) %>%
    select(-S1, -S2, -cell)
  write.csv(samp, file = file_infsamp, row.names = FALSE)
}

replicate <- 500L; n_phase2 <- 500L
if (!dir.exists("./data/params")) dir.create("./data/params", recursive = TRUE)
seed_file <- "./data/params/data_sampling_seed.RData"
if (file.exists(seed_file)) load(seed_file) else { seed <- sample(1:1e6, replicate); save(seed, file = seed_file) }

for (i in 1:replicate) {
  digit <- str_pad(i, 4, pad = "0")
  load(paste0("./data/True/", digit, ".RData"))
  generateSamples(data, n_phase2, seed[i], digit)
}
