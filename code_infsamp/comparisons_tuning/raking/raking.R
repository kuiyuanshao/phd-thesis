calibrateFun <- function(samp) {
  samp$R <- as.logical(samp$R)
  samp$Wstratum <- factor(round(samp$W, 6))

  twophase_des <- twophase(id = list(~1, ~1), strata = list(NULL, ~Wstratum),
                           subset = ~R, data = samp)

  modimp <- svyglm(X ~ Xt + Z1 + Z2 + Z3 + Y, family = gaussian(), design = twophase_des)
  samp$X_impute <- as.vector(predict(modimp, newdata = samp, type = "response", se.fit = FALSE))

  phase1mod <- glm(Y ~ X_impute + Z1 + Z2 + Z3, data = samp, family = gaussian())
  inffun <- dfbeta(phase1mod)
  colnames(inffun) <- paste0("if", seq_len(ncol(inffun)))

  twophase_des_imp <- twophase(id = list(~1, ~1), strata = list(NULL, ~Wstratum),
                               subset = ~R, data = cbind(samp, inffun))
  cali_des <- survey::calibrate(twophase_des_imp, make.formula(colnames(inffun)),
                                phase = 2, calfun = "raking")

  svyglm(Y ~ X + Z1 + Z2 + Z3, design = cali_des, family = gaussian())
}
