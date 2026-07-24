# Pareto-front helpers for comparison-method tuning (base R only).

add_pareto_flag <- function(df, acc = "acc_loss", dist = "dist_loss") {
  a <- df[[acc]]; d <- df[[dist]]
  df$is_pareto <- !vapply(seq_along(a), function(i)
    any(a <= a[i] & d <= d[i] & (a < a[i] | d < d[i])), logical(1))
  df
}

read_tuning <- function(results_csv) {
  res <- read.csv(results_csv)
  res$acc_loss  <- res$avg_mse + res$avg_ce
  res$dist_loss <- res$avg_ed
  res <- res[stats::complete.cases(res), , drop = FALSE]
  add_pareto_flag(res)
}

pareto_medians <- function(results_csv, search_space, digits = 3) {
  res <- read_tuning(results_csv)
  pf  <- res[res$is_pareto, , drop = FALSE]
  round(apply(pf[, search_space, drop = FALSE], 2, median), digits)
}
