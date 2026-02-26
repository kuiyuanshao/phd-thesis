expit <- function(x){
  exp(x) / (1 + exp(x))
}

calcCICover <- function(true, lower, upper){
  return (true >= lower) & (true <= upper)
}

exactAllocation <- function(data, stratum_variable, target_variable, sample_size){
  strata_units <- as.data.frame(table(data[[stratum_variable]]))
  colnames(strata_units) <- c(stratum_variable, "count")
  conversion_functions <- list(
    numeric = "as.numeric",
    integer = "as.integer",
    character = "as.character",
    logical = "as.logical",
    factor = "as.factor"
  )
  strata_units[, 1] <-  do.call(conversion_functions[[class(data[[stratum_variable]])[1]]], list(strata_units[, 1]))
  
  data <- merge(data, strata_units, by = stratum_variable)
  Y_bars <- aggregate(as.formula(paste0(target_variable, " ~ ", stratum_variable)), data = data, FUN = function(x) sum(x) / length(x))
  colnames(Y_bars)[2] <- "Y_bars"
  data <- merge(data, Y_bars, by = stratum_variable)
  Ss <- aggregate(as.formula(paste0("(", target_variable, " - Y_bars", ")^2", " ~ ", stratum_variable)), 
                  data = data, FUN = function(x) sum(x) / (length(x) - 1))
  
  NS <- strata_units$count * sqrt(Ss[, 2])
  names(NS) <- Ss[, 1]
  NS <- NS[order(NS, decreasing = T)]
  # Type-II
  columns <- sample_size - 2 * nrow(Ss)
  priority <- matrix(0, nrow = columns, ncol = nrow(Ss))
  colnames(priority) <- names(NS)
  for (h in names(NS)){
    priority[, h] <- NS[[h]] / sqrt((2:(columns + 1)) * (3:(columns + 2)))
  }
  priority <- as.data.frame(priority)
  priority <- stack(priority)
  colnames(priority) <- c("value", stratum_variable)
  order_priority <- order(priority$value, decreasing = T)
  alloc <- (table(priority[[stratum_variable]][order_priority[1:columns]]) + 2)
  alloc <- alloc[names(table(data[[stratum_variable]]))]
  return (alloc)
}

match_types <- function(new_df, orig_df) {
  common <- intersect(names(orig_df), names(new_df))
  out <- new_df
  
  for (nm in common) {
    tmpl <- orig_df[[nm]]
    col <- out[[nm]]
    if (is.integer(tmpl))        out[[nm]] <- as.integer(col)
    else if (is.numeric(tmpl))   out[[nm]] <- as.numeric(col)
    else if (is.logical(tmpl))   out[[nm]] <- as.logical(as.numeric(col))
    else if (is.factor(tmpl)) {
      out[[nm]] <- factor(col,
                          levels = levels(tmpl),
                          ordered = is.ordered(tmpl))
    }
    else if (inherits(tmpl, "Date")) {
      out[[nm]] <- as.Date(col)
    } else if (inherits(tmpl, "POSIXct")) {
      tz <- attr(tmpl, "tzone")
      out[[nm]] <- as.POSIXct(col, tz = tz)
    }
    else {
      out[[nm]] <- as.character(col)
    }
  }
  out
}

reallocate <- function(samp){
  parts <- do.call(rbind, strsplit(as.character(samp$STRATA), "\\."))
  flag <- parts[,1]
  lvl1 <- as.integer(parts[,2])
  lvl2 <- as.integer(parts[,3])
  
  cnt   <- table(samp$STRATA)
  single <- names(cnt)[cnt == 1]       # lone strata
  multi <- names(cnt)[cnt > 1]        # viable targets
  
  nearest <- function(lbl){
    i <- which(samp$STRATA == lbl)[1]
    f <- flag[i]; x <- lvl1[i]; y <- lvl2[i]
    idx <- match(multi, samp$STRATA)
    cand <- multi[flag[idx] == f]    # same TRUE/FALSE block
    d <- abs(lvl1[idx[flag[idx]==f]] - x) + abs(lvl2[idx[flag[idx]==f]] - y)
    cand[which.min(d)]
  }
  
  map <- setNames(multi, multi)
  for(lbl in single) map[lbl] <- nearest(lbl)
  
  samp$STRATA <- map[samp$STRATA]
  samp$fpc <- cnt[samp$STRATA]
  
  return (samp)
}


