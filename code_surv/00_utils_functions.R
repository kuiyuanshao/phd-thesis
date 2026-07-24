# Shared core helpers live in utils/core_functions.R; locate by walking up from cwd.
.core_path <- getwd()
while (!file.exists(file.path(.core_path, "utils", "core_functions.R")) && dirname(.core_path) != .core_path) {
  .core_path <- dirname(.core_path)
}
source(file.path(.core_path, "utils", "core_functions.R"), local = TRUE)
rm(.core_path)

as.mids <- function(imp_list){
  imp_mids <- miceadds::datlist2mids(imp_list)
  return (imp_mids)
}

# Override: exact allocation that caps each stratum at its available size.
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
  strata_units[, 1] <- do.call(conversion_functions[[class(data[[stratum_variable]])[1]]], list(strata_units[, 1]))

  data <- merge(data, strata_units, by = stratum_variable)
  Y_bars <- aggregate(as.formula(paste0(target_variable, " ~ ", stratum_variable)), data = data, FUN = function(x) sum(x) / length(x))
  colnames(Y_bars)[2] <- "Y_bars"
  data <- merge(data, Y_bars, by = stratum_variable)
  Ss <- aggregate(as.formula(paste0("(", target_variable, " - Y_bars", ")^2", " ~ ", stratum_variable)),
                  data = data, FUN = function(x) sum(x) / (length(x) - 1))

  NS <- strata_units$count * sqrt(Ss[, 2])
  names(NS) <- Ss[, 1]
  NS <- NS[order(NS, decreasing = T)]

  upper <- strata_units$count
  names(upper) <- strata_units[[stratum_variable]]
  upper <- upper[names(NS)]

  columns <- sample_size - 2 * nrow(Ss)
  nc <- max(upper)

  priority <- matrix(0, nrow = nc, ncol = nrow(Ss))
  colnames(priority) <- names(NS)

  for (h in names(NS)){
    p_vals <- NS[[h]] / sqrt((2:(nc + 1)) * (3:(nc + 2)))
    limit <- upper[[h]]
    if (limit - 1 <= nc) {
      start_mask <- max(1, limit - 1)
      p_vals[start_mask:nc] <- -1
    }
    priority[, h] <- p_vals
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

    if (is.integer(tmpl)) {
      out[[nm]] <- as.integer(col)
    } else if (is.numeric(tmpl)) {
      out[[nm]] <- as.numeric(col)
    } else if (is.logical(tmpl)) {
      x <- col
      if (is.factor(x)) x <- as.character(x)

      if (is.character(x)) {
        res <- as.logical(x)

        # "0"/"1" strings fail as.logical; recover them via numeric coercion.
        failed_idx <- is.na(res) & !is.na(x)

        if (any(failed_idx)) {
          num_val <- suppressWarnings(as.numeric(x[failed_idx]))
          res[failed_idx] <- as.logical(num_val)
        }
        out[[nm]] <- res
      } else {
        out[[nm]] <- as.logical(col)
      }

    } else if (is.factor(tmpl)) {
      out[[nm]] <- factor(col,
                          levels = levels(tmpl),
                          ordered = is.ordered(tmpl))
    } else if (inherits(tmpl, "Date")) {
      out[[nm]] <- as.Date(col)
    } else if (inherits(tmpl, "POSIXct")) {
      tz <- attr(tmpl, "tzone")
      if (is.null(tz)) tz <- ""
      out[[nm]] <- as.POSIXct(col, tz = tz)
    } else {
      out[[nm]] <- as.character(col)
    }
  }
  out
}

compare_variances <- function(original_df, imputed_list, target_vars, categorical_vars) {
  results <- data.frame(
    Variable = character(),
    Original_Var = numeric(),
    Imputed_Avg_Var = numeric(),
    Ratio_Imp_vs_Org = numeric(),
    stringsAsFactors = FALSE
  )

  calc_safe_var <- function(vec, var_name) {
    vec <- vec[!is.na(vec)]
    if(length(vec) < 2) return(NA)

    if (var_name %in% categorical_vars) {
      vec <- as.numeric(as.factor(vec))
    } else {
      vec <- as.numeric(vec)
    }
    return(var(vec))
  }

  for (v in target_vars) {
    if (v %in% names(original_df)) {
      orig_var <- calc_safe_var(original_df[[v]], v)
    } else {
      orig_var <- NA
    }

    imp_vars <- sapply(imputed_list, function(df) {
      if (v %in% names(df)) {
        return(calc_safe_var(df[[v]], v))
      } else {
        return(NA)
      }
    })

    avg_imp_var <- mean(imp_vars, na.rm = TRUE)

    ratio <- ifelse(!is.na(orig_var) & orig_var != 0, avg_imp_var / orig_var, NA)

    results[nrow(results) + 1, ] <- list(
      Variable = v,
      Original_Var = round(orig_var, 4),
      Imputed_Avg_Var = round(avg_imp_var, 4),
      Ratio_Imp_vs_Org = round(ratio, 4)
    )
  }

  return(results)
}

data_info_srs_oe <- list(
  cat_vars = c(
    "SEX", "RACE", "SMOKE", "EXER", "ALC", "INSURANCE", "REGION",
    "URBAN", "INCOME", "MARRIAGE", "HYPERTENSION", "EVENT", "EVENT_STAR",
    "rs10811661", "rs17584499", "rs7754840", "rs7756992", "rs9465871",
    "rs11708067", "rs17036101", "rs358806", "rs4402960", "rs4607103",
    "rs1111875", "rs4506565", "rs5015480", "rs5219", "rs9300039",
    "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR", "rs17036101_STAR",
    "rs17584499_STAR", "rs1111875_STAR", "rs4402960_STAR", "rs4607103_STAR",
    "rs7754840_STAR", "rs9300039_STAR", "rs5015480_STAR", "rs9465871_STAR",
    "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
    "SMOKE_STAR", "INCOME_STAR", "ALC_STAR", "EXER_STAR",
    "R", "W"
  ),
  num_vars = c(
    "ID", "AGE", "EDU", "HEIGHT", "BMI", "MED_Count",
    "Creatinine", "Urea", "Potassium", "Sodium", "Chloride",
    "Bicarbonate", "Calcium", "Magnesium", "Phosphate", "Triglyceride",
    "HDL", "LDL", "Hb", "HCT", "RBC", "WBC", "Platelet", "MCV", "RDW",
    "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils",
    "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE", "AST",
    "ALT", "ALP", "GGT", "Bilirubin", "Albumin", "Globulin", "Protein",
    "Glucose", "F_Glucose", "HbA1c", "Insulin", "Ferritin", "SBP",
    "Temperature", "HR", "SpO2", "WEIGHT", "eGFR", "T_I", "C",
    "HbA1c_STAR", "Creatinine_STAR", "eGFR_STAR", "WEIGHT_STAR",
    "BMI_STAR", "EDU_STAR", "C_STAR", "T_I_STAR",
    "Glucose_STAR", "F_Glucose_STAR", "Insulin_STAR",
    "Na_INTAKE_STAR", "K_INTAKE_STAR", "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR"
  ),
  phase2_vars = c(
    "rs10811661", "rs7756992", "rs11708067",
    "rs17036101", "rs17584499", "rs1111875",
    "rs4402960", "rs4607103", "rs7754840",
    "rs9300039", "rs5015480", "rs9465871",
    "rs4506565", "rs5219", "rs358806",
    "HbA1c", "Creatinine", "eGFR", "WEIGHT", "BMI",
    "SMOKE", "INCOME", "ALC", "EXER", "EDU",
    "Glucose", "F_Glucose", "Insulin", "Na_INTAKE", "K_INTAKE",
    "KCAL_INTAKE", "PROTEIN_INTAKE",
    "C", "EVENT", "T_I"
  ),
  phase1_vars = c(
    "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR",
    "rs17036101_STAR", "rs17584499_STAR", "rs1111875_STAR",
    "rs4402960_STAR", "rs4607103_STAR", "rs7754840_STAR",
    "rs9300039_STAR", "rs5015480_STAR", "rs9465871_STAR",
    "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
    "HbA1c_STAR", "Creatinine_STAR", "eGFR_STAR", "WEIGHT_STAR", "BMI_STAR",
    "SMOKE_STAR", "INCOME_STAR", "ALC_STAR", "EXER_STAR", "EDU_STAR",
    "Glucose_STAR", "F_Glucose_STAR", "Insulin_STAR", "Na_INTAKE_STAR", "K_INTAKE_STAR",
    "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR",
    "C_STAR", "EVENT_STAR", "T_I_STAR"
  )
)

data_info_srs_e <- list(
  cat_vars = setdiff(data_info_srs_oe$cat_vars, "EVENT_STAR"),
  num_vars = setdiff(data_info_srs_oe$num_vars, c("C_STAR", "T_I_STAR")),
  phase2_vars = setdiff(data_info_srs_oe$phase2_vars, c("C", "EVENT", "T_I")),
  phase1_vars = setdiff(data_info_srs_oe$phase1_vars, c("C_STAR", "EVENT_STAR", "T_I_STAR"))
)

build_info_list <- function(base_srs) {
  list(
    srs = base_srs,
    balance = list(
      cat_vars = c(base_srs$cat_vars, "STRATA"),
      num_vars = base_srs$num_vars,
      phase2_vars = base_srs$phase2_vars,
      phase1_vars = base_srs$phase1_vars
    ),
    neyman = list(
      cat_vars = c(base_srs$cat_vars, "STRATA"),
      num_vars = base_srs$num_vars,
      phase2_vars = base_srs$phase2_vars,
      phase1_vars = base_srs$phase1_vars
    )
  )
}

DATA_INFOS <- list(
  SampleOE = build_info_list(data_info_srs_oe),
  SampleE = build_info_list(data_info_srs_e)
)
