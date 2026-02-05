import yaml
from sird.sird import SIRD
import os

if not os.path.exists("./simulations/SRS/sird"):
    os.makedirs("./simulations/SRS/sird")

if not os.path.exists("./simulations/Balance/sird"):
    os.makedirs("./simulations/Balance/sird")

if not os.path.exists("./simulations/Neyman/sird"):
    os.makedirs("./simulations/Neyman/sird")

data_info_srs = {
    "weight_var": "W",
    "cat_vars": [
        "SEX", "RACE", "SMOKE", "EXER", "ALC", "INSURANCE", "REGION",
        "URBAN", "INCOME", "MARRIAGE", "HYPERTENSION", "EVENT", "EVENT_STAR",
        "rs10811661", "rs17584499", "rs7754840", "rs7756992", "rs9465871",
        "rs11708067", "rs17036101", "rs358806", "rs4402960", "rs4607103",
        "rs1111875", "rs4506565", "rs5015480", "rs5219", "rs9300039",
        "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR", "rs17036101_STAR",
        "rs17584499_STAR", "rs1111875_STAR", "rs4402960_STAR", "rs4607103_STAR",
        "rs7754840_STAR", "rs9300039_STAR", "rs5015480_STAR", "rs9465871_STAR",
        "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
        "SMOKE_STAR", "INCOME_STAR", "ALC_STAR", "EXER_STAR", "HYPERTENSION_STAR",
        "R"
    ],
    "num_vars": [
        "X", "ID", "AGE", "EDU", "HEIGHT", "BMI", "MED_Count",
        "Creatinine", "Urea", "Potassium", "Sodium", "Chloride",
        "Bicarbonate", "Calcium", "Magnesium", "Phosphate", "Triglyceride",
        "HDL", "LDL", "Hb", "HCT", "RBC", "WBC", "Platelet", "MCV", "RDW",
        "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils",
        "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE", "AST",
        "ALT", "ALP", "GGT", "Bilirubin", "Albumin", "Globulin", "Protein",
        "Glucose", "F_Glucose", "HbA1c", "Insulin", "Ferritin", "SBP",
        "Temperature", "HR", "SpO2", "WEIGHT", "eGFR", "T_I", "C",
        "HbA1c_STAR", "Creatinine_STAR", "eGFR_STAR", "WEIGHT_STAR",
        "BMI_STAR", "EDU_STAR", "SBP_STAR",
        "Triglyceride_STAR", "C_STAR", "T_I_STAR",
        "Glucose_STAR", "F_Glucose_STAR", "Insulin_STAR",
        "Na_INTAKE_STAR", "K_INTAKE_STAR", "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR",
        "W"
    ],
    "phase2_vars": [
        "rs10811661", "rs7756992", "rs11708067",
        "rs17036101", "rs17584499", "rs1111875",
        "rs4402960", "rs4607103", "rs7754840",
        "rs9300039", "rs5015480", "rs9465871",
        "rs4506565", "rs5219", "rs358806",
        "HbA1c", "Creatinine", "eGFR", "WEIGHT", "BMI",
        "SMOKE", "INCOME", "ALC", "EXER", "EDU", "SBP", "Triglyceride",
        "Glucose", "F_Glucose", "Insulin", "Na_INTAKE", "K_INTAKE",
        "KCAL_INTAKE", "PROTEIN_INTAKE", "HYPERTENSION",
        "C", "EVENT", "T_I"
    ],
    "phase1_vars": [
        "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR",
        "rs17036101_STAR", "rs17584499_STAR", "rs1111875_STAR",
        "rs4402960_STAR", "rs4607103_STAR", "rs7754840_STAR",
        "rs9300039_STAR", "rs5015480_STAR", "rs9465871_STAR",
        "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
        "HbA1c_STAR", "Creatinine_STAR", "eGFR_STAR", "WEIGHT_STAR", "BMI_STAR",
        "SMOKE_STAR", "INCOME_STAR", "ALC_STAR", "EXER_STAR", "EDU_STAR", "SBP_STAR", "Triglyceride_STAR",
        "Glucose_STAR", "F_Glucose_STAR", "Insulin_STAR", "Na_INTAKE_STAR", "K_INTAKE_STAR",
        "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR", "HYPERTENSION_STAR",
        "C_STAR", "EVENT_STAR", "T_I_STAR"
    ]
}

# 2. Balanced Sampling Dictionary (Includes STRATA)
data_info_balance = {
    "weight_var": "W",
    "cat_vars": data_info_srs["cat_vars"] + ["STRATA"],
    "num_vars": data_info_srs["num_vars"],
    "phase2_vars": data_info_srs["phase2_vars"],
    "phase1_vars": data_info_srs["phase1_vars"]
}

# 3. Neyman Allocation Dictionary (Identical structure to Balance)
data_info_neyman = {
    "weight_var": "W",
    "cat_vars": data_info_srs["cat_vars"] + ["STRATA"],
    "num_vars": data_info_srs["num_vars"],
    "phase2_vars": data_info_srs["phase2_vars"],
    "phase1_vars": data_info_srs["phase1_vars"]
}

with open("./data/best_config_srs.yaml", "r") as f:
    config_srs = yaml.safe_load(f)
with open("./data/best_config_bal.yaml", "r") as f:
    config_bal = yaml.safe_load(f)
with open("./data/best_config_ney.yaml", "r") as f:
    config_ney = yaml.safe_load(f)

for i in range(1, 101):
    digit = str(i).zfill(4)
    file_path_srs = "F:/phd-thesis/code_surv/data/Sample/SRS/" + digit + ".csv"
    file_path_bal = "F:/phd-thesis/code_surv/data/Sample/Balance/" + digit + ".csv"
    file_path_ney = "F:/phd-thesis/code_surv/data/Sample/Neyman/" + digit + ".csv"

    save_path_srs = "F:/phd-thesis/code_surv/simulations/SRS/sird/" + digit + ".parquet"
    save_path_bal = "F:/phd-thesis/code_surv/simulations/Balance/sird/" + digit + ".parquet"
    save_path_ney = "F:/phd-thesis/code_surv/simulations/Neyman/sird/" + digit + ".parquet"

    sird_mod_srs = SIRD(config_srs, data_info_srs)
    sird_mod_srs.fit(file_path_srs)
    sird_mod_srs.impute(save_path=save_path_srs)

    # # --- 1. Simple Random Sampling ---
    # if not os.path.exists(save_path_srs):
    #     print(f"[SRS] Result not found at {save_path_srs}. Starting training and imputation...")
    #     sird_mod_srs = SIRD(config, data_info_srs)
    #     sird_mod_srs.fit(file_path_srs)
    #     sird_mod_srs.impute(save_path=save_path_srs)
    # else:
    #     print(f"[SRS] Result exists at {save_path_srs}. Skipping.")
    #
    # # --- 2. Balanced Sampling ---
    # if not os.path.exists(save_path_bal):
    #     print(f"[Balance] Result not found at {save_path_bal}. Starting training and imputation...")
    #     sird_mod_bal = SIRD(config, data_info_balance)
    #     sird_mod_bal.fit(file_path_bal)
    #     sird_mod_bal.impute(save_path=save_path_bal)
    # else:
    #     print(f"[Balance] Result exists at {save_path_bal}. Skipping.")
    #
    # # --- 3. Neyman Allocation ---
    # if not os.path.exists(save_path_ney):
    #     print(f"[Neyman] Result not found at {save_path_ney}. Starting training and imputation...")
    #     sird_mod_ney = SIRD(config, data_info_neyman)
    #     sird_mod_ney.fit(file_path_ney)
    #     sird_mod_ney.impute(save_path=save_path_ney)
    # else:
    #     print(f"[Neyman] Result exists at {save_path_ney}. Skipping.")


