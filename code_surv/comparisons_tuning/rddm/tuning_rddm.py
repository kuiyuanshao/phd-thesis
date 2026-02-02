from tpvmi_rddm.tuner import RDDMTuner
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
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

base_config = {
        "train": {
            "epochs": 5000,
            "batch_size": 128,
            "weight_decay": 1e-6,
            "eval_batch_size": 1024,
            "lr": 0.0002,
        },
        "diffusion": {
            "diffusion_embedding_dim": 128,
            "num_steps": 25,
            "sum_scale": 0.5,
        },
        "model": {
            "net": "DenseNet",
            "channels": 512,
            "layers": 3,
            "dropout": 0.25,
            "gamma": 1,
            "zeta": 1
        },
        "else": {
            "samp": "SRS",
            "task": "Res-N",
            "m": 3,
            "mi_approx": "bootstrap"
        }
    }

tuning_grid = {
        "channels": [128, 256, 512],
        "layers": [3, 5],
        "sum_scale": [0.01, 0.1],
        "dropout": [0.25, 0.5],
        "weight_decay": [1e-3, 1e-4]
    }

file_path_srs = "../../data/Sample/SRS/0001.csv"
file_path_bal = "../../data/Sample/Balance/0001.csv"
file_path_ney = "../../data/Sample/Neyman/0001.csv"

df_srs = pd.read_csv(file_path_srs)
df_bal = pd.read_csv(file_path_bal)
df_ney = pd.read_csv(file_path_ney)

categorical_cols = ["rs4506565", "SEX", "INSURANCE", "RACE", "SMOKE"]
for col in categorical_cols:
    df_srs[col] = df_srs[col].astype('category')
    df_bal[col] = df_bal[col].astype('category')
    df_ney[col] = df_ney[col].astype('category')

cox_formula = """
    I((HbA1c - 50) / 5) + 
    I(((HbA1c - 50) / 5) ** 2) + 
    I((HbA1c - 50) / 5) : I((AGE - 50) / 5) + 
    rs4506565 + 
    I((AGE - 50) / 5) + 
    I((eGFR - 90) / 10) + 
    SEX + INSURANCE + RACE + 
    I(BMI / 5) + 
    SMOKE
"""

mod_srs = CoxPHFitter()
mod_bal = CoxPHFitter()
mod_ney = CoxPHFitter()

mod_srs.formula = cox_formula
mod_srs.duration_col = "T_I"
mod_srs.event_col = "EVENT"
mod_srs.fit(df_srs.dropna(), formula=cox_formula, duration_col="T_I", event_col="EVENT")

mod_bal.formula = cox_formula
mod_bal.duration_col = "T_I"
mod_bal.event_col = "EVENT"
mod_bal.fit(df_bal.dropna(), formula=cox_formula, duration_col="T_I", event_col="EVENT")

mod_ney.formula = cox_formula
mod_ney.duration_col = "T_I"
mod_ney.event_col = "EVENT"
mod_ney.fit(df_ney.dropna(), formula=cox_formula, duration_col="T_I", event_col="EVENT")

tuner_srs = RDDMTuner(
    model=mod_srs,
    base_config=base_config,
    data_info=data_info_srs,
    param_grid=tuning_grid,
    file_path=file_path_srs,
    n_trials=30,
    n_folds=4
)
best_conf_srs = tuner_srs.tune(config_path="../../data/best_config_srs.yaml")

tuner_bal = RDDMTuner(
    model=mod_bal,
    base_config=base_config,
    data_info=data_info_balance,
    param_grid=tuning_grid,
    file_path=file_path_bal,
    n_trials=30,
    n_folds=4
)
best_conf_bal = tuner_bal.tune(config_path="../../data/best_config_bal.yaml")

tuner_ney = RDDMTuner(
    model=mod_ney,
    base_config=base_config,
    data_info=data_info_neyman,
    param_grid=tuning_grid,
    file_path=file_path_ney,
    n_trials=30,
    n_folds=4
)
best_conf_ney = tuner_ney.tune(config_path="../../data/best_config_ney.yaml")