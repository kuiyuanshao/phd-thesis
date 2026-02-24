import argparse
import pandas as pd
import sys
import os
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
from sird.tuner import BivariateTuner

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
        "SMOKE_STAR", "INCOME_STAR", "ALC_STAR", "EXER_STAR",
        "R"
    ],
    "num_vars": [
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
        "SMOKE", "INCOME", "ALC", "EXER", "EDU",
        "Glucose", "F_Glucose", "Insulin", "Na_INTAKE", "K_INTAKE",
        "KCAL_INTAKE", "PROTEIN_INTAKE",
        "C", "EVENT", "T_I"
    ],
    "phase1_vars": [
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
    ]
}

# --- 1. Define Profiles Dictionary ---
CONFIG_PROFILES = {
    "base": {
        "yaml_file": "./base_config_base.yaml",
        "output_csv": "base_tuning_results.csv",
        "tuning_grid": {
            "lr": ("cat", [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2]),
            "channels": ("cat", [256, 512, 1024, 2048]),
            "layers": ("int", 2, 7),
            "weight_decay": ("cat", [1e-6, 1e-5, 1e-4, 1e-3]),
            "sum_scale": ("cat", [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]),
            "dropout": ("cat", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]),
            "batch_size": ("cat", [32, 64, 128, 256]),
            "loss_num": ("int", 1, 10),
            "loss_cat": ("int", 1, 10),
            "num_steps": ("cat", [10, 20, 30, 40, 50]),
            "diffusion_embedding_dim": ("cat", [64, 128, 256]),
        },
        "reg_config": {
            'channels': {'min': 256.0, 'max': 2048.0, 'higher_is_more_reg': False},
            'layers': {'min': 2.0, 'max': 7.0, 'higher_is_more_reg': False},
            'lr': {'min': 1.0e-4, 'max': 1.0e-2, 'higher_is_more_reg': False},
            'sum_scale': {'min': 0.01, 'max': 0.25, 'higher_is_more_reg': True},
            'weight_decay': {'min': 1.0e-6, 'max': 1.0e-2, 'higher_is_more_reg': True},
            'dropout': {'min': 0.05, 'max': 0.5, 'higher_is_more_reg': True}
        }
    },
    "ce": {
        "yaml_file": "./base_config_ce.yaml",
        "output_csv": "ce_tuning_results.csv",
        "tuning_grid": {
            "lr": ("cat", [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2]),
            "channels": ("cat", [256, 512, 1024, 2048]),
            "layers": ("int", 2, 7),
            "weight_decay": ("cat", [1e-6, 1e-5, 1e-4, 1e-3]),
            "sum_scale": ("cat", [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]),
            "dropout": ("cat", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]),
            "batch_size": ("cat", [32, 64, 128, 256]),
            "loss_num": ("int", 1, 10),
            "loss_cat": ("int", 1, 10),
            "num_steps": ("cat", [10, 20, 30, 40, 50]),
            "diffusion_embedding_dim": ("cat", [64, 128, 256]),
        },
        "reg_config": {
            'channels': {'min': 256.0, 'max': 2048.0, 'higher_is_more_reg': False},
            'layers': {'min': 2.0, 'max': 7.0, 'higher_is_more_reg': False},
            'lr': {'min': 1.0e-4, 'max': 1.0e-2, 'higher_is_more_reg': False},
            'sum_scale': {'min': 0.01, 'max': 0.25, 'higher_is_more_reg': True},
            'weight_decay': {'min': 1.0e-6, 'max': 1.0e-2, 'higher_is_more_reg': True},
            'dropout': {'min': 0.05, 'max': 0.5, 'higher_is_more_reg': True}
        }
    }
}


def main():
    # --- 2. Setup Argparse ---
    parser = argparse.ArgumentParser(description="Run tuner with selected configuration profile.")
    parser.add_argument(
        "--profile",
        type=str,
        default="base",
        choices=list(CONFIG_PROFILES.keys()),
        help="Choose the tuning profile to run."
    )
    args = parser.parse_args()

    # --- 3. Extract Profile Settings ---
    selected = CONFIG_PROFILES[args.profile]
    yaml_file = selected["yaml_file"]
    tuning_grid = selected["tuning_grid"]
    reg_config = selected["reg_config"]
    output_csv = selected["output_csv"]

    print(f"[Task] Starting Tuning for SRS using profile: {args.profile}")

    # Load the YAML specific to the chosen profile
    with open(yaml_file, "r") as f:
        base_config = yaml.safe_load(f)

    file_path = "../../../data/SampleOE/SRS/0001.csv"
    df = pd.read_csv(file_path).loc[:, lambda d: ~d.columns.str.contains('^Unnamed')]

    # Pass dynamically selected parameters
    tuner = BivariateTuner(df, base_config, tuning_grid, data_info_srs, reg_config, n_splits=1)
    tuner.tune(n_trials=300, output_csv=output_csv)


if __name__ == "__main__":
    main()