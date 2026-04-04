import argparse
import pandas as pd
import sys
import os
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from tuner import BivariateTuner

data_info = {
    "weight_var": "W",
    "cat_vars": [
        "SEX", "RACE", "SMOKE", "EXER", "ALC", "INSURANCE", "REGION",
        "URBAN", "INCOME", "MARRIAGE", "HYPERTENSION", "EVENT",
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
        "BMI_STAR", "EDU_STAR", "Glucose_STAR", "F_Glucose_STAR", "Insulin_STAR",
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
        "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR"
    ]
}

CONFIG_PROFILES = {
    "multinomial": {
        "yaml_file": "./tuning_configs/base_config_multinomial.yaml",
        "output_csv": "./tuning_results/multinomial_tuning_results.csv",
        "tuning_grid": {
            "lr": ("log_float", 1e-4, 1e-2),
            "channels": ("int_exp2", 7, 10),
            "layers": ("int", 2, 7),
            "weight_decay": ("log_float", 1e-6, 1e-3),
            "sum_scale": ("float", 0.01, 0.25),
            "batch_size": ("int_exp2", 5, 8),
            "num_steps": ("int", 20, 100),
            "diffusion_embedding_dim": ("int_exp2", 6, 8),
        }
    },
    "multinomial_epochs": {
        "yaml_file": "./tuning_configs/base_config_multinomial_epochs.yaml",
        "output_csv": "./tuning_results/multinomial_epochs_tuning_results.csv",
        "tuning_grid": {
            "epochs": ("cat", [2500, 5000, 10000, 15000, 20000])
        }
    },
    "analogbits": {
        "yaml_file": "./tuning_configs/base_config_analogbits.yaml",
        "output_csv": "./tuning_results/analogbits_tuning_results.csv",
        "tuning_grid": {
            "epochs": ("cat", [2500, 5000, 10000, 15000, 20000])
        }
    },
    "multinomial_cfg": {
        "yaml_file": "./tuning_configs/base_config_multinomial_cfg.yaml",
        "output_csv": "./tuning_results/multinomial_cfg_tuning_results.csv",
        "tuning_grid": {
            "cond_drop_prob": ("float", 0.05, 0.2),
            "cfg_scale_num": ("float", 1, 2),
            "cfg_scale_cat": ("float", 1, 2),
        }
    },
    "multinomial_swag": {
        "yaml_file": "./tuning_configs/base_config_multinomial_swag.yaml",
        "output_csv": "./tuning_results/multinomial_swag_tuning_results.csv",
        "tuning_grid": {
            "lr": ("log_float", 1e-3, 1e-1),
        }
    },
    "multinomial_swag_epochs": {
        "yaml_file": "./tuning_configs/base_config_multinomial_swag_epochs.yaml",
        "output_csv": "./tuning_results/multinomial_swag_epochs_tuning_results.csv",
        "tuning_grid": {
            "epochs": ("cat", [2500, 5000, 10000, 15000, 20000])
        }
    }
}


def main():
    parser = argparse.ArgumentParser(description="Run tuner with selected configuration profile.")
    parser.add_argument(
        "--profile",
        type=str,
        default="multinomial_joint",
        choices=list(CONFIG_PROFILES.keys()),
        help="Choose the tuning profile to run."
    )
    args = parser.parse_args()

    selected = CONFIG_PROFILES[args.profile]
    yaml_file = selected["yaml_file"]
    tuning_grid = selected["tuning_grid"]
    output_csv = selected["output_csv"]

    num_params = len(tuning_grid)
    total_trials = max(10, 7 * num_params)
    print(f"[Task] Starting Tuning using profile: {args.profile}")
    with open(yaml_file, "r") as f:
        base_config = yaml.safe_load(f)

    file_path = "../../code_surv/data/SampleE/SRS/0001.csv"
    df = pd.read_csv(file_path).loc[:, lambda d: ~d.columns.str.contains('^Unnamed')]

    tuner = BivariateTuner(df, base_config, tuning_grid, data_info, n_splits=3)
    tuner.tune(n_trials=total_trials, output_csv=output_csv)

if __name__ == "__main__":
    main()
