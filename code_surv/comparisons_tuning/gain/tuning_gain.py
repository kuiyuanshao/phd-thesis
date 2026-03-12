import argparse
import pandas as pd
import sys
import os
import yaml
import time
import csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../benchmarks/GAIN")))
from tuner import BivariateTuner

data_info = {
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

CONFIG_PROFILES = {
    "gain": {
        "yaml_file": "base_config_gain.yaml",
        "output_csv": "gain_tuning_results.csv",
        "tuning_grid": {
            "lr": ("log_float", 1e-4, 1e-1),
            "alpha": ("int", 1, 500),
            "beta": ("int", 1, 100),
            "batch_size": ("int_exp2", 4, 8),
            "hint_rate": ("float", 0.5, 0.95)
        }
    },
    "gain_epoch": {
        "yaml_file": "base_config_gain_epoch.yaml",
        "output_csv": "gain_epoch_tuning_results.csv",
        "tuning_grid": {
            "epochs": ("int", 5000, 50000)
        }
    }
}


def main():
    parser = argparse.ArgumentParser(description="Run tuner with selected configuration profile.")
    parser.add_argument(
        "--profile",
        type=str,
        default="gain_epoch",
        choices=list(CONFIG_PROFILES.keys()),
        help="Choose the tuning profile to run."
    )
    args = parser.parse_args()

    selected = CONFIG_PROFILES[args.profile]
    yaml_file = selected["yaml_file"]
    tuning_grid = selected["tuning_grid"]
    output_csv = selected["output_csv"]

    num_params = len(tuning_grid)
    total_trials = 10 if num_params == 1 else max(30, 7 * num_params)
    print(f"[Task] Starting Tuning using profile: {args.profile}")
    with open(yaml_file, "r") as f:
        base_config = yaml.safe_load(f)

    file_path = "../../data/SampleOE/SRS/0001.csv"
    df = pd.read_csv(file_path).loc[:, lambda d: ~d.columns.str.contains('^Unnamed')]
    
    start_time = time.time()
    
    tuner = BivariateTuner(df, base_config, tuning_grid, data_info, n_splits=3)
    tuner.tune(n_trials=total_trials, output_csv=output_csv)
    
    end_time = time.time()
    iteration_time = end_time - start_time

    csv_file_path = f"{args.profile}_tuning_time.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time_Seconds"])
        writer.writerow([iteration_time])

if __name__ == "__main__":
    main()
