import yaml
import os
import argparse
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from sird import SIRD

import socket
from pathlib import Path

def setup_directories():
    is_nesi = "nesi.org.nz" in socket.getfqdn() or os.path.exists("/opt/nesi")
    # if is_nesi:
    #     base_path = Path("~/00_nesi_projects/uoa03789_nobackup/simulations").expanduser()
    # else:
    #     base_path = Path("./simulations")
    base_path = Path("./simulations")
    sub_dirs = ["Multinomial", "Multinomial_CFG", "AnalogBits", "SWAG", "Bootstrap", "Dropout"]
    for folder in sub_dirs:
        target = base_path / folder
        target.mkdir(parents=True, exist_ok=True)
    return base_path
sim_root = setup_directories()
print(f"Simulation root directory: {sim_root}")

data_info_srs = {
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
        "KCAL_INTAKE", "PROTEIN_INTAKE"
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
        "yaml_file": "./result_configs/config_multinomial.yaml",
        "output_dir": "Multinomial"
    },
    "multinomial_cfg": {
        "yaml_file": "./result_configs/config_multinomial_cfg.yaml",
        "output_dir": "Multinomial_CFG"
    },
    "analogbits": {
        "yaml_file": "./result_configs/config_analogbits.yaml",
        "output_dir": "AnalogBits"
    },
    "swag": {
        "yaml_file": "./result_configs/config_swag.yaml",
        "output_dir": "SWAG"
    },
    "dropout": {
        "yaml_file": "./result_configs/config_dropout.yaml",
        "output_dir": "Dropout"
    },
    "bootstrap": {
        "yaml_file": "./result_configs/config_bootstrap.yaml",
        "output_dir": "Bootstrap"
    }
}

def main():
    parser = argparse.ArgumentParser(description="Simulation with selected configuration profile.")
    parser.add_argument(
        "--profile",
        type=str,
        default="multinomial_conddrop",
        choices=list(CONFIG_PROFILES.keys()),
        help="Choose the tuning profile to run."
    )
    parser.add_argument(
        "--enable_slurm_array",
        action="store_true"
    )
    args = parser.parse_args()

    selected = CONFIG_PROFILES[args.profile]
    yaml_file = selected["yaml_file"]
    output_dir = selected["output_dir"]

    with open(yaml_file, "r") as f:
        base_config = yaml.safe_load(f)

    start_idx = 1
    end_idx = 501

    if args.enable_slurm_array:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 1))
        task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

        normalized_id = task_id - task_min
        chunk_size = (500 + task_count - 1) // task_count

        start_idx = 1 + normalized_id * chunk_size
        end_idx = min(start_idx + chunk_size, 501)

        print(f"SLURM Array Mode: Task ID {task_id}, processing indices {start_idx} to {end_idx - 1}")

    for i in range(start_idx, end_idx):
        digit = str(i).zfill(4)
        file_path_srs = f"../../code_surv/data/SampleE/SRS/{digit}.csv"
        save_path = sim_root / output_dir / f"{digit}.parquet"

        if save_path.exists():
            print(f"Output already exists, skipping: {save_path}")
            continue

        print(f"Processing: {file_path_srs}")
        sird_mod = SIRD(base_config, data_info_srs)
        sird_mod.fit(file_path_srs)
        sird_mod.impute(save_path=save_path)

if __name__ == "__main__":
    main()


