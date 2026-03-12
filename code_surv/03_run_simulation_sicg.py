import os
import socket
import argparse
import yaml
import time
import csv
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from SICG.sicg import SICG

data_info_srs_oe = {
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

data_info_srs_e = {
    "weight_var": "W",
    "cat_vars": [v for v in data_info_srs_oe["cat_vars"] if v not in ["EVENT_STAR"]],
    "num_vars": [v for v in data_info_srs_oe["num_vars"] if v not in ["C_STAR", "T_I_STAR"]],
    "phase2_vars": [v for v in data_info_srs_oe["phase2_vars"] if v not in ["C", "EVENT", "T_I"]],
    "phase1_vars": [v for v in data_info_srs_oe["phase1_vars"] if v not in ["C_STAR", "EVENT_STAR", "T_I_STAR"]]
}


def build_info_dict(base_srs):
    return {
        "srs": base_srs,
        "balance": {
            "weight_var": "W",
            "cat_vars": base_srs["cat_vars"] + ["STRATA"],
            "num_vars": base_srs["num_vars"],
            "phase2_vars": base_srs["phase2_vars"],
            "phase1_vars": base_srs["phase1_vars"]
        },
        "neyman": {
            "weight_var": "W",
            "cat_vars": base_srs["cat_vars"] + ["STRATA"],
            "num_vars": base_srs["num_vars"],
            "phase2_vars": base_srs["phase2_vars"],
            "phase1_vars": base_srs["phase1_vars"]
        }
    }


DATA_INFOS = {
    "SampleOE": build_info_dict(data_info_srs_oe),
    "SampleE": build_info_dict(data_info_srs_e)
}


def generate_config_profiles(model_name):
    targets = {
        "srs": "SRS",
        "balance": "Balance",
        "neyman": "Neyman"
    }

    profiles = {}
    yaml_path = f"./data/Config/best_config_{model_name}.yaml"

    for key, folder in targets.items():
        profiles[key] = {
            "yaml_file": yaml_path,
            "data_folder": folder,
            "info_key": key
        }

    return profiles

CONFIG_PROFILES = generate_config_profiles("sicg")

print(f"Loaded {len(CONFIG_PROFILES)} profiles for the sicg model.")


def setup_env(type, data_folder):
    is_nesi = "nesi.org.nz" in socket.getfqdn() or os.path.exists("/opt/nesi")
    sim_root = Path("./simulations")
    data_root = Path("./data")

    save_dir = sim_root / type / data_folder / "sicg"
    save_dir.mkdir(parents=True, exist_ok=True)
    return data_root, save_dir, sim_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="SampleE",
        choices=["SampleOE", "SampleE"]
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="srs",
        choices=list(CONFIG_PROFILES.keys())
    )
    parser.add_argument(
        "--enable_slurm_array",
        action="store_true"
    )
    args = parser.parse_args()

    selected = CONFIG_PROFILES[args.profile]
    yaml_file = selected["yaml_file"]
    data_folder = selected["data_folder"]
    data_info = DATA_INFOS[args.type][selected["info_key"]]

    with open(yaml_file, "r") as f:
        base_config = yaml.safe_load(f)

    data_root, save_dir, sim_root = setup_env(args.type, data_folder)
    print(f"[{args.type} - {args.profile.upper()}] Environment ready.")

    timing_data = []

    start_idx = 1
    end_idx = 501
    task_id = 1

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
        start_time = time.time()
        digit = str(i).zfill(4)

        file_path = data_root / args.type / data_folder / f"{digit}.csv"
        save_path = save_dir / f"{digit}.parquet"

        if not file_path.exists():
            print(f"Warning: File missing: {file_path}, skipping...")
            continue

        if not save_path.exists():
            print(f"[{data_folder}] Result not found. Processing: {file_path} -> {save_path}")
            sicg_mod = SICG(base_config, data_info)
            sicg_mod.fit(str(file_path))
            sicg_mod.impute(save_path=str(save_path))
        else:
            print(f"[{data_folder}] Result exists at {save_path}. Skipping.")

        end_time = time.time()
        iteration_time = end_time - start_time
        timing_data.append([i, iteration_time])

    if timing_data:
        csv_file_path = sim_root / f"{args.type}_sicg_{args.profile}_iteration_times_task_{task_id}.csv"
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Iteration", "Time_Seconds"])
            writer.writerows(timing_data)
        print(f"Saved timing data to {csv_file_path}")

if __name__ == "__main__":
    main()