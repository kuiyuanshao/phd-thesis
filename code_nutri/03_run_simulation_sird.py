import yaml
from SIRD.sird import SIRD
import os

base_dir = "./simulations"
strategies = [
    "SRS",
    "ODS_TAIL",
    "RS",
    "WRS",
    "SFS",
    "Neyman_ODS",
    "Neyman_INF"
]

# Create the directories for each strategy
for strategy in strategies:
    path = os.path.join(base_dir, strategy, "SIRD")
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created: {path}")

data_info_srs = {
    "weight_var": "W",
    "cat_vars": [
        "idx", "usborn", "high_chol", "female",
        "bkg_pr", "bkg_o", "hypertension", "R"
    ],
    "num_vars": [
        "id", "c_age", "c_bmi", "sbp",
        "ln_na_avg", "ln_k_avg", "ln_kcal_avg", "ln_protein_avg",
        "ln_na_true", "ln_k_true", "ln_kcal_true", "ln_protein_true",
        "W"
    ],
    "phase2_vars": [
        "ln_na_true", "ln_k_true", "ln_kcal_true", "ln_protein_true"
    ],
    "phase1_vars": [
        "ln_na_avg", "ln_k_avg", "ln_kcal_avg", "ln_protein_avg"
    ]
}

data_info_rs = {
    "weight_var": "W",
    "cat_vars": data_info_srs["cat_vars"],
    "num_vars": data_info_srs["num_vars"] + ["rs"],
    "phase2_vars": data_info_srs["phase2_vars"],
    "phase1_vars": data_info_srs["phase1_vars"]
}

data_info_wrs = {
    "weight_var": "W",
    "cat_vars": data_info_srs["cat_vars"],
    "num_vars": data_info_srs["num_vars"] + ["wrs"],
    "phase2_vars": data_info_srs["phase2_vars"],
    "phase1_vars": data_info_srs["phase1_vars"]
}

data_info_sfs = {
    "weight_var": "W",
    "cat_vars": data_info_srs["cat_vars"],
    "num_vars": data_info_srs["num_vars"] + ["score"],
    "phase2_vars": data_info_srs["phase2_vars"],
    "phase1_vars": data_info_srs["phase1_vars"]
}

data_info_neyman_ods = {
    "weight_var": "W",
    "cat_vars": data_info_srs["cat_vars"] + ["outcome_strata"],
    "num_vars": data_info_srs["num_vars"],
    "phase2_vars": data_info_srs["phase2_vars"],
    "phase1_vars": data_info_srs["phase1_vars"]
}

data_info_neyman_inf = {
    "weight_var": "W",
    "cat_vars": data_info_srs["cat_vars"] + ["outcome_strata"],
    "num_vars": data_info_srs["num_vars"] + ["inf"],
    "phase2_vars": data_info_srs["phase2_vars"],
    "phase1_vars": data_info_srs["phase1_vars"]
}

with open("./data/Config/best_config_rddm_srs.yaml", "r") as f:
    config_srs = yaml.safe_load(f)

# with open("./data/Config/best_config_rddm_rs.yaml", "r") as f:
#     config_rs = yaml.safe_load(f)
#
# with open("./data/Config/best_config_rddm_wrs.yaml", "r") as f:
#     config_wrs = yaml.safe_load(f)
#
# with open("./data/Config/best_config_rddm_sfs.yaml", "r") as f:
#     config_sfs = yaml.safe_load(f)
#
# with open("./data/Config/best_config_rddm_ods_tail.yaml", "r") as f:
#     config_ods_tail = yaml.safe_load(f)
#
# with open("./data/Config/best_config_rddm_neyman_ods.yaml", "r") as f:
#     config_neyman_ods = yaml.safe_load(f)
#
# with open("./data/Config/best_config_rddm_neyman_inf.yaml", "r") as f:
#     config_neyman_inf = yaml.safe_load(f)

for i in range(1, 2):
    digit = str(i).zfill(4)
    # Input Paths (from R output)
    file_path_srs = f"./data/Sample/SRS/{digit}.csv"
    file_path_rs = f"./data/Sample/RS/{digit}.csv"
    file_path_wrs = f"./data/Sample/WRS/{digit}.csv"
    file_path_sfs = f"./data/Sample/SFS/{digit}.csv"
    file_path_ods = f"./data/Sample/ODS_TAIL/{digit}.csv"
    file_path_nod = f"./data/Sample/Neyman_ODS/{digit}.csv"
    file_path_nif = f"./data/Sample/Neyman_INF/{digit}.csv"

    # Output Paths (for Python simulations)
    save_path_srs = f"./simulations/SRS/sird/{digit}.parquet"
    save_path_rs = f"./simulations/RS/sird/{digit}.parquet"
    save_path_wrs = f"./simulations/WRS/sird/{digit}.parquet"
    save_path_sfs = f"./simulations/SFS/sird/{digit}.parquet"
    save_path_ods = f"./simulations/ODS_TAIL/sird/{digit}.parquet"
    save_path_nod = f"./simulations/Neyman_ODS/sird/{digit}.parquet"
    save_path_nif = f"./simulations/Neyman_INF/sird/{digit}.parquet"

    sird_mod_srs = SIRD(config_srs, data_info_srs)
    sird_mod_srs.fit(file_path_srs)
    sird_mod_srs.impute(save_path=save_path_srs)



