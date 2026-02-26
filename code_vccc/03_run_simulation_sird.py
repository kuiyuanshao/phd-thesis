import yaml
from sird.sird import SIRD
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
    path = os.path.join(base_dir, strategy, "sird")
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created: {path}")

data_info_srs = {
    "weight_var": "W",
    "cat_vars": [
        "SEX", "RACEETH", "RISK1", "RISK2", "DEATH_EVER",
        "GENERIC_ART_NAME_v", "GENERIC_ART_NAME_nv",
        "SOURCE_nv", "ART_SOURCE_nv",
        "DIAGNOSIS_v", "DIAGNOSIS_nv",
        "ANY_OI_v", "ANY_OI_nv"
    ],
    "num_vars": [
        "CFAR_PID", "AGE_AT_FIRST_VISIT", "AGE_AT_LAST_VISIT", "AGE_AT_DEATH",
        "YEAR_OF_ENROLLMENT", "YEAR_OF_LAST_VISIT", "YEAR_OF_DEATH", "YEARS_IN_STUDY",
        "AGE_AT_MED_START_v", "AGE_AT_MED_STOP_v",
        "AGE_AT_MED_START_nv", "AGE_AT_MED_STOP_nv",
        "AGE_AT_DX_ONSET_v", "YEAR_OF_DX_ONSET_v",
        "AGE_AT_DX_ONSET_nv", "YEAR_OF_DX_ONSET_nv",
        "CD4_COUNT_BSL_v", "CD4_COUNT_1Y_v", "VL_COUNT_BSL_v", "VL_COUNT_BSL_nv",
        "CD4_COUNT_BSL_sqrt_v", "CD4_COUNT_1Y_sqrt_v", "VL_COUNT_BSL_LOG_v", "VL_COUNT_BSL_LOG_nv"
    ],
    "phase2_vars": [
        "AGE_AT_MED_START_v", "AGE_AT_MED_STOP_v", "GENERIC_ART_NAME_v",
        "AGE_AT_DX_ONSET_v", "DIAGNOSIS_v", "YEAR_OF_DX_ONSET_v", "ANY_OI_v",
        "VL_COUNT_BSL_v",
        "VL_COUNT_BSL_LOG_v"
    ],
    "phase1_vars": [
        "AGE_AT_MED_START_nv", "AGE_AT_MED_STOP_nv", "GENERIC_ART_NAME_nv",
        "AGE_AT_DX_ONSET_nv", "DIAGNOSIS_nv", "YEAR_OF_DX_ONSET_nv", "ANY_OI_nv",
        "VL_COUNT_BSL_nv",
        "VL_COUNT_BSL_LOG_nv"
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



