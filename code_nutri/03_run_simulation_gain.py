import yaml
from benchmarks.gain.gain import GAIN
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
    path = os.path.join(base_dir, strategy, "gain")
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created: {path}")

data_info_srs = {
    "cat_vars": [
        "idx", "usborn", "high_chol", "female",
        "bkg_pr", "bkg_o", "hypertension", "R"
    ],
    "num_vars": [
        "id", "c_age", "c_bmi", "sbp",
        "ln_na_avg", "ln_k_avg", "ln_kcal_avg", "ln_protein_avg",
        "ln_na_true", "ln_k_true", "ln_kcal_true", "ln_protein_true",
        "W"
    ]
}

data_info_rs = {
    "cat_vars": data_info_srs["cat_vars"],
    "num_vars": data_info_srs["num_vars"] + ["rs"]
}

data_info_wrs = {
    "cat_vars": data_info_srs["cat_vars"],
    "num_vars": data_info_srs["num_vars"] + ["wrs"]
}

data_info_sfs = {
    "cat_vars": data_info_srs["cat_vars"],
    "num_vars": data_info_srs["num_vars"] + ["score"]
}

data_info_neyman_ods = {
    "cat_vars": data_info_srs["cat_vars"] + ["outcome_strata"],
    "num_vars": data_info_srs["num_vars"]
}

data_info_neyman_inf = {
    "cat_vars": data_info_srs["cat_vars"] + ["outcome_strata"],
    "num_vars": data_info_srs["num_vars"] + ["inf"]
}

with open("./data/Config/best_config_gain_srs.yaml", "r") as f:
    config_srs = yaml.safe_load(f)

with open("./data/Config/best_config_gain_rs.yaml", "r") as f:
    config_rs = yaml.safe_load(f)

with open("./data/Config/best_config_gain_wrs.yaml", "r") as f:
    config_wrs = yaml.safe_load(f)

with open("./data/Config/best_config_gain_sfs.yaml", "r") as f:
    config_sfs = yaml.safe_load(f)

with open("./data/Config/best_config_gain_ods_tail.yaml", "r") as f:
    config_ods_tail = yaml.safe_load(f)

with open("./data/Config/best_config_gain_neyman_ods.yaml", "r") as f:
    config_neyman_ods = yaml.safe_load(f)

with open("./data/Config/best_config_gain_neyman_inf.yaml", "r") as f:
    config_neyman_inf = yaml.safe_load(f)

for i in range(1, 2):
    digit = str(i).zfill(4)
    file_path_srs = f"./data/Sample/SRS/{digit}.csv"
    file_path_rs = f"./data/Sample/RS/{digit}.csv"
    file_path_wrs = f"./data/Sample/WRS/{digit}.csv"
    file_path_sfs = f"./data/Sample/SFS/{digit}.csv"
    file_path_ods = f"./data/Sample/ODS_TAIL/{digit}.csv"
    file_path_nod = f"./data/Sample/Neyman_ODS/{digit}.csv"
    file_path_nif = f"./data/Sample/Neyman_INF/{digit}.csv"

    save_path_srs = f"./simulations/SRS/gain/{digit}.parquet"
    save_path_rs = f"./simulations/RS/gain/{digit}.parquet"
    save_path_wrs = f"./simulations/WRS/gain/{digit}.parquet"
    save_path_sfs = f"./simulations/SFS/gain/{digit}.parquet"
    save_path_ods = f"./simulations/ODS_TAIL/gain/{digit}.parquet"
    save_path_nod = f"./simulations/Neyman_ODS/gain/{digit}.parquet"
    save_path_nif = f"./simulations/Neyman_INF/gain/{digit}.parquet"

    # SRS
    gain_mod_srs = GAIN(config_srs, data_info_srs)
    gain_mod_srs.fit(file_path_srs)
    gain_mod_srs.impute(save_path=save_path_srs)

    # RS
    gain_mod_rs = GAIN(config_rs, data_info_rs)
    gain_mod_rs.fit(file_path_rs)
    gain_mod_rs.impute(save_path=save_path_rs)

    # WRS
    gain_mod_wrs = GAIN(config_wrs, data_info_wrs)
    gain_mod_wrs.fit(file_path_wrs)
    gain_mod_wrs.impute(save_path=save_path_wrs)

    # SFS
    gain_mod_sfs = GAIN(config_sfs, data_info_sfs)
    gain_mod_sfs.fit(file_path_sfs)
    gain_mod_sfs.impute(save_path=save_path_sfs)

    # ODS_TAIL
    gain_mod_ods = GAIN(config_ods_tail, data_info_srs)
    gain_mod_ods.fit(file_path_ods)
    gain_mod_ods.impute(save_path=save_path_ods)

    # Neyman_ODS
    gain_mod_nod = GAIN(config_neyman_inf, data_info_neyman_ods)
    gain_mod_nod.fit(file_path_nod)
    gain_mod_nod.impute(save_path=save_path_nod)

    # Neyman_INF
    gain_mod_nif = GAIN(config_neyman_ods, data_info_neyman_inf)
    gain_mod_nif.fit(file_path_nif)
    gain_mod_nif.impute(save_path=save_path_nif)


