import yaml
from benchmarks.tabcsdi.tabcsdi import TabCSDI
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
    path = os.path.join(base_dir, strategy, "tabcsdi")
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

with open("./data/Config/best_config_tabcsdi_srs.yaml", "r") as f:
    config_srs = yaml.safe_load(f)

# with open("./data/Config/best_config_tabcsdi_rs.yaml", "r") as f:
#     config_rs = yaml.safe_load(f)
#
# with open("./data/Config/best_config_tabcsdi_wrs.yaml", "r") as f:
#     config_wrs = yaml.safe_load(f)
#
# with open("./data/Config/best_config_tabcsdi_sfs.yaml", "r") as f:
#     config_sfs = yaml.safe_load(f)
#
# with open("./data/Config/best_config_tabcsdi_ods_tail.yaml", "r") as f:
#     config_ods_tail = yaml.safe_load(f)
#
# with open("./data/Config/best_config_tabcsdi_neyman_ods.yaml", "r") as f:
#     config_neyman_ods = yaml.safe_load(f)
#
# with open("./data/Config/best_config_tabcsdi_neyman_inf.yaml", "r") as f:
#     config_neyman_inf = yaml.safe_load(f)

for i in range(1, 2):
    digit = str(i).zfill(4)
    file_path_srs = f"./data/Sample/SRS/{digit}.csv"
    file_path_rs = f"./data/Sample/RS/{digit}.csv"
    file_path_wrs = f"./data/Sample/WRS/{digit}.csv"
    file_path_sfs = f"./data/Sample/SFS/{digit}.csv"
    file_path_ods = f"./data/Sample/ODS_TAIL/{digit}.csv"
    file_path_nod = f"./data/Sample/Neyman_ODS/{digit}.csv"
    file_path_nif = f"./data/Sample/Neyman_INF/{digit}.csv"

    save_path_srs = f"./simulations/SRS/tabcsdi/{digit}.parquet"
    save_path_rs = f"./simulations/RS/tabcsdi/{digit}.parquet"
    save_path_wrs = f"./simulations/WRS/tabcsdi/{digit}.parquet"
    save_path_sfs = f"./simulations/SFS/tabcsdi/{digit}.parquet"
    save_path_ods = f"./simulations/ODS_TAIL/tabcsdi/{digit}.parquet"
    save_path_nod = f"./simulations/Neyman_ODS/tabcsdi/{digit}.parquet"
    save_path_nif = f"./simulations/Neyman_INF/tabcsdi/{digit}.parquet"

    # SRS
    tabcsdi_mod_srs = TabCSDI(config_srs, data_info_srs)
    tabcsdi_mod_srs.fit(file_path_srs)
    tabcsdi_mod_srs.impute(save_path=save_path_srs)

    # RS
    # tabcsdi_mod_rs = TabCSDI(config_rs, data_info_rs)
    # tabcsdi_mod_rs.fit(file_path_rs)
    # tabcsdi_mod_rs.impute(save_path=save_path_rs)
    #
    # # WRS
    # tabcsdi_mod_wrs = TabCSDI(config_wrs, data_info_wrs)
    # tabcsdi_mod_wrs.fit(file_path_wrs)
    # tabcsdi_mod_wrs.impute(save_path=save_path_wrs)
    #
    # # SFS
    # tabcsdi_mod_sfs = TabCSDI(config_sfs, data_info_sfs)
    # tabcsdi_mod_sfs.fit(file_path_sfs)
    # tabcsdi_mod_sfs.impute(save_path=save_path_sfs)
    #
    # # ODS_TAIL
    # tabcsdi_mod_ods = TabCSDI(config_ods_tail, data_info_srs)
    # tabcsdi_mod_ods.fit(file_path_ods)
    # tabcsdi_mod_ods.impute(save_path=save_path_ods)
    #
    # # Neyman_ODS
    # tabcsdi_mod_nod = TabCSDI(config_neyman_inf, data_info_neyman_ods)
    # tabcsdi_mod_nod.fit(file_path_nod)
    # tabcsdi_mod_nod.impute(save_path=save_path_nod)
    #
    # # Neyman_INF
    # tabcsdi_mod_nif = TabCSDI(config_neyman_ods, data_info_neyman_inf)
    # tabcsdi_mod_nif.fit(file_path_nif)
    # tabcsdi_mod_nif.impute(save_path=save_path_nif)


