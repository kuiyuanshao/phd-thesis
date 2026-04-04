import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np

# Ensure your custom modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from SIRD.data_transformer import DataTransformer

# 1. Define the data_info
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
        "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
        "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
        "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806",
        "HbA1c", "Creatinine", "eGFR", "WEIGHT", "BMI",
        "SMOKE", "INCOME", "ALC", "EXER", "EDU",
        "Glucose", "F_Glucose", "Insulin", "Na_INTAKE", "K_INTAKE",
        "KCAL_INTAKE", "PROTEIN_INTAKE", "C", "EVENT", "T_I"
    ],
    "phase1_vars": [
        "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR", "rs17036101_STAR",
        "rs17584499_STAR", "rs1111875_STAR", "rs4402960_STAR", "rs4607103_STAR",
        "rs7754840_STAR", "rs9300039_STAR", "rs5015480_STAR", "rs9465871_STAR",
        "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
        "HbA1c_STAR", "Creatinine_STAR", "eGFR_STAR", "WEIGHT_STAR", "BMI_STAR",
        "SMOKE_STAR", "INCOME_STAR", "ALC_STAR", "EXER_STAR", "EDU_STAR",
        "Glucose_STAR", "F_Glucose_STAR", "Insulin_STAR", "Na_INTAKE_STAR", "K_INTAKE_STAR",
        "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR", "C_STAR", "EVENT_STAR", "T_I_STAR"
    ]
}


def compare_my_transformer(csv_path):
    # 1. Load the data
    df = pd.read_csv(csv_path)

    # 2. Use a simplified data_info structure for plotting (or you can use data_info_srs_oe above)
    data_info = {
        "num_vars": ["HbA1c", "HbA1c_STAR", "AGE", "BMI", "WEIGHT"],
        "cat_vars": ["SEX", "RACE"],
        "phase1_vars": ["HbA1c_STAR"],
        "phase2_vars": ["HbA1c"]
    }

    # Added spline and kde_spline since your code supports them!
    techniques = ["zscore", "quantile", "spline", "kde_spline"]

    # We will plot the Original Data on a separate figure or axis if you want later,
    # but grouping it with Z-Score or MinMax flattens the view.
    # Let's plot just the normalized techniques together to compare their distributions cleanly.
    plt.figure(figsize=(12, 7))

    for tech in techniques:
        # Create the config your transformer expects
        config = {
            "processing": {
                "normalization": tech,
                "log": True,  # Keeps it to pure normalization based on your code
                "anchor": False
            },
            "diffusion": {"discrete": "Multinomial"}
        }

        # Initialize DataTransformer
        transformer = DataTransformer(data_info, config)

        # FIT: Calculates means, mins, maxes, or fits splines/quantiles
        transformer.fit(df)

        # TRANSFORM: Actually applies the math and returns the final DataFrame
        # Your class method `transform()` takes no arguments, it acts on self.df_transformed
        transformed_df = transformer.transform()

        # Extract the results
        hba1c_scaled = transformed_df['HbA1c']

        # Filter out the 0.0 padding your code assigns to NaNs/masked values
        # Your code does `res_val[~mask] = 0.0`, so we exclude absolute 0.0s to avoid a massive spike at 0
        hba1c_scaled = hba1c_scaled[hba1c_scaled != 0.0]

        # Plot density
        sns.kdeplot(hba1c_scaled, label=f'Method: {tech}', linewidth=2)

    plt.title("Comparison of Normalization Techniques on HbA1c", fontsize=16)
    plt.xlabel("Fully Normalized Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Point this to your actual file
    my_file = "./data/SampleE/srs/0001.csv"
    if os.path.exists(my_file):
        compare_my_transformer(my_file)
    else:
        print(f"Error: Could not find {my_file}")