import pandas as pd
from ctgan import CTGAN
from ctgan import TVAE
import statsmodels.formula.api as smf

df_raw = pd.read_csv(f"./data/Sample/SRS/0001.csv").reset_index(drop=True)
df_raw = df_raw.loc[:, ~df_raw.columns.str.contains('^Unnamed')]
df_raw = df_raw.dropna()
# 2. Define your categorical variables
# CTGAN needs to know which columns are discrete (categorical/string/boolean)
discrete_columns = [
    "idx", "usborn", "high_chol", "female",
    "bkg_pr", "bkg_o", "hypertension", "R"
]

# 3. Initialize the model
# You can customize hyperparameters here (e.g., batch_size, generator_dim)

ctgan_model = CTGAN(epochs=10000, verbose=True, pac=1)
#tvae_model = TVAE(epochs=2000, verbose=True)

# 4. Fit the model to your data
print("Training the CTGAN model...")
ctgan_model.fit(df_raw, discrete_columns)
#tvae_model.fit(df_raw, discrete_columns)

# 5. Generate new synthetic data
num_rows_to_generate = 1000
synthetic_data = ctgan_model.sample(num_rows_to_generate)
#synthetic_data = tvae_model.sample(num_rows_to_generate)

print("\n--- Synthetic Data Sample ---")
print(synthetic_data.head())

synth_model = smf.ols(formula='sbp ~ ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr + ln_na_true:c_age',
                      data=synthetic_data)
synth_results = synth_model.fit()

# 2. Print the extracted coefficients
print("\nSynthetic Data Coefficients:")
print(synth_results.params)
# 6. Save the model to disk (Optional)
# ctgan_model.save('my_trained_ctgan.pkl')