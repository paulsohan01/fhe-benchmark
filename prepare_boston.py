import pandas as pd

# Load the Boston Housing dataset
df = pd.read_csv("boston.csv")

# Drop the target column if it exists (either 'medv' or 'MEDV')
for label in ['medv', 'MEDV']:
    if label in df.columns:
        df.drop(columns=[label], inplace=True)

# Convert everything to float (ensure numeric)
df = df.apply(pd.to_numeric, errors='coerce')

# Save clean numeric CSV
df.to_csv("clean_boston.csv", index=False)

print("✅ Cleaned and saved as clean_boston.csv")
