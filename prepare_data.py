import pandas as pd

# Read original file that has a single string column
with open("data.csv", "r") as f:
    lines = f.readlines()

# Split that string by commas manually
clean_rows = [line.strip().split(",") for line in lines]

# Remove quotes or junk characters
clean_rows = [[val.strip('"').strip("'") for val in row] for row in clean_rows]

# First row is header
header = clean_rows[0]
data = clean_rows[1:]

# Create DataFrame
df = pd.DataFrame(data, columns=header)

# Convert all values to float
df = df.apply(pd.to_numeric, errors="coerce")

# Drop the label column if it exists
if "quality" in df.columns:
    df.drop(columns=["quality"], inplace=True)

# Save as clean, real CSV
df.to_csv("clean_data.csv", index=False)
print("✅ clean_data.csv is now fully fixed and numeric")
