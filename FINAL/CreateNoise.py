import pandas as pd
import numpy as np

df = pd.read_csv("FINAL/wine-quality-white-and-red.csv")

missing_percent_to_add = 0.05  # missing data %

rows, cols = df.shape
total_cells = rows * cols
num_missing = int(total_cells * missing_percent_to_add)

for _ in range(num_missing):
    i = np.random.randint(0, rows)
    j = np.random.randint(0, cols)
    df.iat[i, j] = np.nan


numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    std = df[col].std()
    noise = np.random.normal(0, std * 0.02, size=len(df))  # noise %
    df[col] = df[col] + noise

# Save new dataset
df.to_csv("noisy_missing_dataset.csv", index=False)

print("Done.")
print("New missing percentage:")
print(df.isnull().sum().sum() / df.size * 100, "%")