import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt


df = pd.read_csv("FINAL/noisy_missing_dataset.csv")

total_missing = df.isnull().sum().sum()
percent_missing = (total_missing / df.size) * 100

print("Before Imputation")
print("Total Missing Values:", total_missing)
print("Percent Missing:", round(percent_missing, 2), "%")
print("\nMissing By Column:")
print(df.isnull().sum())


df["type"] = df["type"].map({"red":0, "white":1})
df = df.dropna(subset=["quality"])

y = df["quality"]
X = df.drop(columns=["quality"])

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

missing_by_col = df.isnull().sum()

# keep only columns with missing values
missing_by_col = missing_by_col[missing_by_col > 0]

plt.figure(figsize=(10,5))
plt.bar(missing_by_col.index, missing_by_col.values)
plt.title("Missing Values by Column (Before Imputation)")
plt.xlabel("Columns")
plt.ylabel("Number of Missing Values")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

imputer = KNNImputer(n_neighbors=5)
X_filled = imputer.fit_transform(X_scaled)

X_final = scaler.inverse_transform(X_filled)

X_final = pd.DataFrame(X_final, columns=X.columns)

X_final["quality"] = y.values

X_final = X_final.round(2)


print("\nAfter Imputation")
print("Remaining Missing Values:", X_final.isnull().sum().sum())


X_final.to_csv("FINAL/wine_cleaned.csv", index=False)

