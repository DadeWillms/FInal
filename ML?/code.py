import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("clean_data.csv")

# ---- time fix ----
df["time"] = pd.to_datetime(df["time"], format="%I:%M%p", errors="coerce")
df = df.dropna(subset=["time"])

df["hour"] = df["time"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# ---- ensure numeric ----
df["people"] = pd.to_numeric(df["people"], errors="coerce")
df["max_people"] = pd.to_numeric(df["max_people"], errors="coerce")
df["occupancy_ratio"] = df["people"] / df["max_people"]

df = df.dropna()

# ---- features ----
X = df[[
    "temp",
    "humidity",
    "hour_sin",
    "hour_cos",
    "people",
    "max_people",
    "occupancy_ratio"
]]

y = df["co2"]

# ---- train/test ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))