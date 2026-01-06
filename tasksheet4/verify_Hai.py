import pandas as pd
import numpy as np
from glob import glob

DATA_DIR = "../datasets/hai-22.04/"
FILES = sorted(glob(f"{DATA_DIR}/*.csv"))

df = pd.concat([pd.read_csv(f) for f in FILES], ignore_index=True)

print("=== BASIC INFO ===")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

assert "Attack" in df.columns, "Attack column not found!"

# Separate
normal_df = df[df["Attack"] == 0]
attack_df = df[df["Attack"] == 1]

print("\n=== CLASS DISTRIBUTION ===")
print(df["Attack"].value_counts())
print("Normal ratio:", len(normal_df) / len(df))
print("Attack ratio:", len(attack_df) / len(df))

# Drop non-sensor columns
sensor_df = df.drop(columns=["Attack"], errors="ignore")
sensor_df = sensor_df.select_dtypes(include=[np.number])

normal_sensors = sensor_df.loc[normal_df.index]
attack_sensors = sensor_df.loc[attack_df.index]

print("\n=== SHAPE (sensors only) ===")
print("Total:", sensor_df.shape)
print("Normal:", normal_sensors.shape)
print("Attack:", attack_sensors.shape)

print("\n=== MEAN (first 10 sensors) ===")
print(pd.DataFrame({
    "normal_mean": normal_sensors.mean(),
    "attack_mean": attack_sensors.mean()
}).head(10))

print("\n=== STD (first 10 sensors) ===")
print(pd.DataFrame({
    "normal_std": normal_sensors.std(),
    "attack_std": attack_sensors.std()
}).head(10))

print("\n=== MEAN ABS DIFF (top 10 sensors) ===")
diff = (normal_sensors.mean() - attack_sensors.mean()).abs()
print(diff.sort_values(ascending=False).head(10))
