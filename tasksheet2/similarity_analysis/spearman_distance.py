import os
import numpy as np
import pandas as pd
from haien_mapping import haien_mapping_dict


# ---------------------------------------------------------
# Manual Spearman correlation and distance
# ---------------------------------------------------------
def spearman_corr_manual(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation manually (rank + Pearson)."""
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    return np.mean(rx * ry)


def spearman_distance_manual(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman distance = 1 − Spearman correlation."""
    return 1.0 - spearman_corr_manual(x, y)


# ---------------------------------------------------------
# Compute consecutive Spearman distances
# ---------------------------------------------------------
def compute_consecutive_spearman(df: pd.DataFrame):
    """Compute Spearman distance only between consecutive time steps."""
    n = len(df)
    out = np.zeros(n - 1, dtype=np.float32)
    for i in range(n - 1):
        out[i] = spearman_distance_manual(df.iloc[i].values, df.iloc[i + 1].values)
        if i % 10000 == 0 and i > 0:
            print(f"  processed {i}/{n-1}")
    return out


# ---------------------------------------------------------
# Load dataset and identify attack segments
# ---------------------------------------------------------
def load_dataset(base_path, version):
    folder = os.path.join(base_path, version)

    # Identify correct CSV prefixes
    if "haiend" in version:
        patterns = ("end-train", "end-test")
    else:
        patterns = ("train", "test")

    csv_files = [
        os.path.join(folder, f) for f in sorted(os.listdir(folder))
        if f.endswith(".csv") and f.startswith(patterns)
    ]
    if not csv_files:
        raise FileNotFoundError(f"No train/test CSV files found in {folder}")

    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    print(f"Loaded {version}: {df.shape}")

    # Detect time/timestamp column
    time_candidates = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
    time_col = time_candidates[0] if time_candidates else None
    if time_col:
        print(f"Detected time column: {time_col}")

    # Load continuous sensors
    sensor_file = os.path.join(folder, "continuous_sensors.txt")
    with open(sensor_file) as f:
        sensors = [s.strip() for s in f if s.strip()]

    # Apply HAIEnd mapping
    if "haiend" in version:
        mapped_sensors = []
        for s in sensors:
            match = next((k for k, v in haien_mapping_dict.items() if v == s), s)
            if match in df.columns:
                mapped_sensors.append(match)
        sensors = mapped_sensors
        print(f"Using {len(sensors)} mapped HAIEnd sensors.")

    # Verify available sensors
    available = [s for s in sensors if s in df.columns]
    missing = set(sensors) - set(available)
    if missing:
        print(f"Warning: {len(missing)} sensors not found in {version}: {sorted(missing)}")

    df_sensors = df[available].copy()

    # Dataset-specific attack detection
    if "haiend" in version:
        label_file = next(
            (os.path.join(folder, f) for f in os.listdir(folder)
             if f.startswith("label-test") and f.endswith(".csv")), None)
        if label_file:
            label_df = pd.read_csv(label_file)
            label_cols = [c for c in label_df.columns if "label" in c.lower()]
            time_cols = [c for c in label_df.columns if "time" in c.lower() or "timestamp" in c.lower()]
            if not label_cols or not time_cols:
                raise KeyError("Could not identify label/time columns in label-test CSV.")

            label_col, label_time_col = label_cols[0], time_cols[0]
            if time_col and time_col != label_time_col:
                df = df.rename(columns={time_col: label_time_col})

            df = df.merge(label_df[[label_time_col, label_col]], on=label_time_col, how="left")
            df[label_col] = df[label_col].fillna(0)
            df["attack_flag"] = df[label_col].astype(int) == 1
        else:
            print("No label-test file found; assuming all normal.")
            df["attack_flag"] = False

    else:
        attack_cols = [c for c in df.columns if c.lower().startswith("attack")]
        if attack_cols:
            df["attack_flag"] = df[attack_cols].sum(axis=1) > 0
        elif "attack" in [c.lower() for c in df.columns]:
            df["attack_flag"] = df["Attack"].astype(bool)
        else:
            df["attack_flag"] = False

    # Split normal vs attack
    normal_df = df_sensors[~df["attack_flag"]].reset_index(drop=True)
    attack_df = df_sensors[df["attack_flag"]].reset_index(drop=True)
    print(f"Normal rows: {len(normal_df)}, Attack rows: {len(attack_df)}")

    return normal_df, attack_df


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    base = "/home/safiamed/Study_Project/datasets"
    version = "haiend-23.05"  # or "hai-21.03", "hai-22.04"

    normal_df, attack_df = load_dataset(base, version)

    if len(normal_df) > 1:
        d_normal = compute_consecutive_spearman(normal_df)
        np.save(f"spearman_{version}_normal.npy", d_normal)
        print(f"Saved consecutive normal distances: {len(d_normal)}")

    if len(attack_df) > 1:
        d_attack = compute_consecutive_spearman(attack_df)
        np.save(f"spearman_{version}_attack.npy", d_attack)
        print(f"Saved consecutive attack distances: {len(d_attack)}")

