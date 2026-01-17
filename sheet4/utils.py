
import pandas as pd
import numpy as np
from typing import List, Tuple


def load_data(train_files: List[str], test_files: List[str]):
    """
    Loads HAI22.04 train/test files and returns:
    - X: physical readings (sensor + actuator columns)
    - y: attack labels
    """
    print("\n=== Loading Data for VAE & ML Pipeline ===")

    train_dfs = [pd.read_csv(f) for f in train_files]
    test_dfs  = [pd.read_csv(f) for f in test_files]

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df  = pd.concat(test_dfs, ignore_index=True)

    # Combine all rows (normal + attack)
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Drop timestamp
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])

    # Separate Attack label
    y = df['Attack'].values.astype(int)
    X = df.drop(columns=['Attack']).values.astype(float)

    print(f"Total samples: {X.shape[0]}")
    print(f"Physical reading dimension: {X.shape[1]}")
    print(f"Attack rows: {sum(y==1)}")

    return X, y

def create_windows_for_vae(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    mode: str,
):
    """
    Create sliding windows of M consecutive physical readings.

    - X: [T, F]  (T timesteps, F features)
    - y: [T]     (attack label per timestep)
    - window_size: M
    - mode: "reconstruction" or "classification"

    Returns:
      X_win: [N_windows, M, F]
      y_win: [N_windows] or None

    For reconstruction:
      - Only windows with all y == 0 (purely normal) are used.
    For classification:
      - All windows are used, label = 1 if any timestep in window is attack.
    """
    T, F = X.shape
    M = window_size

    X_windows = []
    y_windows = []

    for start in range(0, T - M + 1):
        end = start + M
        seg_X = X[start:end]      # [M, F]
        seg_y = y[start:end]      # [M]

        if mode == "reconstruction":
            if np.all(seg_y == 0):
                X_windows.append(seg_X)  # keep 3D: [M, F]
        else:
            X_windows.append(seg_X)
            y_windows.append(1 if np.any(seg_y == 1) else 0)

    X_windows = np.array(X_windows, dtype=np.float32)
    if mode == "reconstruction":
        return X_windows, None
    else:
        return X_windows, np.array(y_windows, dtype=int)
