
import pandas as pd
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


