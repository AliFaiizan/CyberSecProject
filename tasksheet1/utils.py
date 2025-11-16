import numpy as np
import pandas as pd
from typing import List, Tuple

def load_and_clean_data(train_files: List[str], test_files: List[str], attack_cols: List[str]=None, label_files: List[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load & Clean Data
    - Read all train CSVs for HAI 21.03
    - Drop timestamp and attack labels
    - Remove rows where Attack == 1 (use only normal data for training)
    """
    print("\n=== Loading & Cleaning Data ===")
    
    # Load training data
    train_dfs = []
    for file in train_files:
        print(f"Loading {file}...")
        df = pd.read_csv(file)
        print(f"  Original shape: {df.shape}")
        
        # # Remove attack rows (keep only normal data for training)
        # if attack_cols and all(col in df.columns for col in attack_cols):
        #     normal_mask = df[attack_cols] == 0
        #     df = df[normal_mask]
        #     print(f"  After removing attacks from normal: {df.shape}")
        
        train_dfs.append(df)
    
    # Load test data
    test_dfs = []
    for file in test_files:
        print(f"Loading {file}...")
        df = pd.read_csv(file)
        print(f"  Original shape: {df.shape}")
        
        # Check if attack columns exist in dataframe
        if attack_cols:  # This checks if attack_cols is not None and not empty
            if all(col in df.columns for col in attack_cols):
                # Count rows with Attack == 1
                attack_count = (df[attack_cols] == 1).any(axis=1).sum()
                print(f"  Number of attack rows: {attack_count}")
                # Remove attack samples (keep only rows where all attack columns are 0)
                df = df[(df[attack_cols] == 0).all(axis=1)]
                print(f"  After removing attacks: {df.shape}")
        else:
            print("  No attack columns provided, skipping attack removal")
        
        test_dfs.append(df)
    

    # Combine all data
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Load and merge labels with test data BEFORE dropping timestamp
    if label_files is not None:
        label_dfs = []
        for label_file in label_files:
            print(f"Loading labels from {label_file}...")
            label_df = pd.read_csv(label_file)
            label_dfs.append(label_df)
        
        labels = pd.concat(label_dfs, ignore_index=True)
        
        # Count attack rows in labels
        if 'label' in labels.columns:
            attack_count = (labels['label'] == 1).sum()
            print(f"  Attack rows in labels: {attack_count}")
        
        # Merge on 'timestamp'
        if 'Timestamp' in test_df.columns:
            test_df = test_df.rename(columns={'Timestamp': 'timestamp'})
        test_df = pd.merge(test_df, labels, on='timestamp', how='left')
        print(f"  Labels merged. Test shape: {test_df.shape}")
        
        # Remove attack rows using label column
        if 'label' in test_df.columns:
            test_df = test_df[test_df['label'] == 0]
            print(f"  After removing attack rows: {test_df.shape}")
    # Drop timestamp and attack columns
    cols_to_drop = ['time', 'Timestamp', 'timestamp','label']

    if attack_cols:
        # Only add attack columns that exist in the DataFrame
        cols_to_drop += [col for col in attack_cols if col in train_df.columns]
        
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"Final training data shape: {train_df.shape}")
    print(f"Final test data shape: {test_df.shape}") 
    
    # # Handle NaN values
    # train_df = train_df.fillna(method='ffill').fillna(0)
    # test_df = test_df.fillna(method='ffill').fillna(0)

    return train_df, test_df


def normalize(train_data: pd.DataFrame, test_data: pd.DataFrame, Q: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize & Quantize
    - Use z-score normalization 
    - Quantize into discrete bins (0 to Q-1)
    """
    print(f"\n===Normalizing & Quantizing (Q={Q}) ===")
    # Round all values before normalization

    # Calculate z-score normalization parameters from training data only
    mean_vals = train_data.mean()
    std_vals = train_data.std()
    
    print(f"Data ranges before normalization:")
    print(f"  Train: min={train_data.min().min():.2f}, max={train_data.max().max():.2f}")
    print(f"  Test: min={test_data.min().min():.2f}, max={test_data.max().max():.2f}")
    
    # Z-score normalization
    train_normalized = (train_data - mean_vals) / std_vals
    test_normalized = (test_data - mean_vals) / std_vals
    
    # Handle any remaining NaN/inf values
    train_normalized = train_normalized.fillna(0)
    test_normalized = test_normalized.fillna(0)
    train_normalized = train_normalized.replace([np.inf, -np.inf], 0)
    test_normalized = test_normalized.replace([np.inf, -np.inf], 0)
    
    # # Clip extreme values to reasonable range (e.g., -3 to 3 standard deviations)
    # train_normalized = np.clip(train_normalized, -3, 3)
    # test_normalized = np.clip(test_normalized, -3, 3)
    
    # # Rescale to [0, 1] range for quantization
    # # Map [-3, 3] to [0, 1]
    # train_scaled = (train_normalized + 3) / 6
    # test_scaled = (test_normalized + 3) / 6

    train_quantized = train_normalized
    test_quantized = test_normalized
    #if Q is not there than return only scaled value
    if Q is not None:
        train_quantized = np.floor(train_normalized * Q).astype(int)
        test_quantized = np.floor(test_normalized * Q).astype(int)

        # Ensure values are in valid range
        train_quantized = np.clip(train_quantized, 0, Q-1)
        test_quantized = np.clip(test_quantized, 0, Q-1)
        print(f"Quantized ranges:")
        print(f"  Train: min={train_quantized.min()}, max={train_quantized.max()}")
        print(f"  Test: min={test_quantized.min()}, max={test_quantized.max()}")
    
    # Quantize into discrete bins [0, Q-1]
    
    return train_quantized, test_quantized

def empirical_cdf(sample):
    sorted_x = np.sort(sample)
    y = np.arange(1, len(sorted_x)+1) / len(sorted_x)
    return sorted_x, y

def ks_statistic(sample1, sample2):
    if len(sample1) == 0 or len(sample2) == 0:
        return np.nan
    x1, F1 = empirical_cdf(sample1)
    x2, F2 = empirical_cdf(sample2)
    all_x = np.sort(np.unique(np.concatenate([x1, x2])))
    F1_all = np.searchsorted(x1, all_x, side="right") / len(x1)
    F2_all = np.searchsorted(x2, all_x, side="right") / len(x2)
    D = np.max(np.abs(F1_all - F2_all))
    return D

def compare_datasets(dfA, dfB):
    ks_values = {}
    for col in dfA.columns:
        D = ks_statistic(dfA[col], dfB[col])
        ks_values[col] = D
    return ks_values

def compute_common_states(train_df: pd.DataFrame, test_df: pd.DataFrame, actuator_cols: List[str]) -> List[Tuple]:
    """
    Quantize actuator values into discrete steps and find common states
    """
    print("\n=== Computing Common States ===")

    # convert each row into a tuple and store in a Python set (which keeps only unique combinations)
    train_s = set(tuple(row) for row in train_df[actuator_cols].to_numpy())
    test_s = set(tuple(row) for row in test_df[actuator_cols].to_numpy())

    common_states = train_s & test_s
    print(len(train_s), len(test_s))
    print(f"Common states: {len(common_states)}")
    return train_s, test_s, list(common_states)


def quantize_valves(df, valve_cols, ignore=[], step=10):
    cols_to_quantize = [col for col in valve_cols if col not in ignore]
    df[cols_to_quantize] = df[cols_to_quantize].clip(lower=0, upper=100)
    df[cols_to_quantize] = df[cols_to_quantize].apply(lambda x: (x / step).round() * step)
    return df

def compute_ccdf(values):
    values = sorted(values)
    N = len(values)
    x = []
    y = []
    for i, v in enumerate(values, start=1):
        x.append(v)
        y.append(1 - i / N)
    return x, y