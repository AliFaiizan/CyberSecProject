import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

# Loading coloumn names
df1 = pd.read_csv('../hai-21.03/train1.csv', nrows=1)
df2 = pd.read_csv('../hai-22.04/train2.csv', nrows=1)
df3 = pd.read_csv('../hai-23.05/hai-train3.csv', nrows=1)

# Get sets of column names
cols1 = set(df1.columns)
cols2 = set(df2.columns)
cols3 = set(df3.columns)

# Find common columns across all three datasets
common_cols = cols1 & cols2 & cols3

# Filter columns that are likely to be sensors based on naming patterns
sensors = [col for col in common_cols if any(x in col for x in [
    'PIT', 'TIT', 'LIT', 'FT', 'FIT', 'VT', 'SIT', 'TT', 'PT', 'FD', 'LD', 'PO', 'PS'
])]
actuators = [col for col in common_cols if any(x in col for x in [
    'FCV', 'LCV', 'PCV', 'PP', 'Auto', 'On', 'Emgy', 'Trip'
])]



hai_21_files = sorted(glob("../hai-21.03/train*.csv")) + sorted(glob("../hai-21.03/test*.csv"))
hai_22_files = sorted(glob("../hai-22.04/train*.csv")) + sorted(glob("../hai-22.04/test*.csv"))
hai_23_files = sorted(glob("../hai-23.05/hai-train*.csv")) + sorted(glob("../hai-23.05/hai-test*.csv"))

hai_21_attack_cols = ['attack', 'attack_P1', 'attack_P2', 'attack_P3']

def load_and_clean(files, sensor_cols, attack_cols):
    """Load data, remove attack samples, and clean NaN values"""
    # Load sensor and attack columns
    df = pd.concat([pd.read_csv(f, usecols=sensor_cols + attack_cols) for f in files], ignore_index=True)
    
    # Remove attack samples (keep only rows where all attack columns are 0)
    df = df[(df[attack_cols] == 0).all(axis=1)]
    
    # Drop attack columns after filtering
    df = df.drop(columns=attack_cols)
    
    # Forward fill to handle any remaining NaN values
    df = df.fillna(method='ffill')
    
    return df


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

# Process hai-21.03 data
df21 = load_and_clean(hai_21_files, sensors, hai_21_attack_cols)
df22 = load_and_clean(hai_22_files, sensors, ['Attack'])


def z_score_normalize(df):
    """Z-score normalization (standardization)"""
    return (df - df.mean()) / df.std()

def min_max_normalize(df):
    """Min-Max normalization to [0, 1] range"""
    return (df - df.min()) / (df.max() - df.min())

# Normalize the datasets
df21_normalized = min_max_normalize(df21)
df22_normalized = min_max_normalize(df22)

ks_21_22 = compare_datasets(df21_normalized, df22_normalized)

print(ks_21_22)

def ccdf(values):
    vals = np.sort(values)
    y = 1 - np.arange(1, len(vals)+1)/len(vals)
    return vals, y

x, y = ccdf(list(ks_21_22.values()))

plt.plot(x, y)
plt.xlabel("K–S statistic")
plt.ylabel("CCDF")
plt.title("CCDF of K–S values (21.03 vs 22.04)")
plt.show()

plt.bar(range(len(ks_21_22)), list(ks_21_22.values()))
plt.xticks(range(len(ks_21_22)), list(ks_21_22.keys()), rotation=90)
plt.ylabel("K–S statistic (21.03 vs 22.04)")
plt.tight_layout()
plt.show()
