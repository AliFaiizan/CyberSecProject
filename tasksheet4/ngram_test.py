import pandas as pd
from glob import glob
import numpy as np
import hashlib
from typing import List, Tuple

# ============================================================
# FILE PATHS
# ============================================================

hai_train_files = sorted(glob("../datasets/hai-22.04/train1.csv"))
hai_test_files  = sorted(glob("../datasets/hai-22.04/test1.csv"))
ATTACK_COL = "Attack"

print(f"Found {len(hai_train_files)} training files")
print(f"Found {len(hai_test_files)} test files")

# ============================================================
# BLOOM FILTER
# ============================================================

class BloomFilter:
    """Bloom filter implementation for n-gram anomaly detection"""

    def __init__(self, size=1_000_000, k=5):
        self.size = size
        self.k = k
        self.bloom = np.zeros(size, dtype=bool)

    def _hashes(self, item: str) -> List[int]:
        return [
            int(hashlib.sha1((str(seed) + item).encode()).hexdigest(), 16) % self.size
            for seed in range(self.k)
        ]

    def add(self, item: str):
        for h in self._hashes(item):
            self.bloom[h] = True

    def check(self, item: str) -> bool:
        return all(self.bloom[h] for h in self._hashes(item))


# ============================================================
# DATA LOADING
# ============================================================

def load_and_clean_data(train_files, test_files):
    print("\n=== Step 1: Loading & Cleaning Data ===")

    features = [
        # actuators
        'P1_PCV01Z','P1_PCV02Z','P1_FCV01Z','P1_FCV02Z','P1_FCV03Z',
        'P1_LCV01Z','P1_PP01AR','P1_PP01BR','P1_PP02R',
        # sensors
        'P1_TIT01','P1_TIT02','P1_PIT01','P1_PIT02','P1_LIT01',
        'P1_FT01','P1_FT02','P1_FT03',
        'P2_SIT01','P2_VT01',
        'P3_FIT01','P3_LIT01','P3_PIT01',
        'P4_ST_TT01','P4_ST_PT01'
    ]

    train_df = pd.read_csv(train_files[0])
    test_df  = pd.read_csv(test_files[0])

    # keep normal only for training
    if ATTACK_COL in train_df.columns:
        train_df = train_df[train_df[ATTACK_COL] == 0]

    # drop timestamp + attack
    drop_cols = ['timestamp', ATTACK_COL]
    train_df = train_df.drop(columns=drop_cols, errors='ignore')
    test_df  = test_df.drop(columns=drop_cols, errors='ignore')

    train_df = train_df[features].ffill().fillna(0)
    test_df  = test_df[features].ffill().fillna(0)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape : {test_df.shape}")

    return train_df, test_df


# ============================================================
# NORMALIZATION & QUANTIZATION
# ============================================================

def normalize_and_quantize(train_df, test_df, Q=20):
    print(f"\n=== Step 2: Normalizing & Quantizing (Q={Q}) ===")

    mean = train_df.mean()
    std  = train_df.std().replace(0, 1e-6)

    train_z = ((train_df - mean) / std).clip(-3, 3)
    test_z  = ((test_df - mean) / std).clip(-3, 3)

    train_q = ((train_z + 3) / 6 * Q).astype(int).clip(0, Q-1).values
    test_q  = ((test_z + 3) / 6 * Q).astype(int).clip(0, Q-1).values

    return train_q, test_q


# ============================================================
# STATE + N-GRAM
# ============================================================

def build_state_strings(qdata):
    return ["_".join(map(str, row)) for row in qdata]

def generate_ngrams(states, n):
    return ["→".join(states[i:i+n]) for i in range(len(states)-n+1)]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n=== HAI N-gram Anomaly Detection ===")

    n = 8
    Q = 20
    k = 5
    M = 100_000

    # Step 1
    train_df, test_df = load_and_clean_data(hai_train_files, hai_test_files)

    # Step 2
    train_q, test_q = normalize_and_quantize(train_df, test_df, Q)

    # Step 3
    train_states = build_state_strings(train_q)
    test_states  = build_state_strings(test_q)

    # Step 4
    train_ngrams = generate_ngrams(train_states, n)
    test_ngrams  = generate_ngrams(test_states, n)

    # Step 5
    bloom = BloomFilter(size=M, k=k)
    for ng in train_ngrams:
        bloom.add(ng)

    # Detection
    scores = []
    for ng in test_ngrams:
        scores.append(0 if bloom.check(ng) else 1)

    anomaly_rate = np.mean(scores)

    print("\n=== Summary ===")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples : {len(test_df)}")
    print(f"N-gram order : {n}")
    print(f"Q levels     : {Q}")
    print(f"Bloom size   : {M}")
    print(f"Hash funcs  : {k}")
    print(f"Anomaly rate: {anomaly_rate:.4f}")

    pd.DataFrame({"anomaly": scores}).to_csv(
        "task3_anomaly_results2.csv", index=False
    )

    print("✓ Results saved to task3_anomaly_results2.csv")
