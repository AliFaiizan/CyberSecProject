#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 3 – Adaptive N-gram Anomaly Detection (HAI 22.04)
------------------------------------------------------
• Raw physical readings (sensors + actuators)
• Normal-only training
• k-fold calibration (k = 5)
• EWMA-smoothed unseen n-gram ratio
"""

import math
import hashlib
import os
from glob import glob
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "/home/safiamed/CyberSecProject/datasets/hai-22.04/"
TRAIN_PATTERN = os.path.join(DATA_DIR, "train1.csv")
TEST_PATTERN  = os.path.join(DATA_DIR, "test1.csv")

# ---- N-gram parameters (CHANGE ONLY THESE) ----
N_GRAM_ORDER = 8        # set to 2, 5, or 8
NUM_FOLDS = 5
QUANTIZATION_LEVELS = 6

SLIDING_WINDOW = 80
EWMA_LAMBDA = 0.1
TARGET_FPR = 0.05
NUM_HASHES = 4

# ============================================================
# PHYSICAL READINGS (SENSORS + ACTUATORS)
# ============================================================

sensor_cols = [
    "P1_FT01","P1_FT02","P1_FT03","P1_FT01Z","P1_FT02Z","P1_FT03Z",
    "P1_LIT01","P1_PIT01","P1_PIT02","P1_TIT01","P1_TIT02",
    "P2_24Vdc","P2_SIT01","P2_VIBTR01","P2_VIBTR02","P2_VIBTR03",
    "P2_VIBTR04","P2_VT01",
    "P3_FIT01","P3_LIT01","P3_PIT01"
]

actuators = [
    "P1_FCV01Z","P1_FCV02Z","P1_FCV03Z","P1_LCV01Z",
    "P1_PCV01Z","P1_PCV02Z",
    "P1_PP01AR","P1_PP01BR","P1_PP02R"
]

PHYSICAL_READINGS = sensor_cols + actuators

# ============================================================
# BLOOM FILTER
# ============================================================

class BloomFilter:
    def __init__(self, size=1_000_000, num_hashes=3):
        self.size = size
        self.num_hashes = num_hashes
        self.bits = np.zeros(size, dtype=bool)

    def _hashes(self, value):
        for i in range(self.num_hashes):
            h = hashlib.sha1(f"{i}-{value}".encode()).hexdigest()
            yield int(h, 16) % self.size

    def add(self, value):
        for p in self._hashes(value):
            self.bits[p] = True

    def contains(self, value):
        return all(self.bits[p] for p in self._hashes(value))


def optimal_bloom_size(n, fpr=0.01):
    return max(1_000_000, int(-n * math.log(fpr) / (math.log(2) ** 2)))

# ============================================================
# HELPERS
# ============================================================

def ewma(values, lam):
    out = [values[0]]
    for v in values[1:]:
        out.append(lam * v + (1 - lam) * out[-1])
    return out


def make_states(arr):
    return ["_".join(map(str, row)) for row in arr]


def make_ngrams(states, n):
    return ["→".join(states[i:i+n]) for i in range(len(states)-n+1)]


def unseen_ratio(ngrams, bloom, window):
    buf, ratios = [], []
    for ng in ngrams:
        buf.append(ng)
        if len(buf) > window:
            buf.pop(0)
        ratios.append(sum(not bloom.contains(x) for x in buf) / len(buf))
    return ratios


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    train = pd.read_csv(TRAIN_PATTERN)
    test  = pd.read_csv(TEST_PATTERN)

    train = train[train["Attack"] == 0]

    train = train.drop(columns=["timestamp","Attack"], errors="ignore")
    test  = test.drop(columns=["timestamp","Attack"], errors="ignore")

    train = train[PHYSICAL_READINGS].ffill().fillna(0)
    test  = test[PHYSICAL_READINGS].ffill().fillna(0)

    print(f"Using {train.shape[1]} physical readings")

    return train, test


def normalize_quantize(df, mean, std):
    z = ((df - mean) / std).clip(-3,3).fillna(0)
    return ((z + 3) / 6 * QUANTIZATION_LEVELS).astype(int).values


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_tau(train, mean, std):
    fold_size = len(train) // NUM_FOLDS
    scores = []

    for i in range(1, NUM_FOLDS):
        tr = train.iloc[:i*fold_size]
        va = train.iloc[i*fold_size:(i+1)*fold_size]

        tr_q = normalize_quantize(tr, mean, std)
        va_q = normalize_quantize(va, mean, std)

        tr_ng = make_ngrams(make_states(tr_q), N_GRAM_ORDER)
        va_ng = make_ngrams(make_states(va_q), N_GRAM_ORDER)

        bloom = BloomFilter(optimal_bloom_size(len(set(tr_ng))), NUM_HASHES)
        for ng in tr_ng:
            bloom.add(ng)

        r = unseen_ratio(va_ng, bloom, SLIDING_WINDOW)
        scores.extend(ewma(r, EWMA_LAMBDA))

        print(f"[Fold {i}/{NUM_FOLDS}] Bloom utilization = {bloom.bits.mean():.3f}")

    tau = float(np.quantile(scores, 1 - TARGET_FPR))
    print(f"Calibrated τ = {tau:.4f}")
    return tau


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=== Adaptive N-gram Detection (HAI 22.04) ===")

    train_df, test_df = load_data()
    mean, std = train_df.mean(), train_df.std().replace(0,1e-6)

    tau = calibrate_tau(train_df, mean, std)

    train_q = normalize_quantize(train_df, mean, std)
    train_ng = make_ngrams(make_states(train_q), N_GRAM_ORDER)

    bloom = BloomFilter(optimal_bloom_size(len(set(train_ng))), NUM_HASHES)
    for ng in train_ng:
        bloom.add(ng)

    print(f"Final Bloom utilization = {bloom.bits.mean():.3f}")

    test_q = normalize_quantize(test_df, mean, std)
    test_ng = make_ngrams(make_states(test_q), N_GRAM_ORDER)

    scores = ewma(unseen_ratio(test_ng, bloom, SLIDING_WINDOW), EWMA_LAMBDA)
    preds = [s > tau for s in scores]

    print(f"Anomaly rate ≈ {np.mean(preds):.4f}")
    pd.DataFrame({"score": scores, "is_anomaly": preds}).to_csv(
        f"ngram_results_N{N_GRAM_ORDER}.csv", index=False
    )
