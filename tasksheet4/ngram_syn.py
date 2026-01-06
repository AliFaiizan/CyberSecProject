#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 3 – Adaptive N-gram Anomaly Detection (GAN data)
----------------------------------------------------
• Uses GAN-generated synthetic training and testing data (.npy)
• Same structure as CSV-based N-gram implementation
• Normal-only synthetic train, mixed synthetic test
"""

import math
import hashlib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SYN_TRAIN = "synthetic_train.npy"
SYN_TEST  = "synthetic_test.npy"
SYN_TEST_LABELS = "synthetic_test_labels.npy"  # optional

# Physical readings (same as your selection)
sensor_cols = [
    "P1_FT01","P1_FT02","P1_FT03","P1_FT01Z","P1_FT02Z","P1_FT03Z",
    "P1_LIT01","P1_PIT01","P1_PIT02","P1_TIT01","P1_TIT02",
    "P2_24Vdc","P2_SIT01","P2_VIBTR01","P2_VIBTR02",
    "P2_VIBTR03","P2_VIBTR04","P2_VT01",
    "P3_FIT01","P3_LIT01","P3_PIT01",
    "P1_FCV01Z","P1_FCV02Z","P1_FCV03Z",
    "P1_LCV01Z","P1_PCV01Z","P1_PCV02Z",
    "P1_PP01AR","P1_PP01BR","P1_PP02R"
]

# Model parameters
QUANTIZATION_LEVELS = 6
N_GRAM_ORDER = 5      # change to 5 or 8
SLIDING_WINDOW = 80
EWMA_LAMBDA = 0.1
TARGET_FPR = 0.05
NUM_HASHES = 4
NUM_FOLDS = 5


# ============================================================
# BLOOM FILTER
# ============================================================

class BloomFilter:
    def __init__(self, size=1_000_000, num_hashes=3):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = np.zeros(size, dtype=bool)

    def _hashes(self, value):
        for i in range(self.num_hashes):
            h = hashlib.sha1(f"{i}-{value}".encode()).hexdigest()
            yield int(h, 16) % self.size

    def add(self, value):
        for p in self._hashes(value):
            self.bit_array[p] = True

    def contains(self, value):
        return all(self.bit_array[p] for p in self._hashes(value))


def optimal_bloom_size(n_items, fpr=0.01):
    return max(1_000_000, int(-n_items * math.log(fpr) / (math.log(2) ** 2)))


# ============================================================
# HELPERS
# ============================================================

def exponential_smoothing(values, lam=EWMA_LAMBDA):
    smoothed = []
    last = values[0]
    for v in values:
        last = lam * v + (1 - lam) * last
        smoothed.append(last)
    return smoothed


def unseen_ngram_ratio(ngrams, bloom, window=SLIDING_WINDOW):
    buf, ratios = [], []
    for ng in ngrams:
        buf.append(ng)
        if len(buf) > window:
            buf.pop(0)
        unseen = sum(not bloom.contains(x) for x in buf)
        ratios.append(unseen / len(buf))
    return ratios


def to_state_strings(arr):
    return ["_".join(map(str, row)) for row in arr]


def make_ngrams(seq, n=N_GRAM_ORDER):
    return ["→".join(seq[i:i+n]) for i in range(len(seq) - n + 1)]


def normalize_and_quantize(x, mean, std):
    z = ((x - mean) / std).clip(-3, 3)
    q = ((z + 3) / 6 * QUANTIZATION_LEVELS).astype(int)
    return np.clip(q, 0, QUANTIZATION_LEVELS - 1)


def split_into_folds(x, k):
    size = len(x) // k
    return [x[i*size:(i+1)*size] for i in range(k)]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=== Adaptive N-gram Detection (GAN data) ===")

    # Load GAN data
    X_train = np.load(SYN_TRAIN)
    X_test  = np.load(SYN_TEST)

    print(f"Using {X_train.shape[1]} physical readings")

    # Normalization stats from synthetic TRAIN only
    mean = X_train.mean(axis=0)
    std  = np.std(X_train, axis=0)
    std[std == 0] = 1e-6

    # --------------------------------------------------------
    # Calibrate threshold τ
    # --------------------------------------------------------
    folds = split_into_folds(X_train, NUM_FOLDS)
    scores = []

    for i in range(1, NUM_FOLDS):
        train_part = np.vstack(folds[:i])
        val_part   = folds[i]

        tq = normalize_and_quantize(train_part, mean, std)
        vq = normalize_and_quantize(val_part, mean, std)

        ts = to_state_strings(tq)
        vs = to_state_strings(vq)

        tng = make_ngrams(ts)
        vng = make_ngrams(vs)

        bloom = BloomFilter(optimal_bloom_size(len(set(tng))), NUM_HASHES)
        for ng in tng:
            bloom.add(ng)

        raw = unseen_ngram_ratio(vng, bloom)
        scores.extend(exponential_smoothing(raw))

        print(f"[Fold {i}/{NUM_FOLDS}] Bloom utilization = {bloom.bit_array.mean():.3f}")

    tau = float(np.quantile(scores, 1 - TARGET_FPR))
    print(f"Calibrated τ = {tau:.4f}")

    # --------------------------------------------------------
    # Final training
    # --------------------------------------------------------
    tq = normalize_and_quantize(X_train, mean, std)
    ts = to_state_strings(tq)
    tng = make_ngrams(ts)

    bloom = BloomFilter(optimal_bloom_size(len(set(tng))), NUM_HASHES)
    for ng in tng:
        bloom.add(ng)

    print(f"Final Bloom utilization = {bloom.bit_array.mean():.3f}")

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------
    tq = normalize_and_quantize(X_test, mean, std)
    ts = to_state_strings(tq)
    tng = make_ngrams(ts)

    raw = unseen_ngram_ratio(tng, bloom)
    smooth = exponential_smoothing(raw)
    is_anomaly = [s > tau for s in smooth]

    anomaly_rate = np.mean(is_anomaly)
    print(f"Anomaly rate ≈ {anomaly_rate:.4f}")

    pd.DataFrame({
        "score": smooth,
        "is_anomaly": is_anomaly
    }).to_csv("results_task3_gan.csv", index=False)

    print("=== Done ===")
