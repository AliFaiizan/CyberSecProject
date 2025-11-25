#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 3 – Adaptive N-gram Anomaly Detection
------------------------------------------
• Uses all training CSVs (normal only) for model building
• Applies global normalization across train + test
• Builds a 2-gram Bloom filter with EWMA smoothing
• Evaluates all test CSVs merged as a continuous stream
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

DATA_DIR = "/home/safiamed/Study_Project/datasets/hai-21.03/"
TRAIN_PATTERN = os.path.join(DATA_DIR, "train*.csv")
TEST_PATTERN  = os.path.join(DATA_DIR, "test*.csv")

ATTACK_COLUMNS = ['attack', 'attack_P1', 'attack_P2', 'attack_P3']

SELECTED_SENSORS = [
    "P1_FT01Z", "P1_PIT01", "P1_TIT01",
    "P2_SIT01", "P3_FIT01", "P4_ST_PT01", "P4_ST_TT01"
]

# Model parameters
QUANTIZATION_LEVELS = 6
N_GRAM_ORDER = 2
SLIDING_WINDOW = 80
EWMA_LAMBDA = 0.1
TARGET_FPR = 0.05
NUM_HASHES = 4
NUM_FOLDS = 10


# ============================================================
# BLOOM FILTER IMPLEMENTATION
# ============================================================

class BloomFilter:
    """Simple Bloom filter for n-gram storage and lookup."""

    def __init__(self, size=1_000_000, num_hashes=3):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = np.zeros(size, dtype=bool)

    def _hashes(self, value):
        """Generate multiple hash values for a string input."""
        for i in range(self.num_hashes):
            combined = f"{i}-{value}"
            yield int(hashlib.sha1(combined.encode()).hexdigest(), 16) % self.size

    def add(self, value):
        for position in self._hashes(value):
            self.bit_array[position] = True

    def contains(self, value):
        return all(self.bit_array[pos] for pos in self._hashes(value))


def optimal_bloom_size(n_items, false_positive_rate=0.01):
    """Compute optimal Bloom filter size for n items."""
    size = -n_items * math.log(false_positive_rate) / (math.log(2) ** 2)
    return max(1_000_000, int(size))


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def exponential_smoothing(values, lam=EWMA_LAMBDA):
    """Apply EWMA smoothing to reduce noise in anomaly scores."""
    smoothed = []
    last_value = values[0]
    for v in values:
        last_value = lam * v + (1 - lam) * last_value
        smoothed.append(last_value)
    return smoothed


def unseen_ngram_ratio(ngrams, bloom_filter, window_size=SLIDING_WINDOW):
    """Compute ratio of unseen n-grams within a moving window."""
    window, ratios = [], []

    for ng in ngrams:
        window.append(ng)
        if len(window) > window_size:
            window.pop(0)

        unseen_count = sum(not bloom_filter.contains(g) for g in window)
        ratios.append(unseen_count / len(window))

    return ratios


def to_state_strings(array_2d):
    """Convert numeric sensor readings to string states."""
    return ["_".join(map(str, row)) for row in array_2d]


def make_ngrams(sequence, n=N_GRAM_ORDER):
    """Generate overlapping n-grams from a list of states."""
    return ["→".join(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def load_dataset(train_pattern, test_pattern):
    """Load and merge all training and test CSV files."""
    train_df = pd.concat([pd.read_csv(f) for f in sorted(glob(train_pattern))], ignore_index=True)
    test_df  = pd.concat([pd.read_csv(f) for f in sorted(glob(test_pattern))], ignore_index=True)

    # Keep only normal data for training
    if 'attack' in train_df.columns:
        train_df = train_df[train_df['attack'] == 0]

    drop_cols = ['time'] + [c for c in ATTACK_COLUMNS if c in train_df.columns]
    train_df = train_df.drop(columns=drop_cols, errors='ignore').ffill().fillna(0)
    test_df  = test_df.drop(columns=drop_cols, errors='ignore').ffill().fillna(0)

    return train_df, test_df


def normalize_and_quantize(df, mean, std, q_levels=QUANTIZATION_LEVELS):
    """Apply Z-score normalization and quantize values to discrete bins."""
    zscores = ((df - mean) / std).clip(-3, 3).replace([np.inf, -np.inf], 0).fillna(0)
    quantized = ((zscores + 3) / 6 * q_levels).astype(int).clip(0, q_levels - 1)
    return quantized


# ============================================================
# CALIBRATION
# ============================================================

def split_into_folds(df, n_folds=NUM_FOLDS):
    """Split dataset into equal time-based folds."""
    fold_size = len(df) // n_folds
    return [df.iloc[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]


def calibrate_threshold(train_df, mean, std):
    """Estimate anomaly threshold (τ) from validation folds."""
    folds = split_into_folds(train_df, NUM_FOLDS)
    calibration_scores = []

    for i in range(1, NUM_FOLDS):
        # Split into training and validation parts
        train_part = pd.concat(folds[:i])
        val_part = folds[i]

        # Normalize and quantize
        train_quant = normalize_and_quantize(train_part, mean, std)
        val_quant   = normalize_and_quantize(val_part, mean, std)

        # Convert to state sequences
        train_states = to_state_strings(train_quant.values)
        val_states   = to_state_strings(val_quant.values)

        # Generate n-grams
        train_ngrams = make_ngrams(train_states)
        val_ngrams   = make_ngrams(val_states)

        # Build Bloom filter
        bloom_size = optimal_bloom_size(len(set(train_ngrams)))
        bloom = BloomFilter(bloom_size, NUM_HASHES)
        for ng in train_ngrams:
            bloom.add(ng)

        # Compute smoothed unseen ratios
        raw_scores = unseen_ngram_ratio(val_ngrams, bloom)
        smoothed_scores = exponential_smoothing(raw_scores)
        calibration_scores.extend(smoothed_scores)

        print(f"[Fold {i}/{NUM_FOLDS}] Bloom utilization = {bloom.bit_array.mean():.3f}")

    tau = float(np.quantile(calibration_scores, 1 - TARGET_FPR))
    print(f"Calibrated τ = {tau:.4f} (based on {len(calibration_scores)} samples)")
    return tau


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=== HAI 21.03 – Adaptive N-gram Anomaly Detection ===")

    # Load and prepare data
    train_df, test_df = load_dataset(TRAIN_PATTERN, TEST_PATTERN)
    train_df = train_df[SELECTED_SENSORS]
    test_df  = test_df[SELECTED_SENSORS]
    print(f"Using {len(SELECTED_SENSORS)} sensors: {SELECTED_SENSORS}")

    # Compute global normalization parameters
    mean, std = train_df.mean(), train_df.std().replace(0, 1e-6)

    # Step 1 – Calibrate detection threshold
    tau = calibrate_threshold(train_df, mean, std)

    # Step 2 – Train final Bloom filter on full training data
    quantized_train = normalize_and_quantize(train_df, mean, std)
    train_states = to_state_strings(quantized_train.values)
    train_ngrams = make_ngrams(train_states)

    bloom_size = optimal_bloom_size(len(set(train_ngrams)))
    bloom = BloomFilter(bloom_size, NUM_HASHES)
    for ng in train_ngrams:
        bloom.add(ng)
    print(f"Final Bloom utilization = {bloom.bit_array.mean():.3f}")

    # Step 3 – Evaluate on merged test data
    quantized_test = normalize_and_quantize(test_df, mean, std)
    test_states = to_state_strings(quantized_test.values)
    test_ngrams = make_ngrams(test_states)

    raw_scores = unseen_ngram_ratio(test_ngrams, bloom)
    smoothed_scores = exponential_smoothing(raw_scores)

    is_anomaly = [s > tau for s in smoothed_scores]
    anomaly_rate = np.mean(is_anomaly)

    # Save results
    output = pd.DataFrame({
        "score": smoothed_scores,
        "is_anomaly": is_anomaly
    })
    output.to_csv("results_task3_combined.csv", index=False)

    print(f"\n=== Done ===")
    print(f"All test files merged | {len(is_anomaly)} rows | Anomaly rate ≈ {anomaly_rate:.4f}")
