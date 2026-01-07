#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scenario 1 – N-gram Anomaly Detection (k-fold CV)
-------------------------------------------------
• Raw physical readings
• Scenario 1 split (normal-only training)
• N ∈ {2,5,8}
• k = 5 folds
• Reports precision & recall
"""

import math
import hashlib
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import precision_score, recall_score
from scenarios import scenario_1_split   # PROVIDED FILE


# =========================
# CONFIG
# =========================

N_GRAMS = [2, 5, 8]
K_FOLDS = 5
Q_LEVELS = 20
BLOOM_BITS = 100_000
NUM_HASHES = 5

ATTACK_COL = "Attack"
DROP_COLS = ["timestamp", "Attack"]


# =========================
# BLOOM FILTER
# =========================
class BloomFilter:
    def __init__(self, size, k):
        self.size = size
        self.k = k
        self.bits = np.zeros(size, dtype=bool)

    def _hashes(self, s):
        for i in range(self.k):
            yield int(hashlib.sha1((str(i)+s).encode()).hexdigest(), 16) % self.size

    def add(self, s):
        for h in self._hashes(s):
            self.bits[h] = True

    def contains(self, s):
        return all(self.bits[h] for h in self._hashes(s))


# =========================
# HELPERS
# =========================
def normalize_and_quantize(df, mean, std, Q):
    z = ((df - mean) / std).clip(-3, 3).fillna(0)
    q = ((z + 3) / 6 * Q).astype(int).clip(0, Q-1)
    return q.values

def build_states(arr):
    return ["_".join(map(str, row)) for row in arr]

def make_ngrams(states, n):
    return ["→".join(states[i:i+n]) for i in range(len(states)-n+1)]

def anomaly_scores(test_ngrams, bloom, n):
    scores = []
    for i in range(len(test_ngrams)):
        window = test_ngrams[max(0, i-n+1):i+1]
        unseen = sum(not bloom.contains(g) for g in window)
        scores.append(unseen / len(window))
    return np.array(scores)


# =========================
# LOAD DATA
# =========================
# train_df = pd.read_csv(TRAIN_FILE)
# test_df  = pd.read_csv(TEST_FILE)

# X = pd.concat([train_df, test_df], ignore_index=True)
# y = X[ATTACK_COL]
# X = X.drop(columns=DROP_COLS).ffill().fillna(0)
train_data = np.load("synthetic_train.npy")  # shape: [N_train, F]
test_data = np.load("synthetic_test.npy")    # shape: [N_test, F]
test_labels = np.load("synthetic_test_labels.npy")  # shape: [N_test,] or [N_test, 1]

# Ensure test_labels is a column vector
if test_labels.ndim == 1:
    test_labels = test_labels[:, None]

# Add label column to train (all zeros)
train_labels = np.zeros((train_data.shape[0], 1))
train_data_with_label = np.hstack([train_data, train_labels])
test_data_with_label = np.hstack([test_data, test_labels])

# Combine
all_data = np.vstack([train_data_with_label, test_data_with_label])

# Now, features and labels:
X = pd.DataFrame(all_data[:, :-1])  # all columns except last as DataFrame
y = pd.Series(all_data[:, -1])   # last column as pandas Series
print(f"Using {X.shape[1]} physical readings")


# =========================
# MAIN LOOP
# =========================
for N in N_GRAMS:
    print(f"\n===== N = {N} =====")
    fold_precisions, fold_recalls = [], []

    for fold, train_idx, test_idx in scenario_1_split(X, y, k=K_FOLDS):
        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_test  = y[test_idx]

        # Normalize using TRAIN ONLY
        mean, std = X_train.mean(), X_train.std().replace(0, 1e-6)

        Xtr_q = normalize_and_quantize(X_train, mean, std, Q_LEVELS)
        Xte_q = normalize_and_quantize(X_test,  mean, std, Q_LEVELS)

        train_states = build_states(Xtr_q)
        test_states  = build_states(Xte_q)

        train_ngrams = make_ngrams(train_states, N)
        test_ngrams  = make_ngrams(test_states, N)

        bloom = BloomFilter(BLOOM_BITS, NUM_HASHES)
        for g in train_ngrams:
            bloom.add(g)

        scores = anomaly_scores(test_ngrams, bloom, N)
        tau = np.quantile(scores, 0.95)   # Scenario-1 threshold
        preds = scores > tau

        y_eval = y_test[N-1:]  # align lengths

        fold_precisions.append(precision_score(y_eval, preds, zero_division=0))
        fold_recalls.append(recall_score(y_eval, preds, zero_division=0))

        print(f"Fold {fold}: P={fold_precisions[-1]:.4f}, R={fold_recalls[-1]:.4f}")

    print(f"\nAVG (N={N}) → Precision={np.mean(fold_precisions):.4f}, "
          f"Recall={np.mean(fold_recalls):.4f}")
