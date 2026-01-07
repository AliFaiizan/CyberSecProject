#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scenario 1 – N-gram Anomaly Detection (k-fold CV, Synthetic Data)
---------------------------------------------------------------
• Raw physical readings
• Scenario 1 split (normal-only training)
• N ∈ {2,5,8}
• k = 5 folds
• Reports precision & recall
• Optional hyperparameter tuning (disabled by default)
"""

import hashlib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from scenarios import scenario_1_split


# ==============================================================
# BASELINE CONFIGURATION (DO NOT CHANGE)
# ==============================================================

N_GRAMS = [2, 5, 8]
K_FOLDS = 5

Q_LEVELS = 20
BLOOM_BITS = 100_000
NUM_HASHES = 5

ATTACK_COL = "Attack"


# ==============================================================
# OPTIONAL TUNING CONFIGURATION
# ==============================================================

ENABLE_TUNING = False  # default OFF to preserve baseline results

TUNING_GRID = {
    "Q_LEVELS": [20, 30],
    "BLOOM_BITS": [100_000, 200_000],
    "NUM_HASHES": [3, 5],
    "TAU_Q": [0.90, 0.95, 0.97, 0.99],
}


# ==============================================================
# BLOOM FILTER
# ==============================================================

class BloomFilter:
    def __init__(self, size, k):
        self.size = size
        self.k = k
        self.bits = np.zeros(size, dtype=bool)

    def _hashes(self, s):
        for i in range(self.k):
            yield int(hashlib.sha1((str(i) + s).encode()).hexdigest(), 16) % self.size

    def add(self, s):
        for h in self._hashes(s):
            self.bits[h] = True

    def contains(self, s):
        return all(self.bits[h] for h in self._hashes(s))


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def normalize_and_quantize(df, mean, std, Q):
    z = ((df - mean) / std).clip(-3, 3).fillna(0)
    q = ((z + 3) / 6 * Q).astype(int).clip(0, Q - 1)
    return q.values

def build_states(arr):
    return ["_".join(map(str, row)) for row in arr]

def make_ngrams(states, n):
    return ["→".join(states[i:i + n]) for i in range(len(states) - n + 1)]

def anomaly_scores(test_ngrams, bloom, n):
    scores = []
    for i in range(len(test_ngrams)):
        window = test_ngrams[max(0, i - n + 1):i + 1]
        unseen = sum(not bloom.contains(g) for g in window)
        scores.append(unseen / len(window))
    return np.array(scores)


# ==============================================================
# LOAD SYNTHETIC DATA
# ==============================================================

train_data = np.load("synthetic_train.npy")
test_data = np.load("synthetic_test.npy")
test_labels = np.load("synthetic_test_labels.npy")

if test_labels.ndim == 1:
    test_labels = test_labels[:, None]

train_labels = np.zeros((train_data.shape[0], 1))

train_data = np.hstack([train_data, train_labels])
test_data = np.hstack([test_data, test_labels])

all_data = np.vstack([train_data, test_data])

X = pd.DataFrame(all_data[:, :-1])
y = pd.Series(all_data[:, -1])

print(f"Using {X.shape[1]} physical readings")


# ==============================================================
# MAIN LOOP
# ==============================================================

for N in N_GRAMS:
    print(f"\n===== N = {N} =====")

    fold_precisions = []
    fold_recalls = []

    for fold, train_idx, test_idx in scenario_1_split(X, y, k=K_FOLDS):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        mean = X_train.mean()
        std = X_train.std().replace(0, 1e-6)

        Xtr_q = normalize_and_quantize(X_train, mean, std, Q_LEVELS)
        Xte_q = normalize_and_quantize(X_test, mean, std, Q_LEVELS)

        train_states = build_states(Xtr_q)
        test_states = build_states(Xte_q)

        train_ngrams = make_ngrams(train_states, N)
        test_ngrams = make_ngrams(test_states, N)

        best_f1 = -1
        best_tau = None

        if not ENABLE_TUNING:
            bloom = BloomFilter(BLOOM_BITS, NUM_HASHES)
            for g in train_ngrams:
                bloom.add(g)

            scores = anomaly_scores(test_ngrams, bloom, N)
            y_eval = y_test[N - 1:]

            for q in [0.90, 0.95, 0.97, 0.99]:
                tau = np.quantile(scores, q)
                preds = scores > tau
                f1 = f1_score(y_eval, preds, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_tau = tau

        else:
            for Q in TUNING_GRID["Q_LEVELS"]:
                for B in TUNING_GRID["BLOOM_BITS"]:
                    for H in TUNING_GRID["NUM_HASHES"]:

                        bloom = BloomFilter(B, H)
                        for g in train_ngrams:
                            bloom.add(g)

                        scores = anomaly_scores(test_ngrams, bloom, N)
                        y_eval = y_test[N - 1:]

                        for q in TUNING_GRID["TAU_Q"]:
                            tau = np.quantile(scores, q)
                            preds = scores > tau
                            f1 = f1_score(y_eval, preds, zero_division=0)

                            if f1 > best_f1:
                                best_f1 = f1
                                best_tau = tau

        preds = scores > best_tau

        out_dir = "exports/Scenario1/NGRAM"
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame({
            "Attack": y_eval.astype(int),
            "predicted_label": preds.astype(int)
        }).to_csv(
            f"{out_dir}/Predictions_N{N}_Fold{fold + 1}.csv",
            index=False
        )

        fold_precisions.append(precision_score(y_eval, preds, zero_division=0))
        fold_recalls.append(recall_score(y_eval, preds, zero_division=0))

        print(f"Fold {fold}: P={fold_precisions[-1]:.4f}, R={fold_recalls[-1]:.4f}")

    print(
        f"\nAVG (N={N}) → Precision={np.mean(fold_precisions):.4f}, "
        f"Recall={np.mean(fold_recalls):.4f}"
    )
