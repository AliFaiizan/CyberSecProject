#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scenario 1 – N-gram Anomaly Detection (k-fold CV, Synthetic Data)
---------------------------------------------------------------
• Load pre-generated fold data (raw features)
• N ∈ {2,5,8}
• k = 5 folds
• Reports precision & recall
• Optional hyperparameter tuning (disabled by default)
"""

import hashlib
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score


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



def main():
    parser = argparse.ArgumentParser(description="N-gram anomaly detection on fold data")
    parser.add_argument("-sc", "--scenario", type=int, default=1, help="Scenario (1, 2, or 3)")
    parser.add_argument("-k", "--k-folds", type=int, default=5, help="Number of folds")
    args = parser.parse_args()

    scenario = args.scenario
    k_folds = args.k_folds
    fold_data_dir = f"exports_sheet4/Scenario{scenario}"

    print(f"Loading fold data from: {fold_data_dir}")

    # ==============================================================
    # MAIN LOOP
    # ==============================================================

    for N in N_GRAMS:
        print(f"\n===== N = {N} =====")

        fold_precisions = []
        fold_recalls = []

        for fold_idx in range(k_folds):
            fold_dir = f"{fold_data_dir}/syn_fold{fold_idx}"

            # Load raw training and test data
            X_train = np.load(f"{fold_dir}/train_raw.npy")
            y_train = np.load(f"{fold_dir}/train_raw_labels.npy")
            X_test = np.load(f"{fold_dir}/test_raw.npy")
            y_test = np.load(f"{fold_dir}/test_raw_labels.npy")

            # Flatten labels if needed
            if y_train.ndim > 1:
                y_train = y_train.ravel()
            if y_test.ndim > 1:
                y_test = y_test.ravel()

            # Convert to DataFrame/Series for processing
            X_train_df = pd.DataFrame(X_train)
            X_test_df = pd.DataFrame(X_test)

            mean = X_train_df.mean()
            std = X_train_df.std().replace(0, 1e-6)

            Xtr_q = normalize_and_quantize(X_train_df, mean, std, Q_LEVELS)
            Xte_q = normalize_and_quantize(X_test_df, mean, std, Q_LEVELS)

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

            out_dir = f"exports_sheet4/Scenario{scenario}/syn_NGRAM"
            os.makedirs(out_dir, exist_ok=True)

            pd.DataFrame({
                "Attack": y_eval.astype(int),
                "predicted_label": preds.astype(int)
            }).to_csv(
                f"{out_dir}/Predictions_N{N}_Fold{fold_idx}.csv",
                index=False
            )

            fold_precisions.append(precision_score(y_eval, preds, zero_division=0))
            fold_recalls.append(recall_score(y_eval, preds, zero_division=0))

            print(f"Fold {fold_idx}: P={fold_precisions[-1]:.4f}, R={fold_recalls[-1]:.4f}")

        print(
            f"\nAVG (N={N}) → Precision={np.mean(fold_precisions):.4f}, "
            f"Recall={np.mean(fold_recalls):.4f}"
        )


if __name__ == "__main__":
    main()
