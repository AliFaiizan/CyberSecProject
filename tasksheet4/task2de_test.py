#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scenario 1 – N-gram Anomaly Detection (k-fold CV) + Precision/Recall
--------------------------------------------------------------------
• Raw physical readings (selected features)
• Scenario 1: train on NORMAL ONLY
• N ∈ {2,5,8}
• k = 5 folds (normal-only folds)
• Bloom-filter n-grams
• Threshold picked on TRAIN-NORMAL calibration (target FPR) to avoid leakage
• Reports precision & recall (per fold + average)
"""

import hashlib
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import precision_score, recall_score
from scenarios import scenario_1_split   # PROVIDED FILE


# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "../datasets/hai-22.04/"
TRAIN_FILE = DATA_DIR + "train1.csv"
TEST_FILE  = DATA_DIR + "test1.csv"

ATTACK_COL = "Attack"
DROP_COLS = ["timestamp", "Attack"]

# same “style” params you use
N_GRAMS = [2, 5, 8]
K_FOLDS = 5
Q_LEVELS = 20
BLOOM_BITS = 100_000
NUM_HASHES = 5

# training-only calibration (Scenario-1 compliant)
CALIB_FRAC   = 0.2     # last 20% of training-normal used for calibration
TARGET_FPR   = 0.05    # 2% of calibration normals flagged
SCORE_WINDOW = 50     # sliding window for unseen-rate smoothing (try 50 or 100)


# ============================================================
# BLOOM FILTER
# ============================================================
class BloomFilter:
    def __init__(self, size=100_000, k=5):
        self.size = size
        self.k = k
        self.bits = np.zeros(size, dtype=bool)

    def _hashes(self, s: str):
        for i in range(self.k):
            yield int(hashlib.sha1((str(i) + s).encode()).hexdigest(), 16) % self.size

    def add(self, s: str):
        for h in self._hashes(s):
            self.bits[h] = True

    def contains(self, s: str) -> bool:
        return all(self.bits[h] for h in self._hashes(s))


# ============================================================
# HELPERS
# ============================================================
def normalize_and_quantize(df: pd.DataFrame, mean: pd.Series, std: pd.Series, Q: int):
    z = ((df - mean) / std).clip(-3, 3).fillna(0)
    q = ((z + 3) / 6 * Q).astype(int).clip(0, Q - 1)
    return q.values

def build_state_strings(qdata: np.ndarray):
    return ["_".join(map(str, row)) for row in qdata]

def generate_ngrams(states, n: int):
    return ["→".join(states[i:i+n]) for i in range(len(states) - n + 1)]

def anomaly_scores(ngrams, bloom: BloomFilter, window_size: int):
    """
    Score at position i = fraction of unseen n-grams in last `window_size` n-grams.
    Output length == len(ngrams).
    """
    scores = []
    for i in range(len(ngrams)):
        window = ngrams[max(0, i - window_size + 1): i + 1]
        unseen = sum(not bloom.contains(g) for g in window)
        scores.append(unseen / len(window))
    return np.asarray(scores)

def split_fit_calib_contiguous(n: int, calib_frac: float):
    """
    Contiguous split to preserve time order (important for n-grams).
    Fit = first (1-calib_frac), Calib = last calib_frac.
    """
    split = int(round(n * (1.0 - calib_frac)))
    split = min(max(split, 1), n - 1)
    fit_idx = np.arange(0, split)
    calib_idx = np.arange(split, n)
    return fit_idx, calib_idx

def pick_tau_by_target_fpr(calib_scores: np.ndarray, target_fpr: float):
    """
    Choose tau using ONLY normal calibration scores:
    tau is the (1-target_fpr) quantile.
    """
    q = 1.0 - float(target_fpr)
    q = min(max(q, 0.01), 0.99)
    tau = float(np.quantile(calib_scores, q))
    return tau, q


# ============================================================
# LOAD DATA (raw physical readings)
# ============================================================
def load_data():
    # If you want to keep your explicit feature list, put it here:
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

    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)

    # combine so scenario_1_split can work across full sequence if needed
    X_all = pd.concat([train_df, test_df], ignore_index=True)

    # labels (keep for evaluation)
    y_all = X_all[ATTACK_COL].astype(int)

    # drop non-features
    X_all = X_all.drop(columns=DROP_COLS, errors="ignore")

    # keep only selected features if they exist
    # (errors='ignore' lets your code run even if a column name differs slightly)
    X_all = X_all.reindex(columns=features)

    # fill
    X_all = X_all.ffill().fillna(0)

    print(f"Using {X_all.shape[1]} physical readings")
    return X_all, y_all


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    X_all, y_all = load_data()

    for N in N_GRAMS:
        print(f"\n===== N = {N} =====")
        fold_precisions, fold_recalls = [], []
        chosen_qs = []

        for fold, train_idx, test_idx in scenario_1_split(X_all, y_all, k=K_FOLDS):
            # IMPORTANT: preserve time order inside fold
            train_idx = np.sort(train_idx)
            test_idx  = np.sort(test_idx)

            X_train = X_all.iloc[train_idx]
            X_test  = X_all.iloc[test_idx]
            y_test  = y_all[test_idx]

            # Normalize using TRAIN ONLY (Scenario 1 train is normal-only)
            mean = X_train.mean()
            std  = X_train.std().replace(0, 1e-6)

            # Quantize
            Xtr_q = normalize_and_quantize(X_train, mean, std, Q_LEVELS)
            Xte_q = normalize_and_quantize(X_test,  mean, std, Q_LEVELS)

            # Build train states (normal-only)
            train_states = build_state_strings(Xtr_q)

            # training-only calibration (contiguous to preserve sequence)
            fit_i, calib_i = split_fit_calib_contiguous(len(train_states), CALIB_FRAC)
            fit_states   = [train_states[i] for i in fit_i]
            calib_states = [train_states[i] for i in calib_i]

            # Bloom fit on fit-normal
            bloom = BloomFilter(size=BLOOM_BITS, k=NUM_HASHES)
            fit_ngrams = generate_ngrams(fit_states, N)
            for g in fit_ngrams:
                bloom.add(g)

            # Calibrate tau on calibration-normal scores
            calib_ngrams = generate_ngrams(calib_states, N)
            calib_scores = anomaly_scores(calib_ngrams, bloom, SCORE_WINDOW)
            tau, tau_q = pick_tau_by_target_fpr(calib_scores, TARGET_FPR)
            chosen_qs.append(tau_q)

            # Test scores
            test_states = build_state_strings(Xte_q)
            test_ngrams = generate_ngrams(test_states, N)
            test_scores = anomaly_scores(test_ngrams, bloom, SCORE_WINDOW)

            preds = (test_scores > tau).astype(int)

            # Align labels: preds length = len(test_states) - N + 1
            y_eval = y_test[N - 1:]

            p = precision_score(y_eval, preds, zero_division=0)
            r = recall_score(y_eval, preds, zero_division=0)
            fold_precisions.append(p)
            fold_recalls.append(r)

            print(f"Fold {fold}: tau_q={tau_q:.2f}, alarms={preds.mean():.3f}, "
                  f"P={p:.4f}, R={r:.4f}")

        print(f"\nAVG (N={N}) → Precision={np.mean(fold_precisions):.4f}, "
              f"Recall={np.mean(fold_recalls):.4f}")
        print(f"Chosen tau quantiles per fold: {chosen_qs} (avg={np.mean(chosen_qs):.3f})")
