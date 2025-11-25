#!/usr/bin/env python3
import os
from utils import load_and_clean_data
from exports import export_model_output
from glob import glob
import numpy as np
import pandas as pd
import argparse


# ==============================================================
# Load ONLY train1.csv + test1.csv (fast mode)
# ==============================================================
train_files = sorted(glob("../datasets/hai-22.04/train1.csv"))
test_files = sorted(glob("../datasets/hai-22.04/test1.csv"))


# ==============================================================
# Helper functions
# ==============================================================

def extract_attack_types(y: pd.Series):
    y_bin = (y.astype(int) != 0).astype(int)
    prev = y_bin.shift(1, fill_value=0)
    change = y_bin - prev

    start_idx = y_bin.index[change == 1].tolist()
    end_idx = y_bin.index[change == -1].tolist()

    if len(end_idx) < len(start_idx):
        end_idx.append(y_bin.index[-1])

    intervals = pd.DataFrame({
        "attack_id": range(1, len(start_idx) + 1),
        "start_index": start_idx,
        "end_index": end_idx
    })

    attack_type = pd.Series(0, index=y_bin.index, dtype=int)
    for _, row in intervals.iterrows():
        attack_type.loc[row.start_index:row.end_index] = row.attack_id

    return attack_type, intervals


def make_kfold_indices(n_samples, k=5, seed=42):
    np.random.seed(seed)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop

    return folds


# ==============================================================
# Scenario 1
# ==============================================================

def scenario_1_split(X, y, k=5, seed=42, balance_attacks=False):
    normal_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    folds = make_kfold_indices(len(normal_idx), k, seed)

    for fold_idx in range(k):
        test_normal_idx = normal_idx[folds[fold_idx]]
        train_normal_idx = np.setdiff1d(normal_idx, test_normal_idx)

        if balance_attacks:
            n_attack = len(test_normal_idx)
            attack_sample_idx = np.random.choice(attack_idx, n_attack, replace=False)
        else:
            attack_sample_idx = attack_idx

        test_idx = np.concatenate([test_normal_idx, attack_sample_idx])
        train_idx = train_normal_idx

        yield fold_idx, train_idx, test_idx


# ==============================================================
# Scenario 2 — Only FIRST attack type is held out
# ==============================================================

def scenario_2_split(X, y, k=5, seed=42):
    attack_type, attack_intervals = extract_attack_types(y)
    attack_ids = attack_intervals["attack_id"].unique()
    held_out = attack_ids[0]          # <-- Option A (first attack type)

    normal_idx = np.where(y == 0)[0]
    folds = make_kfold_indices(len(normal_idx), k, seed)

    for fold_idx in range(k):
        test_normal_idx = normal_idx[folds[fold_idx]]
        train_normal_idx = np.setdiff1d(normal_idx, test_normal_idx)

        train_attack_idx = np.where((attack_type != 0) & (attack_type != held_out))[0]
        test_attack_idx = np.where(attack_type != 0)[0]

        train_idx = np.concatenate([train_normal_idx, train_attack_idx])
        test_idx = np.concatenate([test_normal_idx, test_attack_idx])

        yield fold_idx, held_out, train_idx, test_idx


# ==============================================================
# Scenario 3 — Only FIRST attack type is trained on
# ==============================================================

def scenario_3_split(X, y, k=5, seed=42):
    attack_type, attack_intervals = extract_attack_types(y)
    attack_ids = attack_intervals["attack_id"].unique()
    selected = attack_ids[0]          # <-- Option A (first attack type)

    normal_idx = np.where(y == 0)[0]
    folds = make_kfold_indices(len(normal_idx), k, seed)

    for fold_idx in range(k):
        test_normal_idx = normal_idx[folds[fold_idx]]
        train_normal_idx = np.setdiff1d(normal_idx, test_normal_idx)

        train_attack_idx = np.where(attack_type == selected)[0]
        test_attack_idx = np.where(attack_type != 0)[0]

        train_idx = np.concatenate([train_normal_idx, train_attack_idx])
        test_idx = np.concatenate([test_normal_idx, test_attack_idx])

        yield fold_idx, selected, train_idx, test_idx


# ==============================================================
# Import ML models
# ==============================================================
from models import (
    run_OneClassSVM,
    run_LOF,
    run_EllipticEnvelope,
    run_knn,
    run_binary_svm,
    run_random_forest
)


# ==============================================================
# Internal runner
# ==============================================================

def _internal_main(args):
    model = getattr(args, 'model', getattr(args, 'm', None))
    scenario = getattr(args, 'scenario', getattr(args, 'sc', None))
    k_folds = getattr(args, 'k_fold', getattr(args, 'k', 5))
    export_choice = getattr(args, 'e', '1')

    if model is None:
        print("Error: model is required.")
        return

    one_class = ['ocsvm', 'lof', 'ee']
    supervised = ['knn', 'svm', 'rf']

    if model in one_class:
        scenario = '1'
    elif model in supervised:
        if scenario not in ['2', '3']:
            print("Error: knn/svm/rf require scenario 2 or 3.")
            return

    print("\n=== Loading & Cleaning Data ===")
    merged = load_and_clean_data(train_files, test_files, attack_cols=None)

    X = merged.drop(columns=['Attack', 'timestamp'], errors='ignore')
    y = merged['Attack']

    # Choose scenario function
    if scenario == '1':
        scenario_fn = scenario_1_split
    elif scenario == '2':
        scenario_fn = scenario_2_split
    elif scenario == '3':
        scenario_fn = scenario_3_split

    # Choose output folder
    out_dir = f"exports/Scenario{scenario}/{model.upper()}"
    os.makedirs(out_dir, exist_ok=True)

    # Run model
    if model == 'ocsvm':
        results = run_OneClassSVM(X, y, scenario_fn)
    elif model == 'lof':
        results = run_LOF(X, y, scenario_fn)
    elif model == 'ee':
        results = run_EllipticEnvelope(X, y, scenario_fn)
    elif model == 'svm':
        results = run_binary_svm(X, y, scenario_fn)
    elif model == 'knn':
        results = run_knn(X, y, scenario_fn=scenario_fn)
    elif model == 'rf':
        results = run_random_forest(X, y, scenario_fn)

    print(f"\n=== Running {model.upper()} on Scenario {scenario} ===")

    for entry in results:
        if scenario == '1':
            fold_idx, test_idx, y_pred, y_test = entry
        else:
            fold_idx, atk_id, test_idx, y_pred, y_test = entry

        out_file = f"{out_dir}/Predictions_Fold{fold_idx + 1}.csv"
        export_model_output(merged, test_idx, y_pred, out_file)


# ==============================================================
# Main CLI
# ==============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=['ocsvm', 'lof', 'ee', 'knn', 'svm', 'rf'], required=True)
    parser.add_argument('-sc', '--scenario', choices=['1', '2', '3'])
    parser.add_argument('-k', '--k-fold', type=int, default=5)
    parser.add_argument('-e', choices=['1', '2', '3'], default='1')
    args = parser.parse_args()

    _internal_main(args)


if __name__ == "__main__":
    main()
