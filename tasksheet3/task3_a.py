#!/usr/bin/env python3
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from scenarios import scenario_1_split, scenario_2_split, scenario_3_split


# ---------------------------------------------------------
# Auto-select correct latent file
# ---------------------------------------------------------
def select_latent_file():
    vae_dir = "vae_features"
    files = os.listdir(vae_dir)

    recon = [f for f in files if "reconstruction" in f]
    cls   = [f for f in files if "classification" in f]

    return {
        1: os.path.join(vae_dir, recon[0]),
        2: os.path.join(vae_dir, cls[0]),
        3: os.path.join(vae_dir, cls[0]),
    }


# ---------------------------------------------------------
# Plot helper
# ---------------------------------------------------------
def plot_importance(importances, feature_names, title, out_file):
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(range(len(importances)), feature_names[idx], rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


# ---------------------------------------------------------
# Compute importance
# ---------------------------------------------------------
def compute_importance(model, X_test, y_test, scenario):

    if hasattr(model, "feature_importances_"):
        return model.feature_importances_

    if hasattr(model, "coef_"):
        return np.abs(model.coef_).flatten()

    scoring = "accuracy" if scenario == 1 else "f1"

    perm = permutation_importance(
        model, X_test, y_test,
        scoring=scoring, n_repeats=5, random_state=42
    )
    return perm.importances_mean


# ---------------------------------------------------------
# Task 3(a)
# ---------------------------------------------------------
def run_task3a(scenario):

    # Load latent features
    latent_files = select_latent_file()
    Z = np.load(latent_files[scenario])
    feature_names = np.array([f"z{i}" for i in range(Z.shape[1])])

    # Load labels
    from utils import load_data
    _, y = load_data(
        ["../datasets/hai-22.04/train1.csv"],
        ["../datasets/hai-22.04/test1.csv"]
    )
    y = np.array(y)

    # Trim labels exactly like Task 2
    if len(y) != len(Z):
        print(f"[INFO] Adjusting labels: Z={len(Z)}, y={len(y)} → trimming")
        y = y[:len(Z)]

    # Select scenario
    if scenario == 1:
        scenario_fn = scenario_1_split
        models = ["OCSVM", "LOF", "EllipticEnvelope"]
    else:
        scenario_fn = scenario_2_split if scenario == 2 else scenario_3_split
        models = ["SVM", "kNN", "RandomForest"]

    out_dir = f"feature_importance/Scenario{scenario}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Running Scenario {scenario}")

    # Iterate folds
    for split in scenario_fn(pd.DataFrame(Z), pd.Series(y), k=5):

        if scenario == 1:
            fold_idx, train_idx, test_idx = split
        else:
            fold_idx, _, train_idx, test_idx = split

        X_test = Z[test_idx]
        y_test = y[test_idx]

        # For each model
        for model_name in models:

            model_path = f"saved_models/Scenario{scenario}/{model_name}_Fold{fold_idx+1}.joblib"

            if not os.path.exists(model_path):
                print("[WARN] Missing model:", model_path)
                continue

            print(f"[INFO] Fold {fold_idx+1} → {model_name}")
            model = joblib.load(model_path)

            importances = compute_importance(model, X_test, y_test, scenario)

            out_file = f"{out_dir}/{model_name}_Fold{fold_idx+1}_importance.png"
            title = f"{model_name} Feature Importance (Fold {fold_idx+1})"

            plot_importance(importances, feature_names, title, out_file)

            print("[SAVED]", out_file)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task 3(a) Feature Importance")
    parser.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    run_task3a(args.scenario)
