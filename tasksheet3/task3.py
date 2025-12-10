#!/usr/bin/env python3
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from scenarios import scenario_1_split, scenario_2_split, scenario_3_split


# ---------------------------------------------------------
# Auto-select correct latent file based on scenario
# ---------------------------------------------------------
def select_latent_file():
    vae_dir = "vae_features"
    files = os.listdir(vae_dir)

    recon = [f for f in files if "reconstruction" in f]
    cls   = [f for f in files if "classification" in f]

    if not recon:
        raise FileNotFoundError("No reconstruction latent file found.")
    if not cls:
        raise FileNotFoundError("No classification latent file found.")

    return {
        1: os.path.join(vae_dir, recon[0]),
        2: os.path.join(vae_dir, cls[0]),
        3: os.path.join(vae_dir, cls[0])
    }


# ---------------------------------------------------------
# Plot helper
# ---------------------------------------------------------
def plot_importance(importances, feature_names, title, out_file):
    idx = np.argsort(importances)[::-1]
    sorted_imp = importances[idx]
    sorted_names = feature_names[idx]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(sorted_imp)), sorted_imp)
    plt.xticks(range(len(sorted_imp)), sorted_names, rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


# ---------------------------------------------------------
# Compute importance (FAST + CORRECT)
# ---------------------------------------------------------
def compute_importance(model, X_test, y_test, scenario):

    # -------- Subsample for slow models (LOF, kNN) --------
    if len(X_test) > 4000:
        idx = np.random.choice(len(X_test), 4000, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]

    # -------- Built-in importance (RandomForest) --------
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_

    # -------- Linear SVM --------
    if hasattr(model, "coef_"):
        return np.abs(model.coef_).flatten()

    # -------- Choose scoring based on scenario --------
    if scenario == 1:
        scoring_metric = "accuracy"   # one-class → cannot use F1
    else:
        scoring_metric = "f1"         # binary classifiers

    # -------- Permutation importance --------
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scoring_metric,
        n_repeats=3,
        random_state=42
    )

    return perm.importances_mean

# ---------------------------------------------------------
# Main Task 3(a)
# ---------------------------------------------------------
def run_task3a(scenario):
    latent_files = select_latent_file()
    latent_file = latent_files[scenario]

    print(f"[INFO] Using latent file: {latent_file}")
    Z = np.load(latent_file)
    feature_names = np.array([f"z{i}" for i in range(Z.shape[1])])

    # -------- Load full dataset labels --------
    from utils import load_data
    X_raw, y_raw = load_data(
        ["../../datasets/hai-22.04/train1.csv"],
        ["../../datasets/hai-22.04/test1.csv"]
    )
    y_raw = np.array(y_raw)

    # -------- Select models per scenario --------
    if scenario == 1:
        scenario_fn = scenario_1_split
        models = ["OCSVM", "LOF", "EllipticEnvelope"]
    else:
        scenario_fn = scenario_2_split if scenario == 2 else scenario_3_split
        models = ["SVM", "kNN", "RandomForest"]

    out_dir = f"feature_importance/Scenario{scenario}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Processing Scenario {scenario}")

    # -------- Iterate k-fold splits --------
    for fold_data in scenario_fn(pd.DataFrame(Z), pd.Series(y_raw), k=5):

        if scenario == 1:
            fold_idx, train_idx, test_idx = fold_data
        else:
            fold_idx, _, train_idx, test_idx = fold_data

        X_test = Z[test_idx]
        y_test = y_raw[test_idx]

        # -------- Evaluate each model --------
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
