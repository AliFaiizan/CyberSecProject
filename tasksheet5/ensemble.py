#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import argparse

# ------------------------------------------------------------
# Load prediction file
# ------------------------------------------------------------
def load_prediction(path):
    df = pd.read_csv(path)
    return df["predicted_label"].values, df["Attack"].values

# ------------------------------------------------------------
# Align predictions to same length
# ------------------------------------------------------------
def align(preds_list):
    min_len = min(len(x) for x in preds_list)
    return [x[:min_len] for x in preds_list], min_len

# ------------------------------------------------------------
# Ensemble methods
# ------------------------------------------------------------
def ensemble(preds_list, method):
    preds = np.vstack(preds_list).T  # shape: (samples, models)

    if method == "majority":
        return (np.mean(preds, axis=1) >= 0.5).astype(int)

    elif method == "all":
        return np.all(preds == 1, axis=1).astype(int)

    elif method == "random":
        out = []
        for row in preds:
            out.append(np.random.choice(row))
        return np.array(out, dtype=int)

    else:
        raise ValueError("Unknown ensemble method: " + method)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, required=True, choices=[1,2,3])
    parser.add_argument("--base", default="exports_sheet4")
    parser.add_argument("-k", "--folds", type=int, default=5)
    args = parser.parse_args()

    scenario = args.scenario
    base = args.base
    k = args.folds

    print(f"=== BUILDING ENSEMBLE FOR SCENARIO {scenario} ===")

    # ----------------------------------------
    # Select models per scenario
    # ----------------------------------------
    if scenario == 1:
        models = ["OCSVM", "LOF", "EllipticEnvelope"]
    else:
        models = ["SVM", "kNN", "RandomForest"]

    scenario_dir = f"{base}/Scenario{scenario}"
    ensemble_dir = f"{scenario_dir}/Ensemble"
    os.makedirs(ensemble_dir, exist_ok=True)

    METHODS = ["majority", "all", "random"]

    # ----------------------------------------
    # For each fold
    # ----------------------------------------
    for fold in range(1, k+1):

        preds_list = []
        y_true_ref = None

        # Load each model prediction
        for model in models:
            pred_path = f"{scenario_dir}/{model}/Predictions_Fold{fold}.csv"

            if not os.path.exists(pred_path):
                print(f"[WARNING] Missing: {pred_path}")
                continue

            pred, y_true = load_prediction(pred_path)
            preds_list.append(pred.astype(int))

            if y_true_ref is None:
                y_true_ref = y_true

        # Align lengths
        preds_list, min_len = align(preds_list)
        y_true_ref = y_true_ref[:min_len]

        # Run each ensemble strategy
        for method in METHODS:
            ens_pred = ensemble(preds_list, method)

            out_path = f"{ensemble_dir}/Predictions_{method}_Fold{fold}.csv"

            pd.DataFrame({
                "predicted_label": ens_pred,
                "Attack": y_true_ref
            }).to_csv(out_path, index=False)

            print(f"[Ensemble] Fold {fold} → {method} saved.")

    print("=== DONE ===")

if __name__ == "__main__":
    main()
