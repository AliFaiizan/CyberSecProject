#!/usr/bin/env python3
import argparse, os, time, psutil
import numpy as np
import pandas as pd
from joblib import dump

from utils import load_data
from scenarios import scenario_1_split, scenario_2_split, scenario_3_split

from models import (
    run_OneClassSVM,
    run_EllipticEnvelope,
    run_LOF,
    run_binary_svm,
    run_knn,
    run_random_forest
)

from task2_cnn_latent import run_cnn_latent

process = psutil.Process()


# =====================================================================
# Save trained model
# =====================================================================
def save_model(model, scenario_id, model_name, fold_idx):
    out_dir = f"saved_models/Scenario{scenario_id}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{model_name}_Fold{fold_idx+1}.joblib"
    dump(model, path)
    print("Saved model:", path)


# =====================================================================
# Window → Row Mapping (CNN only)
# =====================================================================
def map_windows_to_rows(window_preds, N, M):
    row_votes = np.zeros(N)
    row_counts = np.zeros(N)

    for w in range(len(window_preds)):
        start, end = w, w + M
        row_votes[start:end] += window_preds[w]
        row_counts[start:end] += 1

    row_counts[row_counts == 0] = 1
    row_preds = (row_votes / row_counts) >= 0.5
    return row_preds.astype(int)


# =====================================================================
# ML Training + Prediction Saving
# =====================================================================
def run_and_save(model_name, run_fn, X_df, y_series, scenario_fn, k, scenario_id, out_base):

    model_dir = os.path.join(out_base, model_name)
    os.makedirs(model_dir, exist_ok=True)
    rows = []

    print(f"\nRunning {model_name} for Scenario {scenario_id}")

    for res in run_fn(X_df, y_series, k, scenario_fn):

        if scenario_id == 1:
            fold_idx, test_idx, y_pred, y_true, model, fe_rt, fe_mem, clf_rt, clf_mem = res
            attack_id = None
        else:
            fold_idx, attack_id, test_idx, y_pred, y_true, model, fe_rt, fe_mem, clf_rt, clf_mem = res

        # Precision / Recall
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)

        # Save predictions
        pd.DataFrame({
            "predicted_label": y_pred,
            "Attack": y_true
        }).to_csv(f"{model_dir}/Predictions_Fold{fold_idx+1}.csv", index=False)

        save_model(model, scenario_id, model_name, fold_idx)

        rows.append({
            "fold": fold_idx+1,
            "attack_id": attack_id,
            "precision": precision,
            "recall": recall,
            "feature_runtime_sec": fe_rt,
            "feature_memory_bytes": fe_mem,
            "runtime_sec": clf_rt,
            "memory_bytes": clf_mem
        })

        print(f"Fold {fold_idx+1}: precision={precision:.4f}, recall={recall:.4f}")

    pd.DataFrame(rows).to_csv(f"{model_dir}/metrics_summary.csv", index=False)


# =====================================================================
# MAIN
# =====================================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-sc", "--scenario", required=True, type=int, choices=[1,2,3])
    parser.add_argument("--latent-file", required=True)
    parser.add_argument("-k", "--folds", type=int, default=5)
    args = parser.parse_args()

    sc = args.scenario
    k = args.folds
    M = 20  # CNN window size

    # ---------------------------------------------------------
    # Load labels
    # ---------------------------------------------------------
    print("Loading labels...")
    _, y = load_data(
        ["../datasets/hai-22.04/train1.csv"],
        ["../datasets/hai-22.04/test1.csv"]
    )
    y_series = pd.Series(y).astype(int)

    # ---------------------------------------------------------
    # Load latent features
    # ---------------------------------------------------------
    print("Loading latent features...")
    Z = np.load(args.latent_file)

    if Z.shape[0] != len(y_series):
        print(f"[INFO] Latent rows = {Z.shape[0]}, Label rows = {len(y_series)}")
        print("[INFO] Trimming labels to match latent features...")
        y_series = y_series.iloc[:Z.shape[0]]

    X_df = pd.DataFrame(Z)

    # ---------------------------------------------------------
    # Select models
    # ---------------------------------------------------------
    if sc == 1:
        scenario_fn = scenario_1_split
        model_map = {
            "OCSVM": run_OneClassSVM,
            "LOF": run_LOF,
            "EllipticEnvelope": run_EllipticEnvelope
        }
    else:
        scenario_fn = scenario_2_split if sc == 2 else scenario_3_split
        model_map = {
            "SVM": run_binary_svm,
            "kNN": run_knn,
            "RandomForest": run_random_forest
        }

    out_base = f"exports/Scenario{sc}"
    os.makedirs(out_base, exist_ok=True)

    # ---------------------------------------------------------
    # Train ML models
    # ---------------------------------------------------------
    for name, fn in model_map.items():
        run_and_save(name, fn, X_df, y_series, scenario_fn, k, sc, out_base)

    # ---------------------------------------------------------
    # CNN for Scenario 2 & 3
    # ---------------------------------------------------------
    if sc in [2, 3]:

        cnn_dir = f"{out_base}/CNN"
        os.makedirs(cnn_dir, exist_ok=True)

        print("\n[CNN] Starting CNN training on latent features...")

        cnn_results = run_cnn_latent(Z, y_series.values, sc, k, cnn_dir, M=M)

        metrics = []

        for (fold_idx, model, Xw, yw, total_runtime, total_memory, test_idx, detail_dict) in cnn_results:

            # CNN window-based predictions
            window_preds = np.argmax(model.predict(Xw, verbose=0), axis=1)

            # Convert windows → rows
            N_test = len(test_idx)
            row_preds = map_windows_to_rows(window_preds, N_test, M)
            y_true = y_series.values[test_idx]

            # Save predictions
            pd.DataFrame({
                "predicted_label": row_preds,
                "Attack": y_true
            }).to_csv(f"{cnn_dir}/Predictions_Fold{fold_idx+1}.csv", index=False)

            # Compute metrics
            tp = ((row_preds == 1) & (y_true == 1)).sum()
            fp = ((row_preds == 1) & (y_true == 0)).sum()
            fn = ((row_preds == 0) & (y_true == 1)).sum()

            precision = tp / (tp + fp + 1e-9)
            recall    = tp / (tp + fn + 1e-9)

            metrics.append({
                "fold": fold_idx+1,
                "precision": precision,
                "recall": recall,
                "feature_runtime_sec": detail_dict["window_time"] + detail_dict["norm_time"],
                "feature_memory_bytes": detail_dict["window_mem"],
                "runtime_sec": total_runtime,
                "memory_bytes": total_memory
            })

        pd.DataFrame(metrics).to_csv(f"{cnn_dir}/metrics_summary.csv", index=False)


if __name__ == "__main__":
    main()
