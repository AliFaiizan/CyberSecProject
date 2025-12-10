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
from task2_ensemble import ensemble, load_pred

process = psutil.Process()


def save_model(model, scenario_id, model_name, fold_idx):
    out_dir = f"saved_models/Scenario{scenario_id}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{model_name}_Fold{fold_idx+1}.joblib"
    dump(model, path)
    print("Saved model:", path)


# =====================================================================
# Runs ML model and saves timing + metrics
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

        # Metrics
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

def map_windows_to_rows(window_preds, N, M):
    """
    Convert window-level predictions back into row-level predictions.
    Uses averaging over overlapping windows.
    
    window_preds: array shape (N-M+1,)
    N: original number of samples
    M: window size
    """
    row_votes = np.zeros(N)
    row_counts = np.zeros(N)

    for w_idx in range(len(window_preds)):
        start = w_idx
        end = w_idx + M
        for row in range(start, end):
            row_votes[row] += window_preds[w_idx]
            row_counts[row] += 1

    # Avoid division by zero
    row_counts[row_counts == 0] = 1

    row_preds = row_votes / row_counts
    row_preds = (row_preds >= 0.5).astype(int)  # majority vote

    return row_preds

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

    print("Loading labels...")
    X_raw, y = load_data(
        ["../../datasets/hai-22.04/train1.csv"],
        ["../../datasets/hai-22.04/test1.csv"]
    )
    y_series = pd.Series(y).astype(int)

    print("Loading latent features...")
    Z = np.load(args.latent_file)
    if Z.shape[0] != len(y_series):
        raise ValueError("Latent dimension mismatch")

    X_df = pd.DataFrame(Z)

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

    # Run ML models
    for name, fn in model_map.items():
        run_and_save(name, fn, X_df, y_series, scenario_fn, k, sc, out_base)

    # CNN (Scenario 2 & 3)
    if sc in [2, 3]:
        cnn_dir = f"{out_base}/CNN"
        os.makedirs(cnn_dir, exist_ok=True)

        cnn_res = run_cnn_latent(Z, y_series.values, sc, k, cnn_dir)

        rows = []
        for fold_idx, model, Xw, yw, fe_rt, fe_mem, clf_rt, clf_mem in cnn_res:

            window_preds = np.argmax(model.predict(Xw, verbose=0), axis=1)

# Convert window predictions → row predictions
            N = len(y_series)   # full sequence length
            row_preds = map_windows_to_rows(window_preds, N, M=20)
            y_true_full = y_series.values

            pd.DataFrame({
                "predicted_label": row_preds,
                "Attack": y_true_full
            }).to_csv(f"{cnn_dir}/Predictions_Fold{fold_idx+1}.csv", index=False)


            tp = ((preds == 1) & (yw == 1)).sum()
            fp = ((preds == 1) & (yw == 0)).sum()
            fn = ((preds == 0) & (yw == 1)).sum()

            precision = tp / (tp + fp + 1e-9)
            recall    = tp / (tp + fn + 1e-9)

            rows.append({
                "fold": fold_idx+1,
                "precision": precision,
                "recall": recall,
                "feature_runtime_sec": fe_rt,
                "feature_memory_bytes": fe_mem,
                "runtime_sec": clf_rt,
                "memory_bytes": clf_mem
            })

        pd.DataFrame(rows).to_csv(f"{cnn_dir}/metrics_summary.csv", index=False)

    print("Task 2 complete.")


if __name__ == "__main__":
    main()
