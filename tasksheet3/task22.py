#!/usr/bin/env python3
import argparse, os, time, psutil
import numpy as np
import pandas as pd

# Load utils & scenarios
from utils import load_data
from scenarios import (
    scenario_1_split,
    scenario_2_split,
    scenario_3_split,
)

# ML models
from models import (
    run_OneClassSVM,
    run_EllipticEnvelope,
    run_LOF,
    run_binary_svm,
    run_knn,
    run_random_forest,
)

# CNN for latent VAE features
from task2_cnn_latent import run_cnn_latent

# Ensemble
from task2_ensemble import ensemble, load_pred

process = psutil.Process()


# -------------------------------------------------------------------
# Helper to run ML models & save predictions + metrics
# -------------------------------------------------------------------
def run_and_save(model_name, run_fn, X_df, y_series, scenario_fn, k, scenario_id, out_base):

    os.makedirs(out_base, exist_ok=True)
    model_dir = os.path.join(out_base, model_name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n[+] Running {model_name} on Scenario {scenario_id}...")

    mem0 = process.memory_info().rss
    t0 = time.time()

    # call ML model → returns a list of results from all folds
    results = run_fn(X_df, y_series, k, scenario_fn)

    t1 = time.time()
    mem1 = process.memory_info().rss

    runtime = t1 - t0
    mem_used = mem1 - mem0

    rows = []

    for item in results:

        # Scenario 1 format:
        #   fold_idx, test_idx, y_pred, y_true
        # Scenario 2 & 3 format:
        #   fold_idx, attack_id, test_idx, y_pred, y_true

        if scenario_id == 1:
            fold_idx, test_idx, y_pred, y_true = item
            attack_id = None
        else:
            fold_idx, attack_id, test_idx, y_pred, y_true = item

        # save predictions
        out_file = f"{model_dir}/Predictions_Fold{fold_idx+1}.csv"
        pd.DataFrame({
            "predicted_label": y_pred,
            "Attack": y_true
        }).to_csv(out_file, index=False)

        # compute metrics
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)

        rows.append({
            "fold": fold_idx + 1,
            "attack_id": attack_id,
            "precision": precision,
            "recall": recall,
            "runtime_sec": runtime,
            "memory_bytes": mem_used,
        })

        print(f"  Fold {fold_idx+1}: precision={precision:.4f}, recall={recall:.4f}")

    pd.DataFrame(rows).to_csv(f"{model_dir}/metrics_summary.csv", index=False)
    print(f"[{model_name}] runtime={runtime:.2f}s | memory={mem_used/1e6:.2f} MB")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(description="Task 2 – ML, CNN, Ensemble on VAE latent features")
    parser.add_argument("-sc", "--scenario", required=True, type=int, choices=[1,2,3])
    parser.add_argument("--latent-file", required=True, help="Path to VAE latent features (.npy)")
    parser.add_argument("-k", "--folds", type=int, default=5)
    args = parser.parse_args()

    sc = args.scenario
    latent_path = args.latent_file
    k = args.folds

    print("\n=== Loading Raw Data (labels only) ===")

    X_raw, y = load_data(
        ["../../datasets/hai-22.04/train1.csv"],
        ["../../datasets/hai-22.04/test1.csv"]
    )
    y_series = pd.Series(y).astype(int)

    print("=== Loading Latent Features ===")
    Z = np.load(latent_path)
    if Z.shape[0] != len(y_series):
        raise ValueError("Latent rows do not match Y labels!")

    X_df = pd.DataFrame(Z)

    # ---------------------------------------------------
    # Select scenario
    # ---------------------------------------------------
    if sc == 1:
        scenario_fn = scenario_1_split
        models = {
            "OCSVM": run_OneClassSVM,
            "LOF": run_LOF,
            "EllipticEnvelope": run_EllipticEnvelope,
        }

    elif sc == 2:
        scenario_fn = scenario_2_split
        models = {
            "SVM": run_binary_svm,
            "kNN": run_knn,
            "RandomForest": run_random_forest,
        }

    else:
        scenario_fn = scenario_3_split
        models = {
            "SVM": run_binary_svm,
            "kNN": run_knn,
            "RandomForest": run_random_forest,
        }

    out_base = f"exports/Scenario{sc}"

    # ---------------------------------------------------
    # Run ML models
    # ---------------------------------------------------
    for name, fn in models.items():
        run_and_save(name, fn, X_df, y_series, scenario_fn, k, sc, out_base)

    # ---------------------------------------------------
    # Run CNN for Scenario 2 & 3 only
    # ---------------------------------------------------
    if sc in [2,3]:
        print("\n[+] Running CNN (6-block architecture) on latent features...")
        cnn_dir = f"{out_base}/CNN"
        os.makedirs(cnn_dir, exist_ok=True)
        run_cnn_latent(Z, y_series.values, scenario_fn, k, cnn_dir)

    # ---------------------------------------------------
    # Ensemble stage
    # ---------------------------------------------------
    print("\n[+] Running Ensemble...")

    if sc == 1:
        model_list = ["OCSVM", "LOF", "EllipticEnvelope"]
    else:
        model_list = ["RandomForest", "kNN", "SVM"]

    ensemble_dir = f"{out_base}/Ensemble"
    os.makedirs(ensemble_dir, exist_ok=True)

    for fold in range(1, k+1):

        preds_list = []
        true_y = None

        for m in model_list:
            p, y_true = load_pred(f"{out_base}/{m}/Predictions_Fold{fold}.csv")
            preds_list.append(p)
            true_y = y_true

        # 3 ensemble methods required
        for method in ["random", "majority", "all"]:
            final = ensemble(preds_list, method)
            out_file = f"{ensemble_dir}/Ensemble_{method}_Fold{fold}.csv"

            pd.DataFrame({
                "predicted_label": final,
                "Attack": true_y
            }).to_csv(out_file, index=False)

            print(f"[Ensemble] Fold {fold} | Method={method} saved.")


if __name__ == "__main__":
    main()
