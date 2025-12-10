#!/usr/bin/env python3
"""
Task 2 – Complete Experiment Pipeline (Tasksheet 3)

Final Version – Option A Fix:
- Normalizes data (prevents NaNs)
- Trains VAE per fold
- Extracts FULL latent Z for ML models (fixes .iloc index errors)
- ML models operate on Z_full_df
- CNN operates on Z_train / Z_test windows
- Supports Scenarios 1 / 2 / 3 / all
"""

import os
import time
import argparse
import psutil
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score

import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------- Project Imports ----------------
from utils import load_data
from scenarios import scenario_1_split, scenario_2_split, scenario_3_split

from task1 import (
    VAE,
    train_vae_reconstruction,
    train_vae_classification,
    extract_latent_features
)

from models import (
    run_OneClassSVM,
    run_LOF,
    run_EllipticEnvelope,
    run_binary_svm,
    run_knn,
    run_random_forest
)

from task2_ensemble import ensemble
from task_cnn2 import run_cnn_latent


process = psutil.Process()


# ---------------------------------------------------------
# Measure runtime + memory
# ---------------------------------------------------------
def measure(func, *args, **kwargs):
    mem_before = process.memory_info().rss
    t0 = time.time()
    result = func(*args, **kwargs)
    dt = time.time() - t0
    mem_after = process.memory_info().rss
    dmem = mem_after - mem_before
    return result, dt, dmem


# ---------------------------------------------------------
# Train VAE + extract FULL latent matrix (Option A Fix)
# ---------------------------------------------------------
def train_and_extract(
        X, y, train_idx, test_idx,
        mode="reconstruction",
        latent_dim=8,
        activation="relu",
        layer_type="dense",
        epochs=10,
        device="cpu"
):
    X_train = X[train_idx]

    # Build VAE
    model = VAE(
        input_dim=X.shape[1],
        latent_dim=latent_dim,
        activation=activation,
        layer_type=layer_type,
        num_classes=None if mode == "reconstruction" else 2
    )

    # ---------------- TRAIN VAE ----------------
    if mode == "reconstruction":
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train).float()),
            batch_size=128,
            shuffle=True
        )
        train_vae_reconstruction(model, train_loader, None,
                                 device=device, epochs=epochs)

    else:
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y[train_idx]).long()
            ),
            batch_size=128,
            shuffle=True
        )
        train_vae_classification(model, train_loader, None,
                                 device=device, epochs=epochs)

    # ------------- FULL LATENT EXTRACTION -------------
    Z_full, t_full, m_full, _ = extract_latent_features(model, X, device=device)

    # Slices for CNN only
    Z_train = Z_full[train_idx]
    Z_test = Z_full[test_idx]

    return Z_full, Z_train, Z_test, t_full, m_full


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def run_all_experiments(train_csv, test_csv, out_dir, scenario):

    print("\n=== LOADING DATA ===")
    X, y = load_data([train_csv], [test_csv])
    y_series = pd.Series(y)

    # 🔥 IMPORTANT FIX – normalize ICS data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    os.makedirs(out_dir, exist_ok=True)
    results = []

    # ============================================================
    # SCENARIO 1 — One-Class Models
    # ============================================================
    if scenario in ["1", "all"]:
        print("\n================ SCENARIO 1 ================")

        for fold_idx, train_idx, test_idx in scenario_1_split(X, y_series):

            print(f"\n--- Scenario 1 | Fold {fold_idx+1} ---")

            Z_full, Z_train, Z_test, fe_time, fe_mem = train_and_extract(
                X, y, train_idx, test_idx, mode="reconstruction"
            )

            Z_full_df = pd.DataFrame(Z_full)

            fake_scn = lambda X_, Y_, k=None: [(fold_idx, train_idx, test_idx)]

            for name, func in [
                ("OCSVM", run_OneClassSVM),
                ("LOF", run_LOF),
                ("EllipticEnvelope", run_EllipticEnvelope)
            ]:

                ml_res, ml_time, ml_mem = measure(
                    func, Z_full_df, y_series, 1, fake_scn
                )

                (_, test_ids, y_pred, y_true, *_rest) = ml_res[0]

                results.append({
                    "scenario": 1,
                    "fold": fold_idx,
                    "classifier": name,
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "feature_time": fe_time,
                    "feature_mem": fe_mem,
                    "ml_time": ml_time,
                    "ml_mem": ml_mem
                })

    # ============================================================
    # SCENARIO 2 — Binary ML + CNN + Ensembles
    # ============================================================
    if scenario in ["2", "all"]:
        print("\n================ SCENARIO 2 ================")

        for fold_idx, attack_id, train_idx, test_idx in scenario_2_split(X, y_series):

            print(f"\n--- Scenario 2 | Fold {fold_idx+1} Attack {attack_id} ---")

            Z_full, Z_train, Z_test, fe_time, fe_mem = train_and_extract(
                X, y, train_idx, test_idx, mode="classification"
            )

            Z_full_df = pd.DataFrame(Z_full)

            fake_scn = lambda X_, Y_, k=None: [(fold_idx, attack_id, train_idx, test_idx)]

            # ---------- ML MODELS ----------
            svm_res, svm_t, svm_m = measure(run_binary_svm, Z_full_df, y_series, 1, fake_scn)
            knn_res, knn_t, knn_m = measure(run_knn,        Z_full_df, y_series, 1, fake_scn)
            rf_res,  rf_t,  rf_m  = measure(run_random_forest, Z_full_df, y_series, 1, fake_scn)

            (_, _, _, svm_pred, y_test, *_r) = svm_res[0]
            (_, _, _, knn_pred, _ , *_r)     = knn_res[0]
            (_, _, _, rf_pred,  _ , *_r)     = rf_res[0]

            # Save ML results
            for name, pred, t, m in [
                ("SVM", svm_pred, svm_t, svm_m),
                ("KNN", knn_pred, knn_t, knn_m),
                ("RF",  rf_pred,  rf_t, rf_m)
            ]:
                results.append({
                    "scenario": 2,
                    "fold": fold_idx,
                    "classifier": name,
                    "precision": precision_score(y_test, pred, zero_division=0),
                    "recall":    recall_score(y_test, pred, zero_division=0),
                    "feature_time": fe_time,
                    "feature_mem": fe_mem,
                    "ml_time": t,
                    "ml_mem": m
                })

            # ---------- ENSEMBLES ----------
            preds = [svm_pred, knn_pred, rf_pred]

            for cname, method in [
                ("Ensemble_Majority", "majority"),
                ("Ensemble_All", "all"),
                ("Ensemble_Random", "random")
            ]:
                ens_pred = ensemble(preds, method)

                results.append({
                    "scenario": 2,
                    "fold": fold_idx,
                    "classifier": cname,
                    "precision": precision_score(y_test, ens_pred, zero_division=0),
                    "recall":    recall_score(y_test, ens_pred, zero_division=0),
                    "feature_time": fe_time,
                    "feature_mem": fe_mem,
                    "ml_time": svm_t + knn_t + rf_t,
                    "ml_mem": svm_m + knn_m + rf_m
                })

            # ---------- CNN ----------
            cnn_res, cnn_t, cnn_m = measure(
                run_cnn_latent,
                Z_train, y[train_idx],
                Z_test,  y[test_idx],
                32, 1, 10,
                f"{out_dir}/scenario2_cnn_fold{fold_idx+1}.csv"
            )

            if cnn_res:
                results.append({
                    "scenario": 2,
                    "fold": fold_idx,
                    "classifier": "CNN",
                    "precision": cnn_res["precision"],
                    "recall":    cnn_res["recall"],
                    "feature_time": fe_time,
                    "feature_mem": fe_mem,
                    "ml_time": cnn_t,
                    "ml_mem": cnn_m
                })

    # ============================================================
    # SCENARIO 3 — Same as Scenario 2
    # ============================================================
    if scenario in ["3", "all"]:
        print("\n================ SCENARIO 3 ================")

        for fold_idx, attack_type, train_idx, test_idx in scenario_3_split(X, y_series):

            print(f"\n--- Scenario 3 | Fold {fold_idx+1} Type {attack_type} ---")

            Z_full, Z_train, Z_test, fe_time, fe_mem = train_and_extract(
                X, y, train_idx, test_idx, mode="classification"
            )

            Z_full_df = pd.DataFrame(Z_full)

            fake_scn = lambda X_, Y_, k=None: [(fold_idx, attack_type, train_idx, test_idx)]

            # ML models
            svm_res, svm_t, svm_m = measure(run_binary_svm, Z_full_df, y_series, 1, fake_scn)
            knn_res, knn_t, knn_m = measure(run_knn,        Z_full_df, y_series, 1, fake_scn)
            rf_res,  rf_t,  rf_m  = measure(run_random_forest, Z_full_df, y_series, 1, fake_scn)

            (_, _, _, svm_pred, y_test, *_r) = svm_res[0]
            (_, _, _, knn_pred, _ , *_r)     = knn_res[0]
            (_, _, _, rf_pred,  _ , *_r)     = rf_res[0]

            # Save ML results
            for name, pred, t, m in [
                ("SVM", svm_pred, svm_t, svm_m),
                ("KNN", knn_pred, knn_t, knn_m),
                ("RF",  rf_pred,  rf_t, rf_m)
            ]:
                results.append({
                    "scenario": 3,
                    "fold": fold_idx,
                    "classifier": name,
                    "precision": precision_score(y_test, pred, zero_division=0),
                    "recall":    recall_score(y_test, pred, zero_division=0),
                    "feature_time": fe_time,
                    "feature_mem": fe_mem,
                    "ml_time": t,
                    "ml_mem": m
                })

            # Ensembles
            preds = [svm_pred, knn_pred, rf_pred]

            for cname, method in [
                ("Ensemble_Majority", "majority"),
                ("Ensemble_All", "all"),
                ("Ensemble_Random", "random")
            ]:
                ens_pred = ensemble(preds, method)

                results.append({
                    "scenario": 3,
                    "fold": fold_idx,
                    "classifier": cname,
                    "precision": precision_score(y_test, ens_pred, zero_division=0),
                    "recall": recall_score(y_test, ens_pred, zero_division=0),
                    "feature_time": fe_time,
                    "feature_mem": fe_mem,
                    "ml_time": svm_t + knn_t + rf_t,
                    "ml_mem": svm_m + knn_m + rf_m
                })

            # CNN
            cnn_res, cnn_t, cnn_m = measure(
                run_cnn_latent,
                Z_train, y[train_idx],
                Z_test,  y[test_idx],
                32, 1, 10,
                f"{out_dir}/scenario3_cnn_fold{fold_idx+1}.csv"
            )

            if cnn_res:
                results.append({
                    "scenario": 3,
                    "fold": fold_idx,
                    "classifier": "CNN",
                    "precision": cnn_res["precision"],
                    "recall":    cnn_res["recall"],
                    "feature_time": fe_time,
                    "feature_mem": fe_mem,
                    "ml_time": cnn_t,
                    "ml_mem": cnn_m
                })

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/task2_results.csv", index=False)
    print("\n=== ALL RESULTS SAVED ===")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["1", "2", "3", "all"])
    parser.add_argument("--train", type=str, default="../../datasets/hai-22.04/train1.csv")
    parser.add_argument("--test",  type=str, default="../../datasets/hai-22.04/test1.csv")
    parser.add_argument("--out",   type=str, default="exports/task2")

    args = parser.parse_args()

    run_all_experiments(args.train, args.test, args.out, args.scenario)
