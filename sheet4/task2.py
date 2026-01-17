#!/usr/bin/env python3
"""
Task Sheet 4 - Task 2: Run Classifiers on Pre-Generated Folds
FIXED: Added feature normalization and better hyperparameters

This script loads pre-generated fold data and runs classifiers.
"""

import argparse
import os
import time
import psutil
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.svm import OneClassSVM, SVC
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

process = psutil.Process()


# =========================================================
# SAVE MODEL
# =========================================================
def save_model(model, scenario_id, model_name, fold_idx):
    out_dir = f"saved_models_sheet4/Scenario{scenario_id}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{model_name}_Fold{fold_idx+1}.joblib"
    dump(model, path)


# =========================================================
# PER-FOLD CLASSIFIERS WITH NORMALIZATION
# =========================================================
def run_OneClassSVM_per_fold(Z_train, y_train, Z_test, y_test):
    """Train OCSVM with feature normalization"""
    
    # CRITICAL FIX: Normalize features to handle distribution mismatch
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    # Better hyperparameters
    model = OneClassSVM(kernel="rbf", nu=0.05, gamma='auto')
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm)
    mem_after = process.memory_info().rss
    
    y_pred_raw = model.predict(Z_test_norm)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def run_LOF_per_fold(Z_train, y_train, Z_test, y_test):
    """Train LOF with feature normalization"""
    
    # Normalize features
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    # Better hyperparameters
    model = LocalOutlierFactor(n_neighbors=20, metric='euclidean', novelty=True)
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm)
    mem_after = process.memory_info().rss
    
    y_pred_raw = model.predict(Z_test_norm)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def run_EllipticEnvelope_per_fold(Z_train, y_train, Z_test, y_test):
    """Train EllipticEnvelope with feature normalization"""
    
    # Normalize features
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    # Better hyperparameters
    model = EllipticEnvelope(contamination=0.01, random_state=42)
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm)
    mem_after = process.memory_info().rss
    
    y_pred_raw = model.predict(Z_test_norm)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def run_binary_svm_per_fold(Z_train, y_train, Z_test, y_test):
    """Train SVM with feature normalization"""
    
    # Normalize features
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    model = SVC(kernel="rbf", C=10.0, gamma='scale')
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm, y_train)
    mem_after = process.memory_info().rss
    
    y_pred = model.predict(Z_test_norm)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def run_knn_per_fold(Z_train, y_train, Z_test, y_test):
    """Train kNN with feature normalization"""
    
    # Normalize features
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    model = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm, y_train)
    mem_after = process.memory_info().rss
    
    y_pred = model.predict(Z_test_norm)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def run_random_forest_per_fold(Z_train, y_train, Z_test, y_test):
    """Train RandomForest with feature normalization"""
    
    # Normalize features
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=5, 
                                   random_state=42, n_jobs=-1)
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm, y_train)
    mem_after = process.memory_info().rss
    
    y_pred = model.predict(Z_test_norm)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


# =========================================================
# RUN AND SAVE RESULTS
# =========================================================
def run_and_save_per_fold(model_name, run_fn, fold_data, k, scenario_id, out_base):
    """Run classifier on pre-split fold data"""
    
    model_dir = os.path.join(out_base, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Running {model_name} for Scenario {scenario_id}")
    print(f"{'='*70}")
    
    rows = []
    
    for fold_idx in range(k):
        print(f"\n  Fold {fold_idx + 1}/{k}:")
        
        # Load fold data
        Z_train = fold_data[fold_idx]['Z_train']
        y_train = fold_data[fold_idx]['y_train']
        Z_test = fold_data[fold_idx]['Z_test']
        y_test = fold_data[fold_idx]['y_test']
        
        # Skip empty folds
        if Z_train.size == 0 or Z_test.size == 0:
            print(f"    ⚠ Skipping empty fold")
            continue
        
        # Run model
        y_pred, model, clf_time, clf_mem = run_fn(Z_train, y_train, Z_test, y_test)
        
        # Compute metrics
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        
        # Save predictions
        pd.DataFrame({
            "predicted_label": y_pred,
            "Attack": y_test
        }).to_csv(f"{model_dir}/Predictions_Fold{fold_idx+1}.csv", index=False)
        
        # Save model
        save_model(model, scenario_id, model_name, fold_idx)
        
        rows.append({
            "fold": fold_idx + 1,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "runtime_sec": clf_time,
            "memory_bytes": clf_mem
        })
        
        print(f"    precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
        print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    
    # Save summary
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(f"{model_dir}/metrics_summary.csv", index=False)
    print(f"\n  ✓ Saved metrics summary to {model_dir}/metrics_summary.csv")
    
    # Print overall stats
    print(f"\n  {model_name} Summary (across all folds):")
    print(f"    Avg Precision: {summary_df['precision'].mean():.4f}")
    print(f"    Avg Recall:    {summary_df['recall'].mean():.4f}")
    print(f"    Avg F1-Score:  {summary_df['f1_score'].mean():.4f}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n" + "="*70)
    print("TASK SHEET 4 - TASK 2: RUN CLASSIFIERS ON PRE-GENERATED FOLDS")
    print("WITH FEATURE NORMALIZATION FIX")
    print("="*70)
    
    parser = argparse.ArgumentParser(description='Task Sheet 4 - Task 2: Run classifiers')
    parser.add_argument('-sc', '--scenario', type=int, required=True, 
                        choices=[1, 2, 3], help='Scenario number')
    parser.add_argument('-k', '--folds', type=int, default=5,
                        help='Number of folds')
    args = parser.parse_args()
    
    sc = args.scenario
    k = args.folds
    
    # Load fold data
    fold_data_dir = f"exports_sheet4/Scenario{sc}"
    
    print("\n" + "="*70)
    print(f"LOADING SAVED FOLD DATA FOR SCENARIO {sc}")
    print("="*70)
    
    if not os.path.exists(fold_data_dir):
        print(f"\n❌ ERROR: Fold data directory not found: {fold_data_dir}")
        print("\nPlease generate fold data first using:")
        print(f"  python generate_folds_FIXED.py -sc {sc} -k {k} -M 20 --vae-checkpoint vae_XXX_real.pt")
        return
    
    fold_data = {}
    for fold_idx in range(k):
        fold_dir = f"{fold_data_dir}/fold{fold_idx}"
        
        try:
            Z_train = np.load(f"{fold_dir}/train_latent.npy")
            y_train = np.load(f"{fold_dir}/train_labels.npy")
            Z_test = np.load(f"{fold_dir}/test_latent.npy")
            y_test = np.load(f"{fold_dir}/test_labels.npy")
            
            fold_data[fold_idx] = {
                'Z_train': Z_train.astype(np.float32),
                'y_train': y_train.astype(np.int32),
                'Z_test': Z_test.astype(np.float32),
                'y_test': y_test.astype(np.int32)
            }
            
            print(f"  Fold {fold_idx + 1}: train {Z_train.shape}, test {Z_test.shape}")
        except Exception as e:
            print(f"  ⚠ Error loading fold {fold_idx}: {e}")
            continue
    
    print("\n✓ All fold data loaded")
    
    # Run classifiers
    out_base = f"exports_sheet4/Scenario{sc}"
    os.makedirs(out_base, exist_ok=True)
    
    if sc == 1:
        print("\n" + "="*70)
        print("RUNNING ANOMALY DETECTION CLASSIFIERS (Scenario 1)")
        print("WITH FEATURE NORMALIZATION")
        print("="*70)
        
        run_and_save_per_fold("OCSVM", run_OneClassSVM_per_fold, fold_data, k, sc, out_base)
        run_and_save_per_fold("LOF", run_LOF_per_fold, fold_data, k, sc, out_base)
        run_and_save_per_fold("EllipticEnvelope", run_EllipticEnvelope_per_fold, fold_data, k, sc, out_base)
    else:
        print("\n" + "="*70)
        print(f"RUNNING BINARY CLASSIFIERS (Scenario {sc})")
        print("WITH FEATURE NORMALIZATION")
        print("="*70)
        
        run_and_save_per_fold("SVM", run_binary_svm_per_fold, fold_data, k, sc, out_base)
        run_and_save_per_fold("kNN", run_knn_per_fold, fold_data, k, sc, out_base)
        run_and_save_per_fold("RandomForest", run_random_forest_per_fold, fold_data, k, sc, out_base)
    
    # Done
    print(f"\n{'='*70}")
    print(f"SCENARIO {sc} COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Results saved to: {out_base}/")
    print(f"Models saved to: saved_models_sheet4/Scenario{sc}/")
    print()


if __name__ == "__main__":
    main()