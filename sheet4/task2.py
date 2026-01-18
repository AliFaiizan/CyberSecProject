#!/usr/bin/env python3
"""
Task Sheet 4 - Task 2: Run Classifiers on Pre-Generated Folds


This script loads pre-generated fold data and runs classifiers.
"""

import argparse
import os
import time
import psutil
import numpy as np
import pandas as pd
from joblib import dump
from sklearn import svm
from sklearn.svm import OneClassSVM, SVC
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# CNN imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

process = psutil.Process()


# =========================================================
# SAVE MODEL
# =========================================================
def save_model(model, scenario_id, model_name, fold_idx):
    out_dir = f"saved_models_sheet4/Scenario{scenario_id}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{model_name}_Fold{fold_idx+1}.joblib"
    dump(model, path)

def save_cnn_model(model, scenario_id, fold_idx):
    out_dir = f"saved_models_sheet4/Scenario{scenario_id}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/CNN_Fold{fold_idx+1}.h5"
    model.save(path)

# =========================================================
# PER-FOLD CLASSIFIERS WITH NORMALIZATION
# =========================================================
def run_OneClassSVM_per_fold(Z_train, y_train, Z_test, y_test, params):
    """Train OCSVM with feature normalization"""
    
    # CRITICAL FIX: Normalize features to handle distribution mismatch
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    # Better hyperparameters
    model = OneClassSVM(kernel="rbf", nu=0.001, gamma='auto')
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm)
    mem_after = process.memory_info().rss
    
    y_pred_raw = model.predict(Z_test_norm)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def run_LOF_per_fold(Z_train, y_train, Z_test, y_test,params):
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


def run_EllipticEnvelope_per_fold(Z_train, y_train, Z_test, y_test,params):
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


def run_binary_svm_per_fold(Z_train, y_train, Z_test, y_test,params):
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


def run_knn_per_fold(Z_train, y_train, Z_test, y_test,params):
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


def run_random_forest_per_fold(Z_train, y_train, Z_test, y_test, params):
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

def build_cnn_model(input_shape, dropout_rate=0.3, lr=1e-3):
    """
    Build CNN with 6 blocks as per Task Sheet 4 requirements:
    - 4 Conv blocks (each with 2 conv layers + batch norm + activation + dropout)
    - 2 FC layers
    - Adam optimizer + categorical cross-entropy
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    # ===== BLOCK 1: Conv Block =====
    model.add(Conv1D(32, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    
    model.add(Conv1D(32, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    
    # ===== BLOCK 2: Conv Block =====
    model.add(Conv1D(64, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    
    model.add(Conv1D(64, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    
    # ===== BLOCK 3: Conv Block =====
    model.add(Conv1D(128, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    
    model.add(Conv1D(128, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    
    # ===== BLOCK 4: Conv Block =====
    model.add(Conv1D(256, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    
    model.add(Conv1D(256, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    
    # Flatten before FC layers
    model.add(Flatten())
    
    # ===== BLOCK 5: Fully Connected Layer 1 =====
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(dropout_rate))
    
    # ===== BLOCK 6: Fully Connected Layer 2 =====
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(2, activation="softmax"))
    
    # Compile with Adam optimizer and categorical cross-entropy
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def run_cnn_per_fold(Z_train, y_train, Z_test, y_test):
    """Train CNN on latent features"""
    
    start = time.time()
    mem_before = process.memory_info().rss
    
    # Normalize features
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    # Reshape for CNN: (samples, timesteps, features)
    # Treat each latent dimension as a timestep
    n_features = Z_train_norm.shape[1]
    Z_train_cnn = Z_train_norm.reshape(len(Z_train_norm), n_features, 1)
    Z_test_cnn = Z_test_norm.reshape(len(Z_test_norm), n_features, 1)
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, 2)
    y_test_cat = to_categorical(y_test, 2)
    
    # Calculate class weights for imbalance
    n_normal = np.sum(y_train == 0)
    n_attack = np.sum(y_train == 1)
    total = len(y_train)
    
    if n_attack > 0:
        class_weight = {
            0: total / (2 * n_normal),
            1: total / (2 * n_attack)
        }
    else:
        class_weight = None
    
    # Build CNN model
    input_shape = (n_features, 1)
    model = build_cnn_model(input_shape, dropout_rate=0.3, lr=1e-3)
    
    # Train with early stopping
    callbacks = [
        EarlyStopping(
            monitor="val_loss", 
            patience=5, 
            restore_best_weights=True
        )
    ]
    
    model.fit(
        Z_train_cnn, y_train_cat,
        validation_data=(Z_test_cnn, y_test_cat),
        epochs=30,
        batch_size=128,
        class_weight=class_weight,
        verbose=0,
        callbacks=callbacks
    )
    
    mem_after = process.memory_info().rss
    
    # Predict
    y_pred_proba = model.predict(Z_test_cnn, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


# =========================================================
# RUN AND SAVE RESULTS
# =========================================================
def run_and_save_per_fold(model_name, run_fn, fold_data, k, scenario_id, out_base, param_grid=None, is_cnn=False):
    """Run classifier on pre-split fold data"""
    from utils2 import optimal_param_search_with_folds
    model_dir = os.path.join(out_base, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Running {model_name} for Scenario {scenario_id}")
    print(f"{'='*70}")
    
    rows = []
    
    #her i want to check optimal params. runfn is the model runing function, the pass that best paras main run function
    best_params = None
    if param_grid is not None:
        print(f"\nPerforming hyperparameter search for {model_name}...")
        scenario_type = 'oneclass' if scenario_id == 1 else 'binary'
        best_params, results = optimal_param_search_with_folds(
            fold_data, run_fn, param_grid, scenario_type
        )
        print(f"\nBest params found: {best_params}")
    for fold_idx in range(k):
        print(f"\n  Fold {fold_idx + 1}/{k}:")
        
        # Load fold data
        Z_train = fold_data[fold_idx]['Z_train']
        y_train = fold_data[fold_idx]['y_train']
        Z_test = fold_data[fold_idx]['Z_test']
        y_test = fold_data[fold_idx]['y_test']
        
        # Skip empty folds
        if Z_train.size == 0 or Z_test.size == 0:
            print(f"    Skipping empty fold")
            continue
        
        # Run model
        y_pred, model, clf_time, clf_mem = run_fn(Z_train, y_train, Z_test, y_test,best_params)
        
        # Handle shape mismatch (if predictions are shorter than labels)
        if len(y_pred) != len(y_test):
            print(f"shape match")
            min_len = min(len(y_pred), len(y_test))
            y_pred = y_pred[:min_len]
            y_test = y_test[:min_len]
        
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
        if is_cnn:
            save_cnn_model(model, scenario_id, fold_idx)
        else:
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
        print(f"\nERROR: Fold data directory not found: {fold_data_dir}")
        print("\nPlease generate fold data first using:")
        print(f"  python generate_folds.py -sc {sc} -k {k} -M 20 --vae-checkpoint vae_XXX_real.pt")
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
        ocsvm_grid = {
            'nu': [0.001], # 0.01, 0.05 
            'gamma': ['scale'] # 0.1 0.01, 0.001 
        }
        run_and_save_per_fold("OCSVM", run_OneClassSVM_per_fold, fold_data, k, sc, out_base, ocsvm_grid)
        lof_grid = {
            'n_neighbors': [10],#, 20, 30, 50
            'metric': ['euclidean'] # 'manhattan'
        }
        run_and_save_per_fold("LOF", run_LOF_per_fold, fold_data, k, sc, out_base, lof_grid)

        elliptic_grid = {
            'contamination': [0.001],# 0.01, 0.05
            'support_fraction': [None]# 0.7, 0.9
        }
        run_and_save_per_fold("EllipticEnvelope", run_EllipticEnvelope_per_fold, fold_data, k, sc, out_base,elliptic_grid)
    else:
        print("\n" + "="*70)
        print(f"RUNNING BINARY CLASSIFIERS (Scenario {sc})")
        print("WITH FEATURE NORMALIZATION")
        print("="*70)
        
        svm_grid = {
            'C': [10.0],#0.1, 1, 
            'gamma': ['scale'], # 0.01, 0.001
        }
        run_and_save_per_fold("SVM", run_binary_svm_per_fold, fold_data, k, sc, out_base, svm_grid)
        knn_grid = {
            'n_neighbors': [3],  #, 5, 7, 11  
            'weights': ['uniform'], # 'distance'
            'metric': ['euclidean'] #  'manhattan'
        }
        run_and_save_per_fold("kNN", run_knn_per_fold, fold_data, k, sc, out_base, knn_grid)

        rf_grid = {
            'n_estimators': [50],#, 100, 200
            'max_depth': [5],#, 10, None
            'min_samples_split': [5] #2,
        }

        run_and_save_per_fold("RandomForest", run_random_forest_per_fold, fold_data, k, sc, out_base, rf_grid)

         # CNN for Scenarios 2 & 3
        print("\n" + "="*70)
        print(f"RUNNING CNN CLASSIFIER (Scenario {sc})")
        print("CNN Architecture: 6 blocks (4 conv + 2 FC)")
        print("="*70)
        run_and_save_per_fold("CNN", run_cnn_per_fold, fold_data, k, sc, out_base, is_cnn=True)
    
    # Done
    print(f"\n{'='*70}")
    print(f"SCENARIO {sc} COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Results saved to: {out_base}/")
    print(f"Models saved to: saved_models_sheet4/Scenario{sc}/")
    print()


if __name__ == "__main__":
    main()