#!/usr/bin/env python3
"""
Task 1: CNN Experiments with Raw Physical Readings
Complete implementation for Scenarios 2 & 3, k=5 and k=10, M=50 and M=100

Usage:
  python run_cnn_task1.py --scenario 2 --k 5 --M 50 --data-type real
  python run_cnn_task1.py --scenario 2 --k 5 --M 50 --data-type synthetic
"""

import argparse
import os
import numpy as np
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import psutil

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

process = psutil.Process()


# =========================================================
# WINDOWING FUNCTIONS
# =========================================================
def create_windows_from_raw(X, y, window_size=20, stride=1):
    """
    Create sliding windows from raw sequential physical readings.
    
    Parameters:
    -----------
    X : np.ndarray
        Raw physical readings, shape (num_timesteps, num_features)
    y : np.ndarray
        Labels for each timestep, shape (num_timesteps,)
    window_size : int
        Number of timesteps per window (default: 20)
    stride : int
        Stride for sliding window (default: 1)
        
    Returns:
    --------
    X_windows : np.ndarray
        Windowed data, shape (num_windows, window_size, num_features)
    y_windows : np.ndarray
        Window labels (1 if ANY timestep in window is attack, else 0)
    """
    if len(X) < window_size:
        print(f"Warning: Data length {len(X)} < window_size {window_size}")
        return np.zeros((0, window_size, X.shape[1])), np.zeros((0,), dtype=int)
    
    num_timesteps, num_features = X.shape
    num_windows = (num_timesteps - window_size) // stride + 1
    
    X_windows = np.zeros((num_windows, window_size, num_features), dtype=np.float32)
    y_windows = np.zeros(num_windows, dtype=np.int32)
    
    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        X_windows[i] = X[start_idx:end_idx]
        # Window label: 1 if ANY timestep in window has attack
        y_windows[i] = int(np.any(y[start_idx:end_idx] == 1))
    
    return X_windows, y_windows


# =========================================================
# CNN MODEL BUILDER
# =========================================================
def build_cnn_model(window_size, num_features, params):
    """
    Build CNN model with 6 blocks (4 conv + 2 FC) as per Task 2 requirements.
    
    Parameters:
    -----------
    window_size : int
        Number of timesteps per window
    num_features : int
        Number of features per timestep
    params : dict
        Hyperparameters
    """
    dropout_rate = params.get("dropout_rate", 0.3)
    lr = params.get("lr", 1e-3)
    conv_filters = params.get("conv_filters", (32, 64, 128, 256))
    kernel_size = params.get("kernel_size", 3)
    dense1_units = params.get("dense1_units", 128)
    dense2_units = params.get("dense2_units", 64)
    
    model = Sequential()
    model.add(Input(shape=(window_size, num_features)))
    
    # ===== Block 1 =====
    model.add(Conv1D(conv_filters[0], kernel_size, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    
    model.add(Conv1D(conv_filters[0], kernel_size, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    
    # ===== Block 2 =====
    model.add(Conv1D(conv_filters[1], kernel_size, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    
    model.add(Conv1D(conv_filters[1], kernel_size, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    
    # ===== Block 3 =====
    model.add(Conv1D(conv_filters[2], kernel_size, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    
    model.add(Conv1D(conv_filters[2], kernel_size, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    
    # ===== Block 4 =====
    model.add(Conv1D(conv_filters[3], kernel_size, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    
    model.add(Conv1D(conv_filters[3], kernel_size, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    
    model.add(Flatten())
    
    # ===== Fully Connected Layers =====
    model.add(Dense(dense1_units, activation="relu"))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(dense2_units, activation="relu"))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(2, activation="softmax"))
    
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


# =========================================================
# TRAIN CNN ON ONE FOLD
# =========================================================
def train_cnn_on_fold(X_train_windows, y_train_windows, 
                      X_test_windows, y_test_windows, params):
    """
    Train CNN on windowed data for one fold.
    """
    tf.keras.backend.clear_session()
    
    start = time.time()
    mem_before = process.memory_info().rss
    
    # Get dimensions
    num_windows_train, window_size, num_features = X_train_windows.shape
    num_windows_test = X_test_windows.shape[0]
    
    # Normalize: flatten, scale, reshape
    X_train_flat = X_train_windows.reshape(-1, num_features)
    X_test_flat = X_test_windows.reshape(-1, num_features)
    
    scaler = StandardScaler()
    X_train_flat_norm = scaler.fit_transform(X_train_flat)
    X_test_flat_norm = scaler.transform(X_test_flat)
    
    X_train_norm = X_train_flat_norm.reshape(num_windows_train, window_size, num_features)
    X_test_norm = X_test_flat_norm.reshape(num_windows_test, window_size, num_features)
    
    # One-hot encode labels
    y_train_cat = to_categorical(y_train_windows, 2)
    y_test_cat = to_categorical(y_test_windows, 2)
    
    # Class weights
    use_class_weight = params.get("use_class_weight", True)
    if use_class_weight:
        n_normal = np.sum(y_train_windows == 0)
        n_attack = np.sum(y_train_windows == 1)
        total = len(y_train_windows)
        if n_attack > 0 and n_normal > 0:
            class_weight = {
                0: total / (2 * n_normal),
                1: total / (2 * n_attack)
            }
        else:
            class_weight = None
    else:
        class_weight = None
    
    # Build model
    model = build_cnn_model(window_size, num_features, params)
    
    # Train
    batch_size = params.get("batch_size", 128)
    epochs = params.get("epochs", 50)
    patience = params.get("patience", 10)
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        )
    ]
    
    history = model.fit(
        X_train_norm, y_train_cat,
        validation_data=(X_test_norm, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        verbose=1,
        callbacks=callbacks
    )
    
    mem_after = process.memory_info().rss
    
    # Predict
    y_pred_proba = model.predict(X_test_norm, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    return y_pred, model, time.time() - start, mem_after - mem_before, history


# =========================================================
# LOAD FOLD DATA
# =========================================================
def load_fold_data(fold_dir, data_type, scenario):
    """Load raw data from fold directory."""
    fold_num = os.path.basename(fold_dir).replace('fold', '')
    parent_dir = os.path.dirname(fold_dir)
    
    if data_type == 'synthetic':
        syn_fold_dir = os.path.join(parent_dir, f'syn_fold{fold_num}')
        X_train = np.load(f"{syn_fold_dir}/train_raw.npy")
        y_train = np.load(f"{syn_fold_dir}/train_raw_labels.npy")
        X_test = np.load(f"{syn_fold_dir}/test_raw.npy")
        y_test = np.load(f"{syn_fold_dir}/test_raw_labels.npy")
    else:
        real_fold_dir = os.path.join(parent_dir, f'real_fold{fold_num}')
        X_train = np.load(f"{real_fold_dir}/train_raw.npy")
        y_train = np.load(f"{real_fold_dir}/train_raw_labels.npy")
        X_test = np.load(f"{real_fold_dir}/test_raw.npy")
        y_test = np.load(f"{real_fold_dir}/test_raw_labels.npy")
    
    return X_train, y_train, X_test, y_test


# =========================================================
# RUN CNN EXPERIMENT
# =========================================================
def run_cnn_experiment(scenario, k, M, data_type, window_size=20, stride=1, params=None):
    """Run complete CNN experiment with k-fold CV."""
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: Scenario {scenario}, k={k}, M={M}, Data={data_type}")
    print("="*80)
    
    # Try cnn_folds first, fallback to exports_sheet4
    fold_base_dir = f"cnn_folds/Scenario{scenario}_k{k}"
    if not os.path.exists(fold_base_dir):
        fold_base_dir = f"exports_sheet4/Scenario{scenario}"
        if not os.path.exists(fold_base_dir):
            print(f"❌ ERROR: Fold directory not found!")
            print(f"Tried: cnn_folds/Scenario{scenario} and exports_sheet4/Scenario{scenario}")
            return None
    
    print(f"Using fold directory: {fold_base_dir}")
    
    if params is None:
        params = {
            "dropout_rate": 0.3,
            "lr": 1e-3,
            "batch_size": 128,
            "epochs": 50,
            "patience": 10,
            "conv_filters": (32, 64, 128, 256),
            "kernel_size": 3,
            "dense1_units": 128,
            "dense2_units": 64,
            "use_class_weight": True
        }
    
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_conf_matrices = []
    
    for fold_idx in range(k):
        fold_dir = f"{fold_base_dir}/fold{fold_idx}"
        
        print(f"\n{'─'*80}")
        print(f"Fold {fold_idx + 1}/{k}")
        print(f"{'─'*80}")
        
        # Load raw data
        X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_fold_data(
            fold_dir, data_type, scenario
        )
        
        # Feature selection: use first M features
        if M < X_train_raw.shape[1]:
            print(f"Selecting first {M} features out of {X_train_raw.shape[1]}")
            X_train_raw = X_train_raw[:, :M]
            X_test_raw = X_test_raw[:, :M]
        
        print(f"Loaded raw data:")
        print(f"  Train: {X_train_raw.shape}, Normal={np.sum(y_train_raw==0)}, Attack={np.sum(y_train_raw==1)}")
        print(f"  Test:  {X_test_raw.shape}, Normal={np.sum(y_test_raw==0)}, Attack={np.sum(y_test_raw==1)}")
        
        # Create windows
        print(f"\nCreating windows (size={window_size}, stride={stride})...")
        X_train_windows, y_train_windows = create_windows_from_raw(
            X_train_raw, y_train_raw, window_size, stride
        )
        X_test_windows, y_test_windows = create_windows_from_raw(
            X_test_raw, y_test_raw, window_size, stride
        )
        
        print(f"  Train windows: {X_train_windows.shape}")
        print(f"  Test windows:  {X_test_windows.shape}")
        print(f"  Train labels: Normal={np.sum(y_train_windows==0)}, Attack={np.sum(y_train_windows==1)}")
        print(f"  Test labels:  Normal={np.sum(y_test_windows==0)}, Attack={np.sum(y_test_windows==1)}")
        
        # Train CNN
        print(f"\nTraining CNN...")
        y_pred, model, train_time, mem_used, history = train_cnn_on_fold(
            X_train_windows, y_train_windows,
            X_test_windows, y_test_windows,
            params
        )
        
        # Calculate metrics
        precision = precision_score(y_test_windows, y_pred, zero_division=0)
        recall = recall_score(y_test_windows, y_pred, zero_division=0)
        f1 = f1_score(y_test_windows, y_pred, zero_division=0)
        cm = confusion_matrix(y_test_windows, y_pred)
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        all_conf_matrices.append(cm)
        
        print(f"\n📊 Fold {fold_idx + 1} Results:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Memory used: {mem_used / 1024**2:.2f} MB")
        print(f"  Confusion Matrix:\n{cm}")
    
    # Aggregate results
    results = {
        "scenario": scenario,
        "k_folds": k,
        "M": M,
        "data_type": data_type,
        "window_size": window_size,
        "stride": stride,
        "mean_precision": float(np.mean(all_precisions)),
        "std_precision": float(np.std(all_precisions)),
        "mean_recall": float(np.mean(all_recalls)),
        "std_recall": float(np.std(all_recalls)),
        "mean_f1": float(np.mean(all_f1s)),
        "std_f1": float(np.std(all_f1s)),
        "per_fold_precision": [float(p) for p in all_precisions],
        "per_fold_recall": [float(r) for r in all_recalls],
        "per_fold_f1": [float(f) for f in all_f1s],
        "params": params
    }
    
    print("\n" + "="*80)
    print("📊 OVERALL RESULTS")
    print("="*80)
    print(f"Precision: {results['mean_precision']:.4f} ± {results['std_precision']:.4f}")
    print(f"Recall:    {results['mean_recall']:.4f} ± {results['std_recall']:.4f}")
    print(f"F1-Score:  {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    print("="*80)
    
    return results


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Task 1: CNN Experiments")
    parser.add_argument("--scenario", type=int, required=True, choices=[2, 3])
    parser.add_argument("--k", type=int, required=True, choices=[5, 10])
    parser.add_argument("--M", type=int, required=True, choices=[50, 86],
                       help="Number of features to use (50 or 86)")
    parser.add_argument("--data-type", type=str, required=True, choices=['real', 'synthetic'])
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    
    # CNN hyperparameters
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    
    parser.add_argument("--output-dir", type=str, default="task1_results")
    
    args = parser.parse_args()
    
    # CNN parameters
    cnn_params = {
        "dropout_rate": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "conv_filters": (32, 64, 128, 256),
        "kernel_size": 3,
        "dense1_units": 128,
        "dense2_units": 64,
        "use_class_weight": True
    }
    
    # Run experiment
    results = run_cnn_experiment(
        scenario=args.scenario,
        k=args.k,
        M=args.M,
        data_type=args.data_type,
        window_size=args.window_size,
        stride=args.stride,
        params=cnn_params
    )
    
    if results is None:
        print("\n❌ Experiment failed!")
        return
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/scenario{args.scenario}_k{args.k}_M{args.M}_{args.data_type}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
