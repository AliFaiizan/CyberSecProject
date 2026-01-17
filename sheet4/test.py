#!/usr/bin/env python3
"""
Task Sheet 4 - Task 2: Classification with Synthetic Data + VAE Latent Features
FIXED VERSION - Task2 controls folds, Scenario 1 uses ONLY normal synthetic data
"""

import argparse
import os
import time
import psutil
import numpy as np
import pandas as pd
import torch
import pickle
from joblib import dump
from glob import glob

# Classifiers
from sklearn.svm import OneClassSVM, SVC
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Dropout, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler as KerasScaler

# All imports from same folder
from task1 import VAE, extract_latent_features
from gan import Generator
from utils import load_data, create_windows_for_vae
from scenarios import scenario_1_split, scenario_2_split, scenario_3_split


process = psutil.Process()


# =========================================================
# LOAD PRE-TRAINED MODELS
# =========================================================
def load_pretrained_models(vae_checkpoint, device, M=20, F=86):
    """Load GAN generators and VAE"""
    
    print("Loading GAN generators...")
    
    # Load GAN generators
    G_normal = Generator(F).to(device)
    G_attack = Generator(F).to(device)
    
    try:
        G_normal.load_state_dict(torch.load("G_normal.pt", map_location=device))
        G_attack.load_state_dict(torch.load("G_attack.pt", map_location=device))
        print("✓ Loaded GAN generators")
    except RuntimeError as e:
        print(f"\n❌ ERROR loading GAN: {e}")
        print("Your GAN model architecture doesn't match the saved weights!")
        print("Please retrain your GAN or check the Generator class.")
        raise
    
    G_normal.eval()
    G_attack.eval()
    
    # Load VAE
    print(f"Loading VAE from: {vae_checkpoint}")
    
    checkpoint = torch.load(vae_checkpoint, map_location=device)
    
    vae = VAE(
        input_dim=checkpoint['input_dim'],
        latent_dim=checkpoint['latent_dim'],
        layer_type=checkpoint['layer_type'],
        activation=checkpoint['activation'],
        num_classes=None if checkpoint['mode'] == 'reconstruction' else 2,
        seq_len=checkpoint['window_size'],
        feature_dim=checkpoint['feature_dim'],
    )
    
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.to(device)
    vae.eval()
    
    print("✓ Loaded VAE")
    
    # Load scalers
    with open('gan_scaler.pkl', 'rb') as f:
        gan_scaler = pickle.load(f)
    print("✓ Loaded GAN scaler")
    
    with open('vae_scaler.pkl', 'rb') as f:
        vae_scaler = pickle.load(f)
    print("✓ Loaded VAE scaler")
    
    return G_normal, G_attack, vae, gan_scaler, vae_scaler


# =========================================================
# GENERATE SYNTHETIC DATA
# =========================================================
@torch.no_grad()
def generate_synthetic(G, n, device, batch_size=128):
    """Generate n synthetic samples using generator G"""
    if n == 0:
        return np.array([]).reshape(0, -1)
    
    G.eval()
    samples = []
    
    for i in range(0, n, batch_size):
        b = min(batch_size, n - i)
        z = torch.randn(b, 64, device=device)
        samples.append(G(z).cpu().numpy())
    
    return np.vstack(samples)


# =========================================================
# EXTRACT VAE LATENT FEATURES
# =========================================================
@torch.no_grad()
def extract_vae_latent_simple(X_raw, vae, window_size, layer_type, device):
    """Extract VAE latent features from raw data"""
    if len(X_raw) == 0:
        return np.array([])
    
    X_windows, _ = create_windows_for_vae(
        X_raw, np.zeros(len(X_raw)), window_size=window_size, mode="classification"
    )
    
    if len(X_windows) == 0:
        return np.array([])
    
    if layer_type == "dense":
        X_input = X_windows.reshape(len(X_windows), -1)
    else:
        X_input = X_windows
    
    Z, _, _, _ = extract_latent_features(vae, X_input, device=device)
    
    return Z


# =========================================================
# GENERATE SYNTHETIC TRAINING DATA PER FOLD
# =========================================================
def generate_synthetic_fold(train_idx, y_real, G_normal, G_attack, gan_scaler, device, scenario_id):
    """
    Generate synthetic training data for one fold
    
    CRITICAL FIX FOR SCENARIO 1:
    - Scenario 1: Generate ONLY normal synthetic data (no attacks)
    - Scenarios 2 & 3: Generate both normal and attack synthetic data
    """
    
    y_train_real = y_real[train_idx]
    n_normal = np.sum(y_train_real == 0)
    n_attack = len(train_idx) - n_normal
    
    # SCENARIO 1: ONLY NORMAL DATA (anomaly detection)
    if scenario_id == 1:
        print(f"    Generating {n_normal} NORMAL synthetic samples (Scenario 1)")
        
        X_synth_norm = generate_synthetic(G_normal, n_normal, device)
        X_synth = gan_scaler.inverse_transform(X_synth_norm * 5.0)
        y_synth = np.zeros(n_normal, dtype=int)
        
        return X_synth, y_synth
    
    # SCENARIOS 2 & 3: NORMAL + ATTACK DATA
    else:
        print(f"    Generating synthetic data:")
        print(f"      - {n_normal} normal samples")
        print(f"      - {n_attack} attack samples")
        
        # Generate normal
        X_synth_normal_norm = generate_synthetic(G_normal, n_normal, device)
        X_synth_normal = gan_scaler.inverse_transform(X_synth_normal_norm * 5.0)
        y_synth_normal = np.zeros(n_normal, dtype=int)
        
        if n_attack > 0:
            # Generate attack
            X_synth_attack_norm = generate_synthetic(G_attack, n_attack, device)
            X_synth_attack = gan_scaler.inverse_transform(X_synth_attack_norm * 5.0)
            y_synth_attack = np.ones(n_attack, dtype=int)
            
            # Combine
            X_synth = np.vstack([X_synth_normal, X_synth_attack])
            y_synth = np.concatenate([y_synth_normal, y_synth_attack])
        else:
            X_synth = X_synth_normal
            y_synth = y_synth_normal
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X_synth))
        X_synth = X_synth[shuffle_idx]
        y_synth = y_synth[shuffle_idx]
        
        return X_synth, y_synth


# =========================================================
# PER-FOLD CLASSIFIER FUNCTIONS
# =========================================================
def train_OneClassSVM(Z_train, y_train, Z_test, y_test):
    """Train OCSVM on one fold"""
    from sklearn.preprocessing import StandardScaler
    
    # Normalize features to reduce distribution mismatch
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    # Increased nu to 0.05 and changed gamma to 'auto' for better generalization
    model = OneClassSVM(kernel="rbf", nu=0.05, gamma='auto')
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm)
    mem_after = process.memory_info().rss
    
    y_pred_raw = model.predict(Z_test_norm)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def train_LOF(Z_train, y_train, Z_test, y_test):
    """Train LOF on one fold"""
    from sklearn.preprocessing import StandardScaler
    
    # Normalize features
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    # Increased n_neighbors for more robust decision boundary
    model = LocalOutlierFactor(n_neighbors=20, metric='euclidean', novelty=True)
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm)
    mem_after = process.memory_info().rss
    
    y_pred_raw = model.predict(Z_test_norm)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def train_EllipticEnvelope(Z_train, y_train, Z_test, y_test):
    """Train EllipticEnvelope on one fold"""
    from sklearn.preprocessing import StandardScaler
    
    # Normalize features
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    # Increased contamination to match expected outlier ratio
    model = EllipticEnvelope(contamination=0.01, random_state=42)
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm)
    mem_after = process.memory_info().rss
    
    y_pred_raw = model.predict(Z_test_norm)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def train_SVM(Z_train, y_train, Z_test, y_test):
    """Train SVM on one fold"""
    model = SVC(kernel="rbf", C=10.0, gamma='scale')
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train, y_train)
    mem_after = process.memory_info().rss
    
    y_pred = model.predict(Z_test)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def train_kNN(Z_train, y_train, Z_test, y_test):
    """Train kNN on one fold"""
    model = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train, y_train)
    mem_after = process.memory_info().rss
    
    y_pred = model.predict(Z_test)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def train_RandomForest(Z_train, y_train, Z_test, y_test):
    """Train RandomForest on one fold"""
    model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=5,
                                   random_state=42, n_jobs=-1)
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train, y_train)
    mem_after = process.memory_info().rss
    
    y_pred = model.predict(Z_test)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def train_CNN(Z_train, y_train, Z_test, y_test, M):
    """Train CNN on one fold"""
    
    # Create windows
    def create_windows(Z, y, M, stride=1):
        Xw, yw = [], []
        N = len(Z)
        for start in range(0, N - M + 1, stride):
            end = start + M
            Xw.append(Z[start:end])
            label = 1 if np.any(y[start:end] == 1) else 0
            yw.append(label)
        return np.array(Xw), np.array(yw, dtype=int)
    
    X_train_w, y_train_w = create_windows(Z_train, y_train, M)
    X_test_w, y_test_w = create_windows(Z_test, y_test, M)
    
    if len(X_train_w) == 0 or len(X_test_w) == 0:
        return None, None, 0, 0
    
    # Normalize
    scaler = KerasScaler()
    Xtr = X_train_w.reshape(-1, X_train_w.shape[-1])
    Xte = X_test_w.reshape(-1, X_test_w.shape[-1])
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    X_train_w = Xtr.reshape(X_train_w.shape)
    X_test_w = Xte.reshape(X_test_w.shape)
    
    # One-hot
    y_train_cat = to_categorical(y_train_w, 2)
    y_test_cat = to_categorical(y_test_w, 2)
    
    # Build model
    model = Sequential([
        Input(shape=(M, Z_train.shape[1])),
        Conv1D(32, 3, padding="same"), BatchNormalization(), Activation("relu"), Dropout(0.3),
        Conv1D(32, 3, padding="same"), BatchNormalization(), Activation("relu"), Dropout(0.3),
        Conv1D(64, 3, padding="same"), BatchNormalization(), Activation("relu"), Dropout(0.3),
        Conv1D(64, 3, padding="same"), BatchNormalization(), Activation("relu"), Dropout(0.3),
        Flatten(),
        Dense(128, activation="relu"), Dropout(0.3),
        Dense(64, activation="relu"), Dropout(0.3),
        Dense(2, activation="softmax")
    ])
    
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    # Train
    start = time.time()
    mem_before = process.memory_info().rss
    
    model.fit(X_train_w, y_train_cat,
              validation_data=(X_test_w, y_test_cat),
              epochs=10, batch_size=128, verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
    
    mem_after = process.memory_info().rss
    
    # Predict
    y_pred = np.argmax(model.predict(X_test_w, verbose=0), axis=1)
    
    return y_pred, model, time.time() - start, mem_after - mem_before, y_test_w


# =========================================================
# RUN CLASSIFIER ON ALL FOLDS
# =========================================================
def run_classifier(model_name, train_fn, fold_data, scenario_id, out_base, M=None):
    """Run classifier on all folds"""
    
    model_dir = os.path.join(out_base, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Running {model_name}")
    print(f"{'='*70}")
    
    rows = []
    
    for fold_idx, data in fold_data.items():
        print(f"\n  Fold {fold_idx + 1}:")
        
        Z_train = data['Z_train']
        y_train = data['y_train']
        Z_test = data['Z_test']
        y_test = data['y_test']
        
        if Z_train.size == 0 or Z_test.size == 0:
            print(f"    ⚠ Skipping empty fold")
            continue
        
        # Train model
        if model_name == "CNN":
            result = train_fn(Z_train, y_train, Z_test, y_test, M)
            if result[0] is None:
                print(f"    ⚠ CNN failed (not enough windows)")
                continue
            y_pred, model, clf_time, clf_mem, y_test_windows = result
            y_test_use = y_test_windows
        else:
            y_pred, model, clf_time, clf_mem = train_fn(Z_train, y_train, Z_test, y_test)
            y_test_use = y_test
        
        # Compute metrics
        tp = ((y_pred == 1) & (y_test_use == 1)).sum()
        fp = ((y_pred == 1) & (y_test_use == 0)).sum()
        fn = ((y_pred == 0) & (y_test_use == 1)).sum()
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        
        # Save predictions
        pd.DataFrame({
            "predicted_label": y_pred,
            "Attack": y_test_use
        }).to_csv(f"{model_dir}/Predictions_Fold{fold_idx+1}.csv", index=False)
        
        # Save model
        out_dir = f"saved_models_sheet4/Scenario{scenario_id}"
        os.makedirs(out_dir, exist_ok=True)
        
        if model_name == "CNN":
            model.save(f"{out_dir}/{model_name}_Fold{fold_idx+1}.h5")
        else:
            dump(model, f"{out_dir}/{model_name}_Fold{fold_idx+1}.joblib")
        
        rows.append({
            "fold": fold_idx + 1,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "runtime_sec": clf_time,
            "memory_bytes": clf_mem
        })
        
        print(f"    precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
    
    # Save summary
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(f"{model_dir}/metrics_summary.csv", index=False)
    print(f"\n  ✓ Saved to {model_dir}/metrics_summary.csv")
    
    # Print summary
    print(f"\n  {model_name} Summary:")
    print(f"    Avg Precision: {summary_df['precision'].mean():.4f}")
    print(f"    Avg Recall:    {summary_df['recall'].mean():.4f}")
    print(f"    Avg F1-Score:  {summary_df['f1_score'].mean():.4f}")


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description='Task Sheet 4 - Task 2')
    parser.add_argument('-sc', '--scenario', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('-k', '--folds', type=int, default=5)
    parser.add_argument('-M', '--window-size', type=int, default=20)
    parser.add_argument('--vae-checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    sc = args.scenario
    k = args.folds
    M = args.window_size
    F = 86
    
    # Load models
    print("\n" + "="*70)
    print("LOADING MODELS")
    print("="*70)
    
    G_normal, G_attack, vae, gan_scaler, vae_scaler = load_pretrained_models(
        args.vae_checkpoint, device, M, F
    )
    
    checkpoint = torch.load(args.vae_checkpoint, map_location='cpu')
    layer_type = checkpoint['layer_type']
    
    # Load real data
    print("\n" + "="*70)
    print("LOADING REAL DATA")
    print("="*70)
    
    train_files = sorted(glob("../datasets/hai-22.04/train1.csv"))
    test_files = sorted(glob("../datasets/hai-22.04/test1.csv"))
    
    if not train_files or not test_files:
        print("\n❌ ERROR: Dataset not found!")
        return
    
    X, y = load_data(train_files, test_files)
    X_real = vae_scaler.transform(X)
    y_real = y
    
    print(f"Data: {X_real.shape}, Normal: {np.sum(y_real == 0)}, Attack: {np.sum(y_real == 1)}")
    
    # Choose scenario
    if sc == 1:
        scenario_fn = scenario_1_split
    elif sc == 2:
        scenario_fn = scenario_2_split
    else:
        scenario_fn = scenario_3_split
    
    # Generate and process folds
    print("\n" + "="*70)
    print(f"PROCESSING SCENARIO {sc} - GENERATING FOLDS")
    print("="*70)
    
    fold_data = {}
    
    for fold_result in scenario_fn(pd.DataFrame(X_real), pd.Series(y_real), k):
        
        if sc == 1:
            fold_idx, train_idx, test_idx = fold_result
        else:
            fold_idx, attack_id, train_idx, test_idx = fold_result
        
        print(f"\nFold {fold_idx + 1}:")
        
        # Generate synthetic training
        X_train_synth_raw, y_train_synth = generate_synthetic_fold(
            train_idx, y_real, G_normal, G_attack, gan_scaler, device, sc
        )
        
        X_train_synth = vae_scaler.transform(X_train_synth_raw)
        
        # Extract latent features
        print(f"    Extracting VAE features...")
        Z_train = extract_vae_latent_simple(X_train_synth, vae, M, layer_type, device)
        
        X_test_real = X_real[test_idx]
        y_test_real = y_real[test_idx]
        Z_test = extract_vae_latent_simple(X_test_real, vae, M, layer_type, device)
        
        # Get window labels
        _, y_train_win = create_windows_for_vae(X_train_synth, y_train_synth, M, "classification")
        _, y_test_win = create_windows_for_vae(X_test_real, y_test_real, M, "classification")
        
        print(f"    Train: {Z_train.shape}, Test: {Z_test.shape}")
        
        fold_data[fold_idx] = {
            'Z_train': Z_train.astype(np.float32),
            'y_train': y_train_win.astype(np.int32),
            'Z_test': Z_test.astype(np.float32),
            'y_test': y_test_win.astype(np.int32)
        }
    
    # Run classifiers
    out_base = f"exports_sheet4/Scenario{sc}"
    os.makedirs(out_base, exist_ok=True)
    
    if sc == 1:
        print("\n" + "="*70)
        print("RUNNING ANOMALY DETECTION CLASSIFIERS")
        print("="*70)
        
        run_classifier("OCSVM", train_OneClassSVM, fold_data, sc, out_base)
        run_classifier("LOF", train_LOF, fold_data, sc, out_base)
        run_classifier("EllipticEnvelope", train_EllipticEnvelope, fold_data, sc, out_base)
    else:
        print("\n" + "="*70)
        print("RUNNING BINARY CLASSIFIERS")
        print("="*70)
        
        run_classifier("SVM", train_SVM, fold_data, sc, out_base)
        run_classifier("kNN", train_kNN, fold_data, sc, out_base)
        run_classifier("RandomForest", train_RandomForest, fold_data, sc, out_base)
        run_classifier("CNN", train_CNN, fold_data, sc, out_base, M=M)
    
    print(f"\n{'='*70}")
    print(f"SCENARIO {sc} COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Results: {out_base}/")
    print(f"Models: saved_models_sheet4/Scenario{sc}/\n")


if __name__ == "__main__":
    main()