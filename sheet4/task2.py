#!/usr/bin/env python3
"""
Task Sheet 4 - Task 2: Classification with Synthetic Data + VAE Latent Features

CORRECTED: Uses REAL HAI-22.04 dataset for test data, synthetic for training
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import pickle
from joblib import dump
from glob import glob

# All imports from same folder
from task1 import VAE, extract_latent_features
from utils import load_data, create_windows_for_vae
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
from gan import Generator


# =========================================================
# LOAD PRE-TRAINED MODELS
# =========================================================
def load_pretrained_models(vae_checkpoint, device, M=20, F=86):
    """Load GAN generators and VAE"""
    
    print("Loading GAN generators...")
    
    # Load GAN generators
    G_normal = Generator(F).to(device)
    G_attack = Generator(F).to(device)
    
    G_normal.load_state_dict(torch.load("G_normal.pt", map_location=device))
    G_attack.load_state_dict(torch.load("G_attack.pt", map_location=device))
    
    G_normal.eval()
    G_attack.eval()
    
    print("✓ Loaded GAN generators")
    
    # Load VAE
    print(f"Loading VAE from: {vae_checkpoint}")
    
    checkpoint = torch.load(vae_checkpoint, map_location=device)
    
    # Reconstruct VAE
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
    
    # Load GAN scaler
    print("Loading GAN scaler...")
    with open('gan_scaler.pkl', 'rb') as f:
        gan_scaler = pickle.load(f)
    print("✓ Loaded GAN scaler")
    
    # Load VAE scaler
    print("Loading VAE scaler...")
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
    
    # Create windows
    X_windows, _ = create_windows_for_vae(
        X_raw,
        np.zeros(len(X_raw)),
        window_size=window_size,
        mode="classification"
    )
    
    if len(X_windows) == 0:
        return np.array([])
    
    # Prepare input based on layer type
    if layer_type == "dense":
        X_input = X_windows.reshape(len(X_windows), -1)
    else:
        X_input = X_windows
    
    # Extract features
    Z, _, _, _ = extract_latent_features(vae, X_input, device=device)
    
    return Z


# =========================================================
# GENERATE TRAINING SET FOR ONE FOLD
# =========================================================
def generate_synthetic_training_set(train_idx, y_real, G_normal, G_attack, 
                                     gan_scaler, device):
    """Generate fully synthetic training set"""
    
    y_train_real = y_real[train_idx]
    n_normal = np.sum(y_train_real == 0)
    n_attack = len(train_idx) - n_normal
    
    print(f"  Generating synthetic training data:")
    print(f"    - {n_normal} normal samples")
    print(f"    - {n_attack} attack samples")
    
    # Generate synthetic normal
    X_synth_normal_norm = generate_synthetic(G_normal, n_normal, device)
    X_synth_normal = gan_scaler.inverse_transform(X_synth_normal_norm * 5.0)
    y_synth_normal = np.zeros(n_normal, dtype=int)
    
    if n_attack > 0:
        # Generate synthetic attack
        X_synth_attack_norm = generate_synthetic(G_attack, n_attack, device)
        X_synth_attack = gan_scaler.inverse_transform(X_synth_attack_norm * 5.0)
        y_synth_attack = np.ones(n_attack, dtype=int)
        
        # Combine
        X_train_synthetic = np.vstack([X_synth_normal, X_synth_attack])
        y_train_synthetic = np.concatenate([y_synth_normal, y_synth_attack])
    else:
        X_train_synthetic = X_synth_normal
        y_train_synthetic = y_synth_normal
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_train_synthetic))
    X_train_synthetic = X_train_synthetic[shuffle_idx]
    y_train_synthetic = y_train_synthetic[shuffle_idx]
    
    return X_train_synthetic, y_train_synthetic


# =========================================================
# SAVE MODEL
# =========================================================
def save_model(model, scenario_id, model_name, fold_idx):
    out_dir = f"saved_models_sheet4/Scenario{scenario_id}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{model_name}_Fold{fold_idx+1}.joblib"
    dump(model, path)


# =========================================================
# RUN AND SAVE RESULTS
# =========================================================
def run_and_save(model_name, run_fn, X, y, k, scenario_fn, scenario_id, out_base):
    """Run classifier using your existing model functions"""
    
    model_dir = os.path.join(out_base, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Running {model_name} for Scenario {scenario_id}")
    print(f"{'='*70}")
    
    # Call the existing model runner (same as Task Sheet 3)
    results = run_fn(X, y, k, scenario_fn)
    
    rows = []
    for res in results:
        if scenario_id == 1:
            fold_idx, test_idx, y_pred, y_test, model, fe_time, fe_mem, clf_time, clf_mem = res
            attack_id = None
        else:
            fold_idx, attack_id, y_pred, y_test, model, fe_time, fe_mem, clf_time, clf_mem = res
        
        # Compute metrics
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        
        # Save predictions
        pd.DataFrame({
            "predicted_label": y_pred,
            "Attack": y_test
        }).to_csv(f"{model_dir}/Predictions_Fold{fold_idx+1}.csv", index=False)
        
        save_model(model, scenario_id, model_name, fold_idx)
        
        rows.append({
            "fold": fold_idx + 1,
            "attack_id": attack_id,
            "precision": precision,
            "recall": recall,
            "feature_runtime_sec": fe_time,
            "feature_memory_bytes": fe_mem,
            "runtime_sec": clf_time,
            "memory_bytes": clf_mem
        })
        
        print(f"  Fold {fold_idx+1}: precision={precision:.4f}, recall={recall:.4f}")
    
    # Save summary
    pd.DataFrame(rows).to_csv(f"{model_dir}/metrics_summary.csv", index=False)
    print(f"  ✓ Saved metrics summary")


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n" + "="*70)
    parser = argparse.ArgumentParser(description='Task Sheet 4 - Task 2')
    parser.add_argument('-sc', '--scenario', type=int, required=True, 
                        choices=[1, 2, 3], help='Scenario number')
    parser.add_argument('-k', '--folds', type=int, default=5, 
                        help='Number of cross-validation folds')
    parser.add_argument('-M', '--window-size', type=int, default=20,
                        help='Window size')
    parser.add_argument('--vae-checkpoint', type=str, required=True,
                        help='Path to trained VAE checkpoint')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    sc = args.scenario
    k = args.folds
    M = args.window_size
    F = 86
    
    # =========================================================
    # LOAD PRE-TRAINED MODELS
    # =========================================================
    print("\n" + "="*70)
    print("LOADING PRE-TRAINED MODELS")
    print("="*70)
    
    G_normal, G_attack, vae, gan_scaler, vae_scaler = load_pretrained_models(
        vae_checkpoint=args.vae_checkpoint,
        device=device,
        M=M,
        F=F
    )
    
    # Get VAE config
    checkpoint = torch.load(args.vae_checkpoint, map_location='cpu')
    layer_type = checkpoint['layer_type']
    
    # =========================================================
    # LOAD REAL HAI-22.04 DATA
    # =========================================================
    print("\n" + "="*70)
    print("LOADING REAL HAI-22.04 DATA")
    print("="*70)
    
    # Load REAL HAI dataset from CSVs
    train_files = sorted(glob("../datasets/hai-22.04/train1.csv"))
    test_files  = sorted(glob("../datasets/hai-22.04/test1.csv"))
    
    if not train_files or not test_files:
        print("\n❌ ERROR: HAI-22.04 dataset not found!")
        print("Expected location: ../datasets/hai-22.04/")
        print("\nPlease ensure you have:")
        print("  - ../datasets/hai-22.04/train1.csv")
        print("  - ../datasets/hai-22.04/test1.csv")
        return
    
    print(f"Found {len(train_files)} training file(s)")
    print(f"Found {len(test_files)} test file(s)")
    
    # Load using your existing load_data function
    X, y = load_data(train_files, test_files)   # X: [T,F], y: [T]
    
    print(f"\nReal HAI-22.04 data loaded:")
    print(f"  Shape: {X.shape}")
    print(f"  Normal samples: {np.sum(y == 0)}")
    print(f"  Attack samples: {np.sum(y == 1)}")
    
    # Normalize using VAE scaler
    X_real = vae_scaler.transform(X)
    y_real = y
    print("Normalized using VAE scaler")
    
    # =========================================================
    # GENERATE SYNTHETIC DATA + EXTRACT LATENT FEATURES
    # =========================================================
    print("\n" + "="*70)
    print(f"PROCESSING SCENARIO {sc}")
    print("="*70)
    
    # Choose scenario
    if sc == 1:
        scenario_fn = scenario_1_split
    elif sc == 2:
        scenario_fn = scenario_2_split
    else:
        scenario_fn = scenario_3_split
    
    # Generate synthetic latent features for ALL folds
    print("\nGenerating synthetic latent features per fold...")
    
    Z_all = []
    y_all = []
    fold_idx_map = []
    
    for fold_result in scenario_fn(pd.DataFrame(X_real), pd.Series(y_real), k):
        
        if sc == 1:
            fold_idx, train_idx, test_idx = fold_result
        else:
            fold_idx, attack_id, train_idx, test_idx = fold_result
        
        print(f"\nFold {fold_idx + 1}:")
        
        # =====================================================
        # STEP 1: Generate Synthetic Training Data
        # =====================================================
        X_train_synthetic_raw, y_train_synthetic = generate_synthetic_training_set(
            train_idx, y_real, G_normal, G_attack, gan_scaler, device
        )
        
        # Normalize synthetic data using VAE scaler
        X_train_synthetic = vae_scaler.transform(X_train_synthetic_raw)
        
        # =====================================================
        # STEP 2: Extract VAE Latent Features
        # =====================================================
        print(f"  Extracting latent features...")
        
        # From synthetic training data
        Z_train_synthetic = extract_vae_latent_simple(
            X_train_synthetic, vae, M, layer_type, device
        )
        
        # From REAL test data (this is the key difference!)
        X_test_real_fold = X_real[test_idx]
        y_test_real_fold = y_real[test_idx]
        
        Z_test_real = extract_vae_latent_simple(
            X_test_real_fold, vae, M, layer_type, device
        )
        
        # Adjust labels to match window count
        _, y_train_windows = create_windows_for_vae(
            X_train_synthetic, y_train_synthetic, M, "classification"
        )
        _, y_test_windows = create_windows_for_vae(
            X_test_real_fold, y_test_real_fold, M, "classification"
        )
        
        print(f"    Train (synthetic): {Z_train_synthetic.shape}")
        print(f"    Test (REAL):       {Z_test_real.shape}")
        
        # Combine train (synthetic) and test (real) for this fold
        Z_fold = np.vstack([Z_train_synthetic, Z_test_real])
        y_fold = np.concatenate([y_train_windows, y_test_windows])
        
        Z_all.append(Z_fold)
        y_all.append(y_fold)
        fold_idx_map.extend([fold_idx] * len(Z_fold))
    
    # Combine all folds
    Z_combined = np.vstack(Z_all)
    y_combined = np.concatenate(y_all)
    
    print(f"\nCombined latent features: {Z_combined.shape}")
    
    # Convert to DataFrame for compatibility with existing functions
    X_latent_df = pd.DataFrame(Z_combined)
    y_latent_series = pd.Series(y_combined)
    
    # =========================================================
    # RUN CLASSIFIERS (using existing functions)
    # =========================================================
    out_base = f"exports_sheet4/Scenario{sc}"
    os.makedirs(out_base, exist_ok=True)
    
    if sc == 1:
        # Anomaly detection classifiers
        print("\n" + "="*70)
        print("RUNNING ANOMALY DETECTION CLASSIFIERS")
        print("="*70)
        
        run_and_save("OCSVM", run_OneClassSVM, X_latent_df, y_latent_series, 
                     k, scenario_fn, sc, out_base)
        run_and_save("LOF", run_LOF, X_latent_df, y_latent_series, 
                     k, scenario_fn, sc, out_base)
        run_and_save("EllipticEnvelope", run_EllipticEnvelope, X_latent_df, y_latent_series, 
                     k, scenario_fn, sc, out_base)
    else:
        # Binary classifiers
        print("\n" + "="*70)
        print("RUNNING BINARY CLASSIFIERS")
        print("="*70)
        
        run_and_save("SVM", run_binary_svm, X_latent_df, y_latent_series, 
                     k, scenario_fn, sc, out_base)
        run_and_save("kNN", run_knn, X_latent_df, y_latent_series, 
                     k, scenario_fn, sc, out_base)
        run_and_save("RandomForest", run_random_forest, X_latent_df, y_latent_series, 
                     k, scenario_fn, sc, out_base)
    
    # =========================================================
    # CNN (for Scenario 2 & 3)
    # =========================================================
    if sc in [2, 3]:
        cnn_dir = f"{out_base}/CNN"
        os.makedirs(cnn_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Running CNN for Scenario {sc}")
        print(f"{'='*70}")
        
        # Run CNN using existing function
        # CNN expects: (Z, y, scenario_id, k, out_dir, M)
        cnn_results = run_cnn_latent(X_latent_df.values, y_latent_series.values, sc, k, cnn_dir, M)
        
        metrics = []
        for res in cnn_results:
            fold_idx, model, y_pred, y_test, total_runtime, total_memory, detail_dict = res
            
            # Compute metrics
            tp = ((y_pred == 1) & (y_test == 1)).sum()
            fp = ((y_pred == 1) & (y_test == 0)).sum()
            fn = ((y_pred == 0) & (y_test == 1)).sum()
            
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            
            # Save predictions
            pd.DataFrame({
                "predicted_label": y_pred,
                "Attack": y_test
            }).to_csv(f"{cnn_dir}/Predictions_Fold{fold_idx+1}.csv", index=False)
            
            # Save model
            os.makedirs(f"saved_models_sheet4/Scenario{sc}", exist_ok=True)
            model.save(f"saved_models_sheet4/Scenario{sc}/CNN_Fold{fold_idx+1}.h5")
            
            metrics.append({
                "fold": fold_idx + 1,
                "precision": precision,
                "recall": recall,
                "feature_runtime_sec": detail_dict.get("window_time", 0) + detail_dict.get("norm_time", 0),
                "feature_memory_bytes": detail_dict.get("window_mem", 0),
                "runtime_sec": total_runtime,
                "memory_bytes": total_memory
            })
            
            print(f"  Fold {fold_idx+1}: precision={precision:.4f}, recall={recall:.4f}")
        
        pd.DataFrame(metrics).to_csv(f"{cnn_dir}/metrics_summary.csv", index=False)
        print(f"  ✓ Saved CNN metrics summary")
    
    # =========================================================
    # DONE
    # =========================================================
    print(f"\n{'='*70}")
    print(f"SCENARIO {sc} COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Results saved to: {out_base}/")
    print(f"Models saved to: saved_models_sheet4/Scenario{sc}/")
    print()


if __name__ == "__main__":
    main()