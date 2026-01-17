#!/usr/bin/env python3
"""
Generate and save fold data for Task Sheet 4 - Task 2
FIXED: Scenario 1 generates ONLY normal synthetic data

This script:
1. Loads pre-trained GAN and VAE models
2. Loads real HAI-22.04 data
3. Generates synthetic training data per fold (ONLY normal for Scenario 1)
4. Extracts VAE latent features
5. Saves fold data for later classifier training

Usage:
    python generate_folds_FIXED.py -sc 1 -k 5 -M 20 --vae-checkpoint vae_reconstruction_real.pt
    python generate_folds_FIXED.py -sc 2 -k 5 -M 20 --vae-checkpoint vae_classification_real.pt
    python generate_folds_FIXED.py -sc 3 -k 5 -M 20 --vae-checkpoint vae_classification_real.pt
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import pickle
from glob import glob

# All imports from same folder
from vae import VAE, extract_latent_features
from gan import Generator
from utils import load_data, create_windows_for_vae
from scenarios import scenario_1_split, scenario_2_split, scenario_3_split


# =========================================================
# LOAD PRE-TRAINED MODELS
# =========================================================
def load_pretrained_models(vae_checkpoint, device, M=20, F=86):
    """Load GAN generators and VAE"""
    
    print("Loading GAN generators...")
    
    G_normal = Generator(F).to(device)
    G_attack = Generator(F).to(device)
    
    G_normal.load_state_dict(torch.load("G_normal.pt", map_location=device))
    G_attack.load_state_dict(torch.load("G_attack.pt", map_location=device))
    
    G_normal.eval()
    G_attack.eval()
    
    print("✓ Loaded GAN generators")
    
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
# GENERATE TRAINING SET FOR SCENARIOS 2 & 3
# =========================================================
def generate_synthetic_training_set(train_idx, y_real, G_normal, G_attack, 
                                     gan_scaler, device):
    """Generate synthetic training set for Scenarios 2 & 3"""
    
    y_train_real = y_real[train_idx]
    n_normal = np.sum(y_train_real == 0)
    n_attack = len(train_idx) - n_normal
    
    print(f"    Generating synthetic training data:")
    print(f"      - {n_normal} normal samples")
    print(f"      - {n_attack} attack samples")
    
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
# MAIN
# =========================================================
def main():
    print("\n" + "="*70)
    print("FOLD DATA GENERATION FOR TASK SHEET 4")
    print("="*70)
    
    parser = argparse.ArgumentParser(description='Generate fold data for Task Sheet 4')
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
    
    # Load models
    print("\n" + "="*70)
    print("LOADING PRE-TRAINED MODELS")
    print("="*70)
    
    G_normal, G_attack, vae, gan_scaler, vae_scaler = load_pretrained_models(
        vae_checkpoint=args.vae_checkpoint, device=device, M=M, F=F
    )
    
    checkpoint = torch.load(args.vae_checkpoint, map_location='cpu')
    layer_type = checkpoint['layer_type']
    
    # Load real data
    print("\n" + "="*70)
    print("LOADING REAL HAI-22.04 DATA")
    print("="*70)
    
    train_files = sorted(glob("../datasets/hai-22.04/train1.csv"))
    test_files  = sorted(glob("../datasets/hai-22.04/test1.csv"))
    
    if not train_files or not test_files:
        print("\n❌ ERROR: HAI-22.04 dataset not found!")
        return
    
    print(f"Found {len(train_files)} training file(s)")
    print(f"Found {len(test_files)} test file(s)")
    
    X, y = load_data(train_files, test_files)
    
    print(f"\nReal HAI-22.04 data loaded:")
    print(f"  Shape: {X.shape}")
    print(f"  Normal samples: {np.sum(y == 0)}")
    print(f"  Attack samples: {np.sum(y == 1)}")
    
    X_real = vae_scaler.transform(X)
    y_real = y
    print("✓ Normalized using VAE scaler")
    
    # Choose scenario
    print("\n" + "="*70)
    print(f"PROCESSING SCENARIO {sc}")
    print("="*70)
    
    if sc == 1:
        scenario_fn = scenario_1_split
    elif sc == 2:
        scenario_fn = scenario_2_split
    else:
        scenario_fn = scenario_3_split
    
    # Generate folds
    print("\nGenerating synthetic latent features per fold...")
    
    fold_data_dir = f"exports_sheet4/Scenario{sc}"
    os.makedirs(fold_data_dir, exist_ok=True)
    
    for fold_result in scenario_fn(pd.DataFrame(X_real), pd.Series(y_real), k):
        
        if sc == 1:
            fold_idx, train_idx, test_idx = fold_result
        else:
            fold_idx, attack_id, train_idx, test_idx = fold_result
        
        print(f"\nFold {fold_idx + 1}:")
        
        # =======================================================
        # STEP 1: Generate Synthetic Training Data
        # CRITICAL FIX: Scenario 1 = ONLY normal synthetic
        # =======================================================
        if sc == 1:
            # Scenario 1: ONLY normal (no attacks in training)
            y_train_real = y_real[train_idx]
            n_normal = np.sum(y_train_real == 0)
            
            print(f"    Generating {n_normal} NORMAL synthetic samples (Scenario 1)")
            
            X_synth_normal_norm = generate_synthetic(G_normal, n_normal, device)
            X_train_synthetic_raw = gan_scaler.inverse_transform(X_synth_normal_norm)
            y_train_synthetic = np.zeros(n_normal, dtype=int)
        else:
            # Scenarios 2 & 3: Normal + Attacks
            X_train_synthetic_raw, y_train_synthetic = generate_synthetic_training_set(
                train_idx, y_real, G_normal, G_attack, gan_scaler, device
            )
        
        # Normalize
        X_train_synthetic = vae_scaler.transform(X_train_synthetic_raw)
        
        # =======================================================
        # STEP 2: Extract VAE Latent Features
        # =======================================================
        print(f"    Extracting latent features...")
        
        # From synthetic training data
        Z_train_synthetic = extract_vae_latent_simple(
            X_train_synthetic, vae, M, layer_type, device
        )
        
        # From REAL test data
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
        
        print(f"      Train (synthetic): {Z_train_synthetic.shape}")
        print(f"      Test (REAL):       {Z_test_real.shape}")
        
        # =======================================================
        # STEP 3: Save Fold Data
        # =======================================================
        fold_dir = f"{fold_data_dir}/fold{fold_idx}"
        os.makedirs(fold_dir, exist_ok=True)
        
        np.save(f"{fold_dir}/train_latent.npy", Z_train_synthetic.astype(np.float32))
        np.save(f"{fold_dir}/train_labels.npy", y_train_windows.astype(np.int32)) # y_train_synthetic
        np.save(f"{fold_dir}/test_latent.npy", Z_test_real.astype(np.float32))
        np.save(f"{fold_dir}/test_labels.npy", y_test_windows.astype(np.int32)) # y_test_real_fold
        
        print(f"    ✓ Saved to {fold_dir}/")
    
    # Done
    print(f"\n{'='*70}")
    print(f"FOLD DATA GENERATION COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Fold data saved to: {fold_data_dir}/")
    print(f"\nNext step: Run classifiers using:")
    print(f"  python task2_FIXED.py -sc {sc} -k {k}")
    print()


if __name__ == "__main__":
    main()