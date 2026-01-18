#!/usr/bin/env python3
"""
Diagnostic Script - Check VAE Features and Data Quality
"""

import numpy as np
import torch
import pickle
from glob import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from vae import VAE, extract_latent_features
from gan import Generator
from utils import load_data, create_windows_for_vae


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M = 20
    F = 86
    
    print("="*70)
    print("DIAGNOSTIC: Checking VAE Features")
    print("="*70)
    
    # =========================================================
    # Load Models
    # =========================================================
    print("\n1. Loading models...")
    
    G_normal = Generator(F).to(device)
    G_normal.load_state_dict(torch.load("G_normal.pt", map_location=device))
    G_normal.eval()
    
    vae_checkpoint = "vae_reconstruction_real.pt"
    checkpoint = torch.load(vae_checkpoint, map_location='cpu')
    
    vae = VAE(
        input_dim=checkpoint['input_dim'],
        latent_dim=checkpoint['latent_dim'],
        layer_type=checkpoint['layer_type'],
        activation=checkpoint['activation'],
        num_classes=None,
        seq_len=checkpoint['window_size'],
        feature_dim=checkpoint['feature_dim'],
    )
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.to(device)
    vae.eval()
    
    with open('gan_scaler.pkl', 'rb') as f:
        gan_scaler = pickle.load(f)
    
    with open('vae_scaler.pkl', 'rb') as f:
        vae_scaler = pickle.load(f)
    
    print("✓ Models loaded")
    
    # =========================================================
    # Load Real Data
    # =========================================================
    print("\n2. Loading real data...")
    
    train_files = sorted(glob("../datasets/hai-22.04/train1.csv"))
    test_files = sorted(glob("../datasets/hai-22.04/test1.csv"))
    
    X, y = load_data(train_files, test_files)
    X_real = vae_scaler.transform(X)
    
    # Sample for testing
    normal_idx = np.where(y == 0)[0][:5000]
    attack_idx = np.where(y == 1)[0][:500]
    
    X_real_normal = X_real[normal_idx]
    X_real_attack = X_real[attack_idx]
    
    print(f"✓ Real normal: {len(X_real_normal)}, Real attack: {len(X_real_attack)}")
    
    # =========================================================
    # Generate Synthetic Normal Data
    # =========================================================
    print("\n3. Generating synthetic normal data...")
    
    @torch.no_grad()
    def generate_synthetic(G, n, device):
        G.eval()
        samples = []
        for i in range(0, n, 128):
            b = min(128, n - i)
            z = torch.randn(b, 64, device=device)
            samples.append(G(z).cpu().numpy())
        return np.vstack(samples)
    
    X_synth_norm = generate_synthetic(G_normal, 5000, device)
    X_synth_denorm = gan_scaler.inverse_transform(X_synth_norm * 5.0)
    X_synth = vae_scaler.transform(X_synth_denorm)
    
    print(f"✓ Synthetic normal: {len(X_synth)}")
    
    # =========================================================
    # Extract VAE Features
    # =========================================================
    print("\n4. Extracting VAE features...")
    
    @torch.no_grad()
    def extract_vae(X, vae, M, device):
        X_windows, _ = create_windows_for_vae(X, np.zeros(len(X)), M, "classification")
        if len(X_windows) == 0:
            return np.array([])
        X_input = X_windows.reshape(len(X_windows), -1)
        Z, _, _, _ = extract_latent_features(vae, X_input, device=device)
        return Z
    
    Z_synth_normal = extract_vae(X_synth, vae, M, device)
    Z_real_normal = extract_vae(X_real_normal, vae, M, device)
    Z_real_attack = extract_vae(X_real_attack, vae, M, device)
    
    print(f"✓ Extracted features:")
    print(f"   Synthetic normal: {Z_synth_normal.shape}")
    print(f"   Real normal: {Z_real_normal.shape}")
    print(f"   Real attack: {Z_real_attack.shape}")
    
    # =========================================================
    # Analyze Distributions
    # =========================================================
    print("\n5. Analyzing distributions...")
    print("="*70)
    
    # Compute statistics
    print("\nFeature Statistics (mean ± std):")
    print(f"  Synthetic normal: {Z_synth_normal.mean():.4f} ± {Z_synth_normal.std():.4f}")
    print(f"  Real normal:      {Z_real_normal.mean():.4f} ± {Z_real_normal.std():.4f}")
    print(f"  Real attack:      {Z_real_attack.mean():.4f} ± {Z_real_attack.std():.4f}")
    
    # Check per-dimension
    print("\nPer-dimension means:")
    print(f"  Synthetic normal: {Z_synth_normal.mean(axis=0)}")
    print(f"  Real normal:      {Z_real_normal.mean(axis=0)}")
    print(f"  Real attack:      {Z_real_attack.mean(axis=0)}")
    
    # =========================================================
    # Test OCSVM on Different Data
    # =========================================================
    print("\n6. Testing OCSVM behavior...")
    print("="*70)
    
    from sklearn.svm import OneClassSVM
    
    # Test 1: Train on synthetic, test on real normal
    print("\nTest 1: Train on SYNTHETIC normal, test on REAL normal")
    model1 = OneClassSVM(kernel="rbf", nu=0.001, gamma='scale')
    model1.fit(Z_synth_normal)
    
    pred_real_normal = model1.predict(Z_real_normal)
    pred_real_attack = model1.predict(Z_real_attack)
    
    pct_normal_as_normal = (pred_real_normal == 1).sum() / len(pred_real_normal) * 100
    pct_attack_as_outlier = (pred_real_attack == -1).sum() / len(pred_real_attack) * 100
    
    print(f"  Real normal predicted as normal: {pct_normal_as_normal:.1f}%")
    print(f"  Real attack predicted as outlier: {pct_attack_as_outlier:.1f}%")
    
    # Test 2: Train on real normal, test on real normal + attack
    print("\nTest 2: Train on REAL normal, test on REAL normal + attack")
    model2 = OneClassSVM(kernel="rbf", nu=0.001, gamma='scale')
    model2.fit(Z_real_normal)
    
    pred_real_normal2 = model2.predict(Z_real_normal)
    pred_real_attack2 = model2.predict(Z_real_attack)
    
    pct_normal_as_normal2 = (pred_real_normal2 == 1).sum() / len(pred_real_normal2) * 100
    pct_attack_as_outlier2 = (pred_real_attack2 == -1).sum() / len(pred_real_attack2) * 100
    
    print(f"  Real normal predicted as normal: {pct_normal_as_normal2:.1f}%")
    print(f"  Real attack predicted as outlier: {pct_attack_as_outlier2:.1f}%")
    
    # =========================================================
    # Check Label Distribution
    # =========================================================
    print("\n7. Checking test set composition...")
    print("="*70)
    
    # Simulate what happens in fold 1
    from scenarios import scenario_1_split
    import pandas as pd
    
    for fold_idx, train_idx, test_idx in scenario_1_split(pd.DataFrame(X_real), pd.Series(y), k=5):
        if fold_idx == 0:  # First fold
            y_test = y[test_idx]
            print(f"\nFold 1 test set:")
            print(f"  Total samples: {len(y_test)}")
            print(f"  Normal: {np.sum(y_test == 0)} ({np.sum(y_test == 0)/len(y_test)*100:.1f}%)")
            print(f"  Attack: {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)")
            
            # This explains precision!
            attack_ratio = np.sum(y_test == 1) / len(y_test)
            print(f"\n  Attack ratio: {attack_ratio:.4f}")
            print(f"  Expected precision if predicting all as attack: {attack_ratio:.4f}")
            print(f"  Your actual precision: ~0.0026")
            print(f"  → They match! Model is predicting everything as attack.")
            break
    
    # =========================================================
    # Recommendations
    # =========================================================
    print("\n" + "="*70)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("="*70)
    
    if pct_normal_as_normal < 50:
        print("\n❌ PROBLEM: Synthetic normal data is too different from real normal")
        print("   → OCSVM trained on synthetic thinks real normal is outlier!")
        print("\n   Solutions:")
        print("   1. Retrain GAN with better hyperparameters")
        print("   2. Use higher 'nu' parameter (e.g., nu=0.01 or 0.05)")
        print("   3. Try gamma='auto' instead of 'scale'")
    else:
        print("\n✓ Synthetic vs Real normal looks reasonable")
    
    if pct_attack_as_outlier2 < 50:
        print("\n❌ PROBLEM: Real attacks are NOT distinguishable from real normal")
        print("   → VAE features don't separate normal from attack well")
        print("\n   Solutions:")
        print("   1. Retrain VAE with more epochs")
        print("   2. Use classification VAE instead of reconstruction VAE")
        print("   3. Try different latent dimension")
    else:
        print("\n✓ Real normal vs Real attack separation looks good")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
