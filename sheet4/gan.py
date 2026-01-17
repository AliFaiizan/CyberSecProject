#!/usr/bin/env python3
"""
Task 3: GAN for Synthetic Data Generation
PROPER NORMALIZATION VERSION - No artificial clipping

Key change: Use StandardScaler normalization WITHOUT clipping to [-1, 1]
This preserves the original data distribution better.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle
import argparse
import json
from itertools import product
from glob import glob

from utils import load_data


# =================================================
# DISCRIMINATOR (Same as before)
# =================================================
class Discriminator(nn.Module):
    """Task requirement (c): 9 Conv1d + 3 FC"""
    def __init__(self, dropout_p=0.3, fc_neurons=(512, 128)):
        super().__init__()
        
        ch = [1, 16, 32, 64, 64, 128, 128, 256, 256, 256]
        layers = []
        
        for i in range(9):
            layers.append(nn.Conv1d(ch[i], ch[i+1], 3, padding=1))
            layers.append(nn.ReLU())
            if i not in (0, 8):
                layers.append(nn.Dropout(dropout_p))
        
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(ch[-1], fc_neurons[0])
        self.fc2 = nn.Linear(fc_neurons[0], fc_neurons[1])
        self.fc3 = nn.Linear(fc_neurons[1], 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        fc1_input = x.squeeze(-1)  # Features for loss
        
        x = torch.relu(self.fc1(fc1_input))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return torch.sigmoid(x), fc1_input


# =================================================
# GENERATOR (Same as before)
# =================================================
class Generator(nn.Module):
    """Task requirement (d): 3 FC + 9 Conv with upsampling"""
    def __init__(self, M, zdim=64, fc_neurons=(256, 512, 256)):
        super().__init__()
        self.M = M
        self.zdim = zdim
        self.init_len = 4
        
        self.fc1 = nn.Linear(zdim, fc_neurons[0])
        self.fc2 = nn.Linear(fc_neurons[0], fc_neurons[1])
        self.fc3 = nn.Linear(fc_neurons[1], fc_neurons[2] * self.init_len)
        
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv1d(fc_neurons[2], 256, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2), nn.Conv1d(256, 256, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2), nn.Conv1d(256, 128, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2), nn.Conv1d(128, 128, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 64, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 32, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 16, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(16, 1, 3, 1, 1), nn.LeakyReLU(0.2),
        )
        
        self.final = nn.Linear(64, M)

    def forward(self, z):
        x = F.leaky_relu(self.fc1(z), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = x.view(z.size(0), -1, self.init_len)
        x = self.conv(x)
        x = x.squeeze(1)
        x = self.final(x)
        return x


# =================================================
# TRAINING
# =================================================
def train_gan(X, M, device, epochs=50, batch_size=128, 
              lr_g=2e-4, lr_d=5e-5, dropout_p=0.3, fm_weight=100,
              fc_neurons_g=(256, 512, 256), fc_neurons_d=(512, 128),
              name="GAN", verbose=True):
    """Train GAN - Task requirement (e): Feature matching loss"""
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training {name}")
        print(f"Samples: {len(X)}, M: {M}, Epochs: {epochs}")
        print(f"Data range: [{X.min():.4f}, {X.max():.4f}]")
        print(f"Data mean: {X.mean():.4f}, std: {X.std():.4f}")
        print(f"{'='*70}")
    
    G = Generator(M, zdim=64, fc_neurons=fc_neurons_g).to(device)
    D = Discriminator(dropout_p, fc_neurons=fc_neurons_d).to(device)
    
    g_opt = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    best_g_loss = float('inf')
    best_G_state = None
    
    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []
        
        for (real,) in loader:
            real = real.to(device)
            B = real.size(0)
            
            # ==========================================
            # Train Discriminator
            # ==========================================
            d_opt.zero_grad()
            
            z = torch.randn(B, 64, device=device)
            fake = G(z).unsqueeze(1).detach()
            
            d_real, _ = D(real)
            d_fake, _ = D(fake)
            
            # Standard GAN loss with label smoothing
            d_loss = -(torch.log(d_real * 0.9 + 0.05 + 1e-8).mean() + 
                       torch.log(1 - d_fake + 1e-8).mean())
            
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            d_opt.step()
            d_losses.append(d_loss.item())
            
            # ==========================================
            # Train Generator (2x per discriminator)
            # ==========================================
            for _ in range(2):
                g_opt.zero_grad()
                
                z = torch.randn(B, 64, device=device)
                fake = G(z).unsqueeze(1)
                
                d_fake_score, fake_feat = D(fake)
                _, real_feat = D(real)
                
                # Task requirement (e): Feature matching on FC1 input
                # Match mean and std of features
                real_mean = real_feat.mean(dim=0)
                fake_mean = fake_feat.mean(dim=0)
                real_std = real_feat.std(dim=0) + 1e-8
                fake_std = fake_feat.std(dim=0) + 1e-8
                
                fm_loss = (torch.mean((real_mean - fake_mean) ** 2) + 
                          torch.mean((real_std - fake_std) ** 2)) * fm_weight
                
                # Adversarial loss
                adv_loss = -torch.log(d_fake_score + 1e-8).mean()
                
                g_loss = fm_loss + 0.1 * adv_loss
                
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                g_opt.step()
                g_losses.append(g_loss.item())
        
        avg_g_loss = np.mean(g_losses)
        
        # Save best model
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            best_G_state = G.state_dict().copy()
        
        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch:>3}/{epochs} | "
                  f"D: {np.mean(d_losses):.4f} | "
                  f"G: {avg_g_loss:.4f}")
    
    # Load best model
    if best_G_state is not None:
        G.load_state_dict(best_G_state)
        print(f"\n✓ Loaded best model (G loss: {best_g_loss:.4f})")
    
    metrics = {
        'final_d_loss': np.mean(d_losses),
        'final_g_loss': avg_g_loss,
        'best_g_loss': best_g_loss
    }
    
    return G, metrics


@torch.no_grad()
def generate(G, n, M, device, batch_size=128):
    """Generate n synthetic samples"""
    G.eval()
    out = []
    for i in range(0, n, batch_size):
        b = min(batch_size, n - i)
        z = torch.randn(b, 64, device=device)
        generated = G(z).cpu().numpy()
        
        # Verify shape
        assert generated.shape[1] == M, f"Shape mismatch! Expected M={M}, got {generated.shape[1]}"
        out.append(generated)
    
    result = np.vstack(out)
    assert result.shape == (n, M), f"Final shape mismatch! Expected ({n}, {M}), got {result.shape}"
    
    return result


# =================================================
# HYPERPARAMETER TUNING (Task requirement f)
# =================================================
def hyperparameter_tuning(X, M, device, param_ranges, epochs_per_trial=20):
    """Comprehensive hyperparameter tuning"""
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING")
    print("="*70)
    
    n_samples = min(len(X), 50000)
    X_tune = X[:n_samples]
    
    results = []
    configs = list(product(*param_ranges.values()))
    
    print(f"Testing {len(configs)} configurations...\n")
    
    for i, values in enumerate(configs, 1):
        config = dict(zip(param_ranges.keys(), values))
        print(f"[{i}/{len(configs)}] Testing: {config}")
        
        try:
            _, metrics = train_gan(
                X_tune, M, device,
                epochs=epochs_per_trial,
                name=f"Tune_{i}",
                verbose=False,
                **config
            )
            
            score = metrics['best_g_loss']
            results.append({**config, **metrics, 'score': score})
            
            print(f"  → Score: {score:.4f}")
            
        except Exception as e:
            print(f"  → FAILED: {e}")
            continue
    
    # Save results
    with open('tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    best = min(results, key=lambda x: x['score'])
    
    print("\n" + "="*70)
    print("BEST CONFIGURATION:")
    print("="*70)
    for k, v in best.items():
        if k in param_ranges:
            print(f"  {k}: {v}")
    print(f"\nBest Score: {best['score']:.4f}")
    print("="*70)
    
    return {k: v for k, v in best.items() if k in param_ranges}


# =================================================
# MAIN
# =================================================
def main():
    parser = argparse.ArgumentParser(description='Task 3: GAN for Synthetic Data')
    parser.add_argument('--M', type=int, required=True,
                        help='Number of physical readings per timestep')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--tune', action='store_true',
                        help='Run hyperparameter tuning')
    parser.add_argument('--dataset_base', default='../datasets/hai-22.04',
                        help='Dataset directory')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"M: {args.M}\n")
    
    # Load data
    train_files = sorted(glob(f"{args.dataset_base}/train1.csv"))
    test_files = sorted(glob(f"{args.dataset_base}/test1.csv"))
    
    X, y = load_data(train_files, test_files)
    y = y.astype(int)
    
    # Select M features
    if X.shape[1] != args.M:
        if X.shape[1] < args.M:
            raise ValueError(f"M={args.M} exceeds available features ({X.shape[1]})")
        print(f"Using first {args.M} of {X.shape[1]} features")
        X = X[:, :args.M]
    
    print(f"Data: {len(X)} samples × {X.shape[1]} features")
    print(f"Normal: {sum(y==0)}, Attack: {sum(y==1)}")
    
    # ==========================================
    # PROPER NORMALIZATION
    # ==========================================
    # CRITICAL: Just StandardScaler, NO clipping!
    # This preserves the distribution shape
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Save scaler for denormalization
    with open('gan_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nAfter StandardScaler normalization:")
    print(f"  Mean: {X_normalized.mean():.6f} (should be ~0)")
    print(f"  Std:  {X_normalized.std():.6f} (should be ~1)")
    print(f"  Range: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")
    
    # Split by label
    X_normal = X_normalized[y == 0]
    X_attack = X_normalized[y == 1]
    
    # Hyperparameter tuning
    if args.tune:
        param_ranges = {
            'lr_g': [1e-4, 2e-4],
            'lr_d': [2e-5, 5e-5],
            'dropout_p': [0.3],
            'fm_weight': [10, 50, 100],
            'batch_size': [128, 256],
            'fc_neurons_g': [(256, 512, 256)],
            'fc_neurons_d': [(512, 128)],
        }
        best_config = hyperparameter_tuning(X_normal, args.M, device, param_ranges)
    else:
        best_config = {
            'lr_g': 1e-4,
            'lr_d': 5e-5,
            'dropout_p': 0.3,
            'fm_weight': 50,
            'batch_size': 256,
            'fc_neurons_g': (256, 512, 256),
            'fc_neurons_d': (512, 128),
        }
    
    # ==========================================
    # TRAIN GAN FOR NORMAL DATA
    # ==========================================
    print("\n" + "="*70)
    print("TRAINING GAN #1 (NORMAL DATA)")
    print("="*70)
    
    G_normal, _ = train_gan(
        X_normal, args.M, device,
        epochs=args.epochs,
        name="GAN_normal",
        **best_config
    )
    
    # Generate synthetic data
    synth_normal_normalized = generate(G_normal, len(X_normal), args.M, device)
    
    # CRITICAL: Denormalize using inverse_transform
    # No multiplication/division by 5!
    synth_normal = scaler.inverse_transform(synth_normal_normalized)
    
    # Save
    np.save("synthetic_normal.npy", synth_normal)
    torch.save(G_normal.state_dict(), "G_normal.pt")
    
    print(f"\n✓ Generated synthetic_normal.npy: {synth_normal.shape}")
    print(f"  Synthetic mean: {synth_normal_normalized.mean():.4f}")
    print(f"  Synthetic std:  {synth_normal_normalized.std():.4f}")
    print(f"  Real mean:      {X_normal.mean():.4f}")
    print(f"  Real std:       {X_normal.std():.4f}")
    
    # ==========================================
    # TRAIN GAN FOR ATTACK DATA
    # ==========================================
    print("\n" + "="*70)
    print("TRAINING GAN #2 (ATTACK DATA)")
    print("="*70)
    
    attack_config = best_config.copy()
    attack_config['batch_size'] = min(64, len(X_attack) // 4)
    
    G_attack, _ = train_gan(
        X_attack, args.M, device,
        epochs=args.epochs,
        name="GAN_attack",
        **attack_config
    )
    
    synth_attack_normalized = generate(G_attack, len(X_attack), args.M, device)
    synth_attack = scaler.inverse_transform(synth_attack_normalized)
    
    np.save("synthetic_attack.npy", synth_attack)
    torch.save(G_attack.state_dict(), "G_attack.pt")
    
    print(f"\n✓ Generated synthetic_attack.npy: {synth_attack.shape}")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  - synthetic_normal.npy ({synth_normal.shape[0]} × {synth_normal.shape[1]})")
    print(f"  - synthetic_attack.npy ({synth_attack.shape[0]} × {synth_attack.shape[1]})")
    print(f"  - G_normal.pt, G_attack.pt")
    print(f"  - gan_scaler.pkl (StandardScaler only, no clipping)")
    print("\n✓ All data properly normalized/denormalized")
    print("="*70)


if __name__ == "__main__":
    main()
