#!/usr/bin/env python3
"""
Task 3: GAN for Synthetic Data Generation

Complete implementation with:
- 2 GANs: normal and attack
- Hyperparameter tuning
- All task requirements met
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle
import argparse
import json
from itertools import product

from utils import load_data

# =================================================
# DISCRIMINATOR
# =================================================
class Discriminator(nn.Module):
    """
    Task requirement (c):
    - 9 Conv1d + ReLU
    - Dropout after all except first & last
    - 3 FC + ReLU
    - Output [0,1)
    """
    def __init__(self, dropout_p=0.3):
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
        
        self.fc1 = nn.Linear(ch[-1], 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        fc1_input = x.squeeze(-1)  # Features for matching
        
        x = torch.relu(self.fc1(fc1_input))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return torch.sigmoid(x), fc1_input


# =================================================
# GENERATOR
# =================================================
class Generator(nn.Module):
    """
    Task requirement (d):
    - 3 FC (LeakyReLU)
    - 9 Conv (LeakyReLU)
    - First 4 conv with upsampling
    """
    def __init__(self, M, zdim=64):
        super().__init__()
        self.M = M
        self.zdim = zdim
        self.init_len = 4
        
        # 3 FC layers (LeakyReLU)
        self.fc = nn.Sequential(
            nn.Linear(zdim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256 * self.init_len), nn.LeakyReLU(0.2),
        )
        
        # 9 conv layers (LeakyReLU), first 4 with upsampling
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv1d(256, 256, 3, 1, 1), nn.LeakyReLU(0.2),
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
        x = self.fc(z).view(z.size(0), 256, self.init_len)
        x = self.conv(x)
        x = x.squeeze(1)
        x = self.final(x)
        return x


# =================================================
# TRAINING
# =================================================
def train_gan(X, M, device, epochs, batch_size, lr_g, lr_d, 
              dropout_p, fm_weight, name="GAN", verbose=True):
    """
    Train GAN with given hyperparameters
    
    Returns:
        G: trained generator
        metrics: dict with final losses for tuning
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"Samples: {len(X)}, Epochs: {epochs}")
        print(f"Hyperparams: lr_g={lr_g}, lr_d={lr_d}, dropout={dropout_p}, fm_weight={fm_weight}")
        print(f"{'='*60}")
    
    G = Generator(M).to(device)
    D = Discriminator(dropout_p).to(device)
    
    g_opt = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    d_loss_history, g_loss_history = [], []
    
    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []
        
        for (real,) in loader:
            real = real.to(device)
            B = real.size(0)
            
            # Train Discriminator
            z = torch.randn(B, 64, device=device)
            fake = G(z).unsqueeze(1).detach()
            
            d_real, _ = D(real)
            d_fake, _ = D(fake)
            
            d_loss = -(torch.log(d_real * 0.9 + 0.05 + 1e-8).mean() + 
                       torch.log(1 - d_fake + 1e-8).mean())
            
            d_opt.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            d_opt.step()
            d_losses.append(d_loss.item())
            
            # Train Generator (2x per D)
            for _ in range(2):
                z = torch.randn(B, 64, device=device)
                fake = G(z).unsqueeze(1)
                
                d_fake_score, fake_feat = D(fake)
                _, real_feat = D(real)
                
                # Task requirement (e): Feature matching on FC1 input
                # Normalize features to prevent vanishing gradients
                real_feat_norm = real_feat / (real_feat.std(dim=0, keepdim=True) + 1e-8)
                fake_feat_norm = fake_feat / (fake_feat.std(dim=0, keepdim=True) + 1e-8)
                
                fm_loss = torch.mean((real_feat_norm - fake_feat_norm) ** 2) * fm_weight
                adv_loss = -torch.log(d_fake_score + 1e-8).mean()
                g_loss = fm_loss + 0.1 * adv_loss
                
                g_opt.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                g_opt.step()
                g_losses.append(g_loss.item())
        
        d_loss_history.append(np.mean(d_losses))
        g_loss_history.append(np.mean(g_losses))
        
        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch:>3}/{epochs} | "
                  f"D: {d_loss_history[-1]:.4f} | "
                  f"G: {g_loss_history[-1]:.4f}")
    
    # Return metrics for tuning
    metrics = {
        'final_d_loss': d_loss_history[-1],
        'final_g_loss': g_loss_history[-1],
        'avg_d_loss': np.mean(d_loss_history[-10:]),
        'avg_g_loss': np.mean(g_loss_history[-10:]),
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
        out.append(G(z).cpu().numpy())
    return np.vstack(out)


# =================================================
# HYPERPARAMETER TUNING
# =================================================
def hyperparameter_tuning(X, M, device, epochs_per_trial=20):
    """
    Task requirement (f): Hyperparameter tuning
    
    Tests different combinations and returns best config
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Parameter grid (small for speed)
    param_grid = {
        'lr_g': [1e-4, 2e-4],
        'lr_d': [2e-5, 5e-5],
        'dropout_p': [0.2, 0.3],
        'fm_weight': [50, 100],
        'batch_size': [128],
    }
    
    # Use subset of data for tuning (faster)
    n_samples = min(len(X), 50000)
    X_tune = X[:n_samples]
    
    results = []
    configs = list(product(*param_grid.values()))
    
    for i, values in enumerate(configs, 1):
        config = dict(zip(param_grid.keys(), values))
        print(f"\n[{i}/{len(configs)}] Testing: {config}")
        
        try:
            _, metrics = train_gan(
                X_tune, M, device,
                epochs=epochs_per_trial,
                name=f"Tuning {i}",
                verbose=False,
                **config
            )
            
            # Score: lower G loss is better, D loss should be ~0.5
            score = metrics['avg_g_loss'] + abs(metrics['avg_d_loss'] - 0.5) * 10
            
            results.append({
                **config,
                **metrics,
                'score': score
            })
            
            print(f"  D: {metrics['avg_d_loss']:.4f} | "
                  f"G: {metrics['avg_g_loss']:.4f} | "
                  f"Score: {score:.4f}")
            
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
    
    # Save results
    with open('tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Get best config
    best = min(results, key=lambda x: x['score'])
    
    print("\n" + "="*60)
    print("BEST CONFIGURATION:")
    print("="*60)
    for k, v in best.items():
        if k != 'score':
            print(f"  {k}: {v}")
    print("="*60)
    
    return {k: v for k, v in best.items() if k in param_grid}


# =================================================
# MAIN
# =================================================
def main():
    parser = argparse.ArgumentParser(description='Task 3: GAN Training')
    parser.add_argument('--M', type=int, default=None, 
                        help='Number of physical readings (auto-detected if not provided)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--tune', action='store_true',
                        help='Run hyperparameter tuning first')
    parser.add_argument('--dataset_base', default='../datasets/hai-22.04',
                        help='Dataset directory')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}\n")
    
    # Load data
    train_files = [f"{args.dataset_base}/train1.csv"]
    test_files = [f"{args.dataset_base}/test1.csv"]
    
    X, y = load_data(train_files, test_files)
    y = y.astype(int)
    M = args.M if args.M is not None else X.shape[1]
    
    print(f"Data: {len(X)} samples, {M} features")
    print(f"Normal: {sum(y==0)}, Attack: {sum(y==1)}\n")
    
    # Normalize
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    X_normalized = np.clip(X_normalized, -5, 5) / 5.0
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Split by attack label
    X_normal = X_normalized[y == 0]
    X_attack = X_normalized[y == 1]
    
    # Hyperparameter tuning (optional)
    if args.tune:
        best_config = hyperparameter_tuning(X_normal, M, device)
    else:
        # Default good config
        best_config = {
            'lr_g': 2e-4,
            'lr_d': 5e-5,
            'dropout_p': 0.3,
            'fm_weight': 100,
            'batch_size': 128,
        }
    
    # ==========================================
    # GAN 1: NORMAL DATA
    # ==========================================
    print("\n" + "="*60)
    print("TRAINING GAN #1 (NORMAL DATA)")
    print("="*60)
    
    G_normal, _ = train_gan(
        X_normal, M, device,
        epochs=args.epochs,
        name="GAN_normal",
        **best_config
    )
    
    synth_normal_norm = generate(G_normal, len(X_normal), M, device)
    synth_normal = scaler.inverse_transform(synth_normal_norm * 5.0)
    
    np.save("synthetic_normal.npy", synth_normal)
    torch.save(G_normal.state_dict(), "G_normal.pt")
    print(f"\n synthetic_normal.npy: {synth_normal.shape}")
    
    # ==========================================
    # GAN 2: ATTACK DATA
    # ==========================================
    print("\n" + "="*60)
    print("TRAINING GAN #2 (ATTACK DATA)")
    print("="*60)
    
    # Use smaller batch for attack data (fewer samples)
    attack_config = best_config.copy()
    attack_config['batch_size'] = min(64, len(X_attack) // 4)
    
    G_attack, _ = train_gan(
        X_attack, M, device,
        epochs=args.epochs,
        name="GAN_attack",
        **attack_config
    )
    
    synth_attack_norm = generate(G_attack, len(X_attack), M, device)
    synth_attack = scaler.inverse_transform(synth_attack_norm * 5.0)
    
    np.save("synthetic_attack.npy", synth_attack)
    torch.save(G_attack.state_dict(), "G_attack.pt")
    print(f"\n synthetic_attack.npy: {synth_attack.shape}")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "="*60)
    print("TASK 1 COMPLETE")
    print("="*60)
    print(f"Generated files:")
    print(f"  - synthetic_normal.npy ({synth_normal.shape[0]} samples)")
    print(f"  - synthetic_attack.npy ({synth_attack.shape[0]} samples)")
    print(f"  - G_normal.pt, G_attack.pt (model weights)")
    print(f"  - scaler.pkl (for denormalization)")
    if args.tune:
        print(f"  - tuning_results.json (hyperparameter search)")
    print("\nThese will be used in Task 2 for the 3 evaluation scenarios.")
    print("="*60)


if __name__ == "__main__":
    main()
