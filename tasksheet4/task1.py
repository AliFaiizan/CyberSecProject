#!/usr/bin/env python3
"""
Task 1 (Task 3): GAN for Synthetic Data Generation
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle

from utils import load_data


# =================================================
# CONFIG
# =================================================
DATASET_BASE = "../datasets/hai-22.04"

TRAIN_FILES = [f"{DATASET_BASE}/train1.csv"]
TEST_FILES  = [f"{DATASET_BASE}/test1.csv"]

EPOCHS = 10
BATCH_SIZE = 128

# IMPORTANT: prevents CUDA OOM during generation on 4GB GPUs
GEN_BATCH_SIZE = 128

# # Optional: reduce memory during training if you still OOM
USE_AMP = False  # mixed precision (only active on CUDA)


# =================================================
# DISCRIMINATOR
# =================================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        ch = [1, 16, 32, 64, 64, 128, 128, 256, 256, 256]
        layers = []

        for i in range(9):
            layers.append(nn.Conv1d(ch[i], ch[i + 1], 3, padding=1))
            layers.append(nn.ReLU())
            if i not in (0, 8):
                layers.append(nn.Dropout(0.3))

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(ch[-1], 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.squeeze(-1)

        features = self.relu(self.fc1(x))
        x = self.relu(self.fc2(features))
        x = self.fc3(x)

        return self.sigmoid(x), features


# =================================================
# GENERATOR
# =================================================
class Generator(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.zdim = M
        self.init_len = 4

        self.fc = nn.Sequential(
            nn.Linear(self.zdim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256 * self.init_len), nn.LeakyReLU(0.2),
        )

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
        x = self.conv(x)  # [B, 1, 64]
        x = x.squeeze(1)  # [B, 64]
        x = self.final(x)  # [B, M]
        return x
    
# =================================================
# TRAINING
# =================================================
def train_gan(X, M, device, epochs, batch_size, name):
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"Samples: {len(X)}, Epochs: {epochs}")
    print(f"{'='*60}")

    G = Generator(M).to(device)
    D = Discriminator().to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=5e-6, betas=(0.5, 0.999))

    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device == "cuda"))

    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []
        d_updates = 0

        for batch_idx, (real,) in enumerate(loader):
            real = real.to(device, non_blocking=True)
            B = real.size(0)

            # ===== Train Generator 3 times =====
            for _ in range(3):
                z = torch.randn(B, M, device=device)

                with torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
                    fake = G(z).unsqueeze(1)
                    _, rf = D(real)
                    _, ff = D(fake)
                    g_loss = torch.mean(torch.abs(rf - ff))  # feature matching (L1)

                g_opt.zero_grad(set_to_none=True)
                scaler.scale(g_loss).backward()
                scaler.unscale_(g_opt)
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                scaler.step(g_opt)
                scaler.update()

                g_losses.append(float(g_loss.detach().cpu()))

            # ===== Train Discriminator 1 time (every 3rd batch) =====
            if batch_idx % 3 == 0:
                z = torch.randn(B, M, device=device)
                with torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
                    fake = G(z).unsqueeze(1).detach()
                    d_real, _ = D(real)
                    d_fake, _ = D(fake)
                    d_loss = -torch.mean(
                        torch.log(d_real * 0.9 + 0.05 + 1e-8) +
                        torch.log(1 - d_fake + 1e-8)
                    )

                d_opt.zero_grad(set_to_none=True)
                scaler.scale(d_loss).backward()
                scaler.unscale_(d_opt)
                torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                scaler.step(d_opt)
                scaler.update()

                d_losses.append(float(d_loss.detach().cpu()))
                d_updates += 1

        if epoch % max(1, epochs // 5) == 0:
            print(
                f"Epoch {epoch:>3}/{epochs} | "
                f"D: {np.mean(d_losses) if d_losses else 0.0:.4f} ({d_updates} updates) | "
                f"G: {np.mean(g_losses):.4f}"
            )

    return G


@torch.no_grad()
def generate(G, n, M, device, batch_size=GEN_BATCH_SIZE):
    """
    Batched generation to avoid CUDA OOM.
    Returns: [n, M] numpy array
    """
    G.eval()
    out = []

    for i in range(0, n, batch_size):
        b = min(batch_size, n - i)
        z = torch.randn(b, M, device=device)
        fake = G(z).float().cpu().numpy()
        out.append(fake)

    return np.vstack(out)


# =================================================
# MAIN
# =================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}\n")

    X, y = load_data(TRAIN_FILES, TEST_FILES)
    y = y.astype(int)
    M = X.shape[1]

    print(f"Data: {len(X)} samples, {M} features")
    print(f"Normal: {int((y==0).sum())}, Attack: {int((y==1).sum())}\n")

    # Normalize
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    Xn = np.clip(Xn, -5, 5) / 5.0

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    X_normal = Xn[y == 0]
    X_attack = Xn[y == 1]

    # GAN 1: Normal
    G1 = train_gan(X_normal, M, device, EPOCHS, BATCH_SIZE, "GAN #1 (Normal)")
    torch.save(G1.state_dict(), "G_normal.pt")
    if device == "cuda":
        torch.cuda.empty_cache()

    synth_normal_norm = generate(G1, len(X_normal), M, device)
    synth_normal = scaler.inverse_transform(synth_normal_norm * 5.0)
    np.save("synthetic_normal.npy", synth_normal)
    print(f"\nsynthetic_normal.npy: {synth_normal.shape}")

    # GAN 2: Attack
    bs = min(BATCH_SIZE, max(16, len(X_attack) // 4))
    G2 = train_gan(X_attack, M, device, EPOCHS, bs, "GAN #2 (Attack)")
    torch.save(G2.state_dict(), "G_attack.pt")
    if device == "cuda":
        torch.cuda.empty_cache()

    synth_attack_norm = generate(G2, len(X_attack), M, device)
    synth_attack = scaler.inverse_transform(synth_attack_norm * 5.0)
    np.save("synthetic_attack.npy", synth_attack)
    print(f"synthetic_attack.npy: {synth_attack.shape}\n")


if __name__ == "__main__":
    # Helps CUDA allocator fragmentation on small GPUs (optional)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
