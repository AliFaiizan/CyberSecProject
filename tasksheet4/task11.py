#!/usr/bin/env python3
"""
Task 1: GAN for Synthetic Data Generation (FINAL, BUG-FREE)
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from utils import load_data


# =================================================
# DISCRIMINATOR
# =================================================
class Discriminator(nn.Module):
    def __init__(self, dropout_p=0.2):
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

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        feat = x.squeeze(-1)
        x = F.relu(self.fc1(feat))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x), feat


# =================================================
# GENERATOR (FIXED)
# =================================================
class Generator(nn.Module):
    def __init__(self, M, zdim=64):
        super().__init__()
        self.init_len = 4

        self.fc = nn.Sequential(
            nn.Linear(zdim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256 * self.init_len), nn.LeakyReLU(0.2),
        )

        self.conv = nn.Sequential(
            nn.Upsample(2), nn.Conv1d(256, 256, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Upsample(2), nn.Conv1d(256, 256, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Upsample(2), nn.Conv1d(256, 128, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Upsample(2), nn.Conv1d(128, 128, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 64, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 32, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 16, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(16, 1, 3, 1, 1), nn.LeakyReLU(0.2),
        )

        # 🔑 compute output length dynamically (THE FIX)
        with torch.no_grad():
            dummy = torch.zeros(1, zdim)
            x = self.fc(dummy).view(1, 256, self.init_len)
            x = self.conv(x)
            conv_len = x.shape[-1]

        self.final = nn.Linear(conv_len, M)

    def forward(self, z):
        x = self.fc(z).view(z.size(0), 256, self.init_len)
        x = self.conv(x).squeeze(1)
        return self.final(x)


# =================================================
# TRAINING
# =================================================
def train_gan(X, M, device, epochs, batch_size, lr_g, lr_d, fm_weight, name):
    print(f"\nTraining {name} | Samples={len(X)}")

    G = Generator(M).to(device)
    D = Discriminator().to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    for epoch in range(1, epochs + 1):
        d_ls, g_ls = [], []

        for (real,) in loader:
            real = real.to(device)
            B = real.size(0)

            # ---- Discriminator ----
            z = torch.randn(B, 64, device=device)
            fake = G(z).unsqueeze(1).detach()

            d_real, _ = D(real)
            d_fake, _ = D(fake)

            d_loss = F.binary_cross_entropy(d_real, torch.full_like(d_real, 0.9)) + \
                     F.binary_cross_entropy(d_fake, torch.full_like(d_fake, 0.1))

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()
            d_ls.append(d_loss.item())

            # ---- Generator ----
            z = torch.randn(B, 64, device=device)
            fake = G(z).unsqueeze(1)

            d_fake, fake_feat = D(fake)
            _, real_feat = D(real)

            fm_loss = ((real_feat.mean(0) - fake_feat.mean(0))**2).mean() * fm_weight
            adv_loss = F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))

            g_loss = fm_loss + 0.1 * adv_loss

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()
            g_ls.append(g_loss.item())

        if epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch:>3}/{epochs} | D={np.mean(d_ls):.4f} | G={np.mean(g_ls):.4f}")

    return G


@torch.no_grad()
def generate(G, n, device):
    G.eval()
    out = []
    for i in range(0, n, 128):
        z = torch.randn(min(128, n - i), 64, device=device)
        out.append(G(z).cpu().numpy())
    return np.vstack(out)


# =================================================
# MAIN
# =================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dataset_base", default="../datasets/hai-22.04")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    X, y = load_data(
        [f"{args.dataset_base}/train1.csv"],
        [f"{args.dataset_base}/test1.csv"]
    )
    y = y.astype(int)
    M = X.shape[1]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs = np.clip(Xs, -5, 5) / 5.0

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    Xn, Xa = Xs[y == 0], Xs[y == 1]

    # ---- NORMAL GAN ----
    G_normal = train_gan(
        Xn, M, device,
        epochs=args.epochs,
        batch_size=128,
        lr_g=1e-4,
        lr_d=2e-4,
        fm_weight=10,
        name="GAN_NORMAL"
    )

    sn = generate(G_normal, len(Xn), device)
    np.save("synthetic_normal.npy", scaler.inverse_transform(sn * 5))
    torch.save(G_normal.state_dict(), "G_normal.pt")

    # ---- ATTACK GAN ----
    G_attack = train_gan(
        Xa, M, device,
        epochs=min(30, args.epochs),
        batch_size=min(32, len(Xa)),
        lr_g=1e-4,
        lr_d=2e-4,
        fm_weight=10,
        name="GAN_ATTACK"
    )

    sa = generate(G_attack, len(Xa), device)
    np.save("synthetic_attack.npy", scaler.inverse_transform(sa * 5))
    torch.save(G_attack.state_dict(), "G_attack.pt")

    print("\nTASK 1 COMPLETE")


if __name__ == "__main__":
    main()

