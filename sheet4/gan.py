#!/usr/bin/env python3
"""
Task Sheet 4 - Task 3: GAN for Synthetic Dataset Generation (STRICT ARCHITECTURE)

Meets requirements:
- D: 9x Conv1d + ReLU, Dropout after all convs except 1st/last, then 3x FC + ReLU, outputs prob in (0,1)
- G: noise -> 3x FC (LeakyReLU) -> 9x Conv1d (LeakyReLU) with Upsample before first 4 convs
- Custom loss anchored at D feature vector = input to D's first FC layer (requirement e)
- Hyperparameter tuning option (--tune) without WGAN

Quality goal:
- Better per-feature mean/std (output moment matching)
- Better correlation structure (output covariance + feature covariance warmup)
- Stable training (BCEWithLogitsLoss, grad clipping, best-snapshot saved after warmup)

Outputs:
- synthetic_normal.npy (RAW space)
- (optional) synthetic_attack.npy
- G_normal.pt / G_attack.pt
- gan_scaler.pkl
"""

import argparse
import json
import pickle
from glob import glob
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from utils import load_data  # must return X (N,F), y (N,)


# =========================
# Discriminator (Requirement c)
# =========================
class Discriminator(nn.Module):
    def __init__(self, dropout_p=0.3, fc_neurons=(512, 128)):
        super().__init__()
        ch = [1, 16, 32, 64, 64, 128, 128, 256, 256, 256]  # 9 convs

        layers = []
        for i in range(9):
            layers.append(nn.Conv1d(ch[i], ch[i + 1], kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            if i not in (0, 8):  # dropout after all convs except first & last
                layers.append(nn.Dropout(dropout_p))

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(ch[-1], fc_neurons[0])
        self.fc2 = nn.Linear(fc_neurons[0], fc_neurons[1])
        self.fc3 = nn.Linear(fc_neurons[1], 1)  # logits

    def forward(self, x):
        """
        x: (B,1,M)
        returns:
          prob:  (B,1) in (0,1)
          feat:  (B,C) = input to fc1 (anchor for requirement e)
          logits:(B,1)
        """
        x = self.conv(x)
        x = self.pool(x)        # (B,C,1)
        feat = x.squeeze(-1)    # (B,C)
        h = F.relu(self.fc1(feat), inplace=True)
        h = F.relu(self.fc2(h), inplace=True)
        logits = self.fc3(h)
        prob = torch.sigmoid(logits)
        return prob, feat, logits


# =========================
# Generator (Requirement d)
# =========================
class Generator(nn.Module):
    def __init__(self, M, zdim=64, fc_neurons=(256, 512, 256)):
        super().__init__()
        self.M = M
        self.zdim = zdim
        self.init_len = 4  # after 4 upsamples -> 64

        self.fc1 = nn.Linear(zdim, fc_neurons[0])
        self.fc2 = nn.Linear(fc_neurons[0], fc_neurons[1])
        self.fc3 = nn.Linear(fc_neurons[1], fc_neurons[2] * self.init_len)

        # 9 conv layers; first 4 preceded by upsampling
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv1d(fc_neurons[2], 256, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv1d(256, 256, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv1d(256, 128, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv1d(128, 128, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 64, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 64, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 16, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 1, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
        )

        # conv output length = 64; map to M readings
        self.final = nn.Linear(64, M)

    def forward(self, z):
        x = F.leaky_relu(self.fc1(z), 0.2, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc3(x), 0.2, inplace=True)

        x = x.view(z.size(0), -1, self.init_len)  # (B,C,4)
        x = self.conv(x)                          # (B,1,64)
        x = x.squeeze(1)                          # (B,64)
        x = self.final(x)                         # (B,M)
        return x


# =========================
# Helpers: covariance + normalized losses
# =========================
def batch_cov(x: torch.Tensor) -> torch.Tensor:
    """x: (B,D) -> cov: (D,D)"""
    B = x.size(0)
    if B <= 1:
        return torch.zeros((x.size(1), x.size(1)), device=x.device, dtype=x.dtype)
    xc = x - x.mean(dim=0, keepdim=True)
    return (xc.t() @ xc) / (B - 1)


def normalized_cov_mse(fake_cov: torch.Tensor, real_cov: torch.Tensor) -> torch.Tensor:
    """Scale-invariant cov loss."""
    den = real_cov.pow(2).mean().detach() + 1e-8
    return F.mse_loss(fake_cov, real_cov) / den


# =========================
# Losses
# =========================
def feature_match_loss(real_feat, fake_feat, w_meanstd=10.0, w_cov=0.1):
    """
    Requirement (e) anchored at D's fc1 input features.
    Matches mean/std and (normalized) covariance.
    """
    rm = real_feat.mean(dim=0)
    fm = fake_feat.mean(dim=0)
    rs = real_feat.std(dim=0) + 1e-8
    fs = fake_feat.std(dim=0) + 1e-8

    loss_mean = F.mse_loss(fm, rm)
    loss_std = F.mse_loss(fs, rs)

    rc = batch_cov(real_feat)
    fc = batch_cov(fake_feat)
    loss_cov = normalized_cov_mse(fc, rc)

    return w_meanstd * (loss_mean + loss_std) + w_cov * loss_cov


def output_moment_loss(real_x, fake_x, w_out_mom=1.0):
    """Directly matches per-feature mean/std in output space (B,M)."""
    if w_out_mom <= 0:
        return fake_x.new_tensor(0.0)
    rm = real_x.mean(dim=0)
    fm = fake_x.mean(dim=0)
    rs = real_x.std(dim=0) + 1e-8
    fs = fake_x.std(dim=0) + 1e-8
    return w_out_mom * (F.mse_loss(fm, rm) + F.mse_loss(fs, rs))


def output_cov_loss(real_x, fake_x, w_out_cov=0.05):
    """Matches covariance across M (helps preserve correlation)."""
    if w_out_cov <= 0:
        return fake_x.new_tensor(0.0)
    rc = batch_cov(real_x)
    fc = batch_cov(fake_x)
    return w_out_cov * normalized_cov_mse(fc, rc)


# =========================
# Training
# =========================
def train_gan(
    X_z, M, device,
    epochs=50,
    batch_size=256,
    lr_g=1e-4,
    lr_d=2e-5,
    dropout_p=0.3,
    w_fm_meanstd=10.0,
    w_fm_cov=0.1,
    fm_cov_warmup=10,
    w_out_mom=1.0,
    w_out_cov=0.05,
    adv_weight=0.2,
    g_steps=2,
    clip_z=None,
    name="GAN",
    verbose=True,
):
    if len(X_z) < 2:
        raise ValueError(f"{name}: not enough samples: {len(X_z)}")

    X_train = np.clip(X_z, -clip_z, clip_z) if (clip_z is not None) else X_z

    if verbose:
        print(f"\n{'='*70}")
        print(f"Training {name}")
        print(f"Samples: {len(X_train)}, M: {M}, Epochs: {epochs}, batch: {batch_size}")
        print(f"Z range: [{X_train.min():.4f}, {X_train.max():.4f}] | mean={X_train.mean():.4f} std={X_train.std():.4f}")
        print(f"FM: mean+std={w_fm_meanstd} | cov={w_fm_cov} (warmup {fm_cov_warmup} ep)")
        print(f"OUT: mom={w_out_mom} | cov={w_out_cov} | ADV: adv_weight={adv_weight} | g_steps={g_steps}")
        print(f"LR: lr_d={lr_d} lr_g={lr_g} | dropout={dropout_p}")
        print(f"{'='*70}")

    G = Generator(M, zdim=64).to(device)
    D = Discriminator(dropout_p=dropout_p).to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()

    ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1))  # (N,1,M)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    best_score = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []

        w_cov_now = w_fm_cov if epoch > fm_cov_warmup else 0.0

        for (real_1m,) in loader:
            real_1m = real_1m.to(device)   # (B,1,M)
            real_x = real_1m.squeeze(1)    # (B,M)
            B = real_1m.size(0)

            # ---- D step ----
            D.train()
            d_opt.zero_grad(set_to_none=True)

            z = torch.randn(B, 64, device=device)
            fake_x = G(z)
            if clip_z is not None:
                fake_x = torch.clamp(fake_x, -clip_z, clip_z)
            fake_1m = fake_x.unsqueeze(1).detach()

            _, _, real_logits = D(real_1m)
            _, _, fake_logits = D(fake_1m)

            real_targets = torch.full_like(real_logits, 0.9)  # label smoothing
            fake_targets = torch.zeros_like(fake_logits)

            d_loss = bce(real_logits, real_targets) + bce(fake_logits, fake_targets)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            d_opt.step()
            d_losses.append(float(d_loss.item()))

            # ---- G step(s) ----
            D.eval()
            for p in D.parameters():
                p.requires_grad_(False)

            for _ in range(g_steps):
                g_opt.zero_grad(set_to_none=True)

                z = torch.randn(B, 64, device=device)
                fake_x = G(z)
                if clip_z is not None:
                    fake_x = torch.clamp(fake_x, -clip_z, clip_z)
                fake_1m = fake_x.unsqueeze(1)

                with torch.no_grad():
                    _, real_feat, _ = D(real_1m)

                _, fake_feat, fake_logits = D(fake_1m)

                fm = feature_match_loss(real_feat, fake_feat, w_meanstd=w_fm_meanstd, w_cov=w_cov_now)
                mom = output_moment_loss(real_x, fake_x, w_out_mom=w_out_mom)
                outcov = output_cov_loss(real_x, fake_x, w_out_cov=w_out_cov)

                g_targets = torch.ones_like(fake_logits)
                adv = bce(fake_logits, g_targets)

                g_loss = fm + mom + outcov + adv_weight * adv
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                g_opt.step()
                g_losses.append(float(g_loss.item()))

            for p in D.parameters():
                p.requires_grad_(True)

        avg_d = float(np.mean(d_losses)) if d_losses else float("inf")
        avg_g = float(np.mean(g_losses)) if g_losses else float("inf")

        # Save best AFTER warmup to avoid picking under-dispersed early epochs
        if epoch > fm_cov_warmup and avg_g < best_score:
            best_score = avg_g
            best_state = {k: v.detach().cpu().clone() for k, v in G.state_dict().items()}

        if verbose and (epoch % max(1, epochs // 10) == 0):
            print(f"Epoch {epoch:>3}/{epochs} | D: {avg_d:.4f} | G: {avg_g:.4f}")

    if best_state is not None:
        G.load_state_dict(best_state)
        if verbose:
            print(f"\n✓ Loaded best model (best G score after warmup: {best_score:.4f})")

    return G


@torch.no_grad()
def generate_z(G, n, M, device, batch_size=256, clip_z=None):
    G.eval()
    out = []
    for i in range(0, n, batch_size):
        b = min(batch_size, n - i)
        z = torch.randn(b, 64, device=device)
        x = G(z)
        if clip_z is not None:
            x = torch.clamp(x, -clip_z, clip_z)
        out.append(x.cpu().numpy())
    out = np.vstack(out)
    assert out.shape == (n, M), f"Expected {(n, M)} got {out.shape}"
    return out


def quick_quality_score(X_real_z, X_fake_z):
    mean_diff = np.mean(np.abs(X_real_z.mean(axis=0) - X_fake_z.mean(axis=0)))
    std_diff = np.mean(np.abs(X_real_z.std(axis=0) - X_fake_z.std(axis=0)))
    k = min(20, X_real_z.shape[1])
    cr = np.corrcoef(X_real_z[:, :k].T)
    cf = np.corrcoef(X_fake_z[:, :k].T)
    corr_diff = np.nanmean(np.abs(cr - cf))
    return float(mean_diff + std_diff + 0.2 * corr_diff)


def hyperparameter_tuning(X_z, M, device, trials, epochs_per_trial=12):
    n = min(len(X_z), 50000)
    Xz = X_z[:n]
    v = min(10000, len(Xz) // 5)
    X_train, X_val = Xz[:-v], Xz[-v:]

    results = []
    configs = list(product(*[t["values"] for t in trials]))
    print(f"\nTesting {len(configs)} configs...\n")

    for i, vals in enumerate(configs, 1):
        cfg = {trials[j]["name"]: vals[j] for j in range(len(trials))}
        print(f"[{i}/{len(configs)}] {cfg}")

        try:
            G = train_gan(X_train, M, device, epochs=epochs_per_trial, verbose=False, **cfg)
            X_fake = generate_z(G, len(X_val), M, device, clip_z=cfg.get("clip_z", None))
            score = quick_quality_score(X_val, X_fake)
            results.append({"cfg": cfg, "score": score})
            print(f"  score={score:.6f}")
        except Exception as e:
            print(f"  FAILED: {e}")

    results.sort(key=lambda x: x["score"])
    with open("tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    best = results[0]["cfg"]
    print("\nBest config:", best)
    print("Saved: tuning_results.json")
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--dataset_base", default="../datasets/hai-22.04")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=2e-5)
    p.add_argument("--dropout_p", type=float, default=0.3)

    p.add_argument("--w_fm_meanstd", type=float, default=10.0)
    p.add_argument("--w_fm_cov", type=float, default=0.1)
    p.add_argument("--fm_cov_warmup", type=int, default=10)

    p.add_argument("--w_out_mom", type=float, default=1.0)
    p.add_argument("--w_out_cov", type=float, default=0.05)

    p.add_argument("--adv_weight", type=float, default=0.2)
    p.add_argument("--g_steps", type=int, default=2)

    p.add_argument("--clip_z", type=float, default=5.0, help="z-space clip; set 0 to disable")
    p.add_argument("--no_attack_gan", action="store_true")
    p.add_argument("--tune", action="store_true")

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"M: {args.M}\n")

    clip_z = None if (args.clip_z is None or args.clip_z <= 0) else float(args.clip_z)

    train_files = sorted(glob(f"{args.dataset_base}/train1.csv"))
    test_files = sorted(glob(f"{args.dataset_base}/test1.csv"))
    X, y = load_data(train_files, test_files)
    y = y.astype(int)

    if X.shape[1] < args.M:
        raise ValueError(f"M={args.M} exceeds available features ({X.shape[1]})")
    if X.shape[1] != args.M:
        X = X[:, :args.M]

    print(f"Data: {len(X)} samples × {X.shape[1]} features")
    print(f"Normal: {int((y==0).sum())}, Attack: {int((y==1).sum())}")

    # Fit scaler on NORMAL only; transform ALL with same scaler
    scaler = StandardScaler()
    scaler.fit(X[y == 0])
    X_z = scaler.transform(X)
    X_normal_z = X_z[y == 0]
    X_attack_z = X_z[y == 1]

    with open("gan_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\nAfter StandardScaler normalization (fit on NORMAL only):")
    print(f"  Global mean: {X_z.mean():.6f}")
    print(f"  Global std:  {X_z.std():.6f}")
    print(f"  Range: [{X_z.min():.4f}, {X_z.max():.4f}]")
    if clip_z is not None:
        print(f"  clip_z: {clip_z}")

    if args.tune:
        trials = [
            {"name": "batch_size", "values": [128, 256]},
            {"name": "lr_d", "values": [1e-5, 2e-5]},
            {"name": "adv_weight", "values": [0.15, 0.2]},
            {"name": "w_out_mom", "values": [0.5, 1.0, 2.0]},
            {"name": "w_out_cov", "values": [0.02, 0.05]},
            {"name": "w_fm_meanstd", "values": [5.0, 10.0]},
            {"name": "w_fm_cov", "values": [0.05, 0.1]},
            {"name": "fm_cov_warmup", "values": [10]},
            {"name": "dropout_p", "values": [0.3]},
            {"name": "g_steps", "values": [2]},
            {"name": "clip_z", "values": [clip_z]},
        ]
        best = hyperparameter_tuning(X_normal_z, args.M, device, trials)
    else:
        best = dict(
            batch_size=args.batch_size,
            lr_g=args.lr_g,
            lr_d=args.lr_d,
            dropout_p=args.dropout_p,
            w_fm_meanstd=args.w_fm_meanstd,
            w_fm_cov=args.w_fm_cov,
            fm_cov_warmup=args.fm_cov_warmup,
            w_out_mom=args.w_out_mom,
            w_out_cov=args.w_out_cov,
            adv_weight=args.adv_weight,
            g_steps=args.g_steps,
            clip_z=clip_z,
        )

    print("\n" + "=" * 70)
    print("TRAINING GAN #1 (NORMAL DATA)")
    print("=" * 70)

    G_normal = train_gan(
        X_normal_z, args.M, device,
        epochs=args.epochs,
        batch_size=best["batch_size"],
        lr_g=best["lr_g"],
        lr_d=best["lr_d"],
        dropout_p=best["dropout_p"],
        w_fm_meanstd=best["w_fm_meanstd"],
        w_fm_cov=best["w_fm_cov"],
        fm_cov_warmup=best["fm_cov_warmup"],
        w_out_mom=best["w_out_mom"],
        w_out_cov=best["w_out_cov"],
        adv_weight=best["adv_weight"],
        g_steps=best["g_steps"],
        clip_z=best["clip_z"],
        name="GAN_normal",
        verbose=True,
    )

    synth_normal_z = generate_z(G_normal, len(X_normal_z), args.M, device, clip_z=clip_z)
    synth_normal_raw = scaler.inverse_transform(synth_normal_z)

    np.save("synthetic_normal.npy", synth_normal_raw)
    torch.save(G_normal.state_dict(), "G_normal.pt")

    print(f"\n✓ Generated synthetic_normal.npy: {synth_normal_raw.shape}")
    print(f"  Synthetic (z) mean: {synth_normal_z.mean():.4f}")
    print(f"  Synthetic (z) std:  {synth_normal_z.std():.4f}")
    print(f"  Real normal (z) mean:{X_normal_z.mean():.4f}")
    print(f"  Real normal (z) std: {X_normal_z.std():.4f}")

    if (not args.no_attack_gan) and (len(X_attack_z) > 0):
        print("\n" + "=" * 70)
        print("TRAINING GAN #2 (ATTACK DATA)")
        print("=" * 70)

        attack_bs = min(64, max(16, len(X_attack_z) // 4))
        G_attack = train_gan(
            X_attack_z, args.M, device,
            epochs=args.epochs,
            batch_size=attack_bs,
            lr_g=best["lr_g"],
            lr_d=best["lr_d"],
            dropout_p=best["dropout_p"],
            w_fm_meanstd=best["w_fm_meanstd"],
            w_fm_cov=best["w_fm_cov"],
            fm_cov_warmup=best["fm_cov_warmup"],
            w_out_mom=best["w_out_mom"],
            w_out_cov=best["w_out_cov"],
            adv_weight=best["adv_weight"],
            g_steps=best["g_steps"],
            clip_z=best["clip_z"],
            name="GAN_attack",
            verbose=True,
        )

        synth_attack_z = generate_z(G_attack, len(X_attack_z), args.M, device, clip_z=clip_z)
        synth_attack_raw = scaler.inverse_transform(synth_attack_z)

        np.save("synthetic_attack.npy", synth_attack_raw)
        torch.save(G_attack.state_dict(), "G_attack.pt")

        print(f"\n✓ Generated synthetic_attack.npy: {synth_attack_raw.shape}")
    else:
        print("\n(Skipping attack GAN)")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("Generated files:")
    print("  - synthetic_normal.npy")
    if (not args.no_attack_gan) and (len(X_attack_z) > 0):
        print("  - synthetic_attack.npy")
        print("  - G_normal.pt, G_attack.pt")
    else:
        print("  - G_normal.pt")
    print("  - gan_scaler.pkl")


if __name__ == "__main__":
    main()
