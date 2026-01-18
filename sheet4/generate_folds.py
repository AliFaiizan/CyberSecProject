#!/usr/bin/env python3
"""
Generate and save fold data for Task Sheet 4 - Task 2

Necessary changes:
- Scenario 1: allow timestep shuffle (improves window diversity; reduces FP for one-class models).
- Scenario 2/3: DO NOT shuffle timesteps (prevents unrealistic mixed windows).
- Clip synthetic raw features to real min/max ONLY for Scenario 2/3 (reduces domain shift).
- Shuffle at WINDOW/LATENT level (safe for all scenarios).

Usage:
  python generate_folds.py -sc 1 -k 5 -M 20 --vae-checkpoint vae_classification_real.pt
  python generate_folds.py -sc 2 -k 5 -M 20 --vae-checkpoint vae_classification_real.pt
  python generate_folds.py -sc 3 -k 5 -M 20 --vae-checkpoint vae_classification_real.pt
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import pickle
from glob import glob

from vae import VAE, extract_latent_features
from gan import Generator
from utils import load_data, create_windows_for_vae
from scenarios import scenario_1_split, scenario_2_split, scenario_3_split


# =========================================================
# SYNTHETIC FIXUPS (constants / binary / categorical)
# =========================================================
def fix_synthetic_data(X_synth: np.ndarray) -> np.ndarray:
    X_fixed = X_synth.copy()

    # Constant features
    X_fixed[:, 28] = 0.0
    X_fixed[:, 30] = 1.0
    X_fixed[:, 31] = 1.0
    X_fixed[:, 32] = 0.0
    X_fixed[:, 33] = 0.0
    X_fixed[:, 34] = 1.0
    X_fixed[:, 35] = 1.0
    X_fixed[:, 38] = 0.0
    X_fixed[:, 39] = 0.0
    X_fixed[:, 40] = 1.0
    X_fixed[:, 48] = 0.0
    X_fixed[:, 53] = 1.0
    X_fixed[:, 54] = 2880.0
    X_fixed[:, 58] = 1.0
    X_fixed[:, 64] = 50.0
    X_fixed[:, 65] = 50.0
    X_fixed[:, 66] = 50.0
    X_fixed[:, 67] = 50.0
    X_fixed[:, 71] = 70.0
    X_fixed[:, 73] = 25.0

    # Binary features
    binary_features = [45, 46, 49, 50, 51]
    for feat_idx in binary_features:
        X_fixed[:, feat_idx] = np.round(np.clip(X_fixed[:, feat_idx], 0, 1))

    # Categorical features
    valid_vals_52 = np.array([12.66931, 12.90343761])
    X_fixed[:, 52] = valid_vals_52[np.argmin(np.abs(X_fixed[:, 52, None] - valid_vals_52), axis=1)]

    valid_vals_77 = np.array([0.716041982, 2.22778, 7.08818])
    X_fixed[:, 77] = valid_vals_77[np.argmin(np.abs(X_fixed[:, 77, None] - valid_vals_77), axis=1)]

    valid_vals_83 = np.array([2.5498, 8.90254, 13.62378, 15.10069, 20.98959351, 21.25391, 26.74452])
    X_fixed[:, 83] = valid_vals_83[np.argmin(np.abs(X_fixed[:, 83, None] - valid_vals_83), axis=1)]

    return X_fixed


# =========================================================
# LOAD PRE-TRAINED MODELS
# =========================================================
def load_pretrained_models(vae_checkpoint: str, device: str, F: int = 86):
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
        input_dim=checkpoint["input_dim"],
        latent_dim=checkpoint["latent_dim"],
        layer_type=checkpoint["layer_type"],
        activation=checkpoint["activation"],
        num_classes=None if checkpoint["mode"] == "reconstruction" else 2,
        seq_len=checkpoint["window_size"],
        feature_dim=checkpoint["feature_dim"],
    )
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.to(device)
    vae.eval()
    print("Loaded VAE")

    with open("gan_scaler.pkl", "rb") as f:
        gan_scaler = pickle.load(f)
    print("Loaded GAN scaler")

    with open("vae_scaler.pkl", "rb") as f:
        vae_scaler = pickle.load(f)
    print("Loaded VAE scaler")

    return G_normal, G_attack, vae, gan_scaler, vae_scaler, checkpoint


# =========================================================
# GENERATE SYNTHETIC DATA
# =========================================================
@torch.no_grad()
def generate_synthetic(G, n: int, device: str, batch_size: int = 128) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    G.eval()
    out = []
    for i in range(0, n, batch_size):
        b = min(batch_size, n - i)
        z = torch.randn(b, 64, device=device)
        out.append(G(z).cpu().numpy())
    return np.vstack(out)


def interleave_blocks(
    Xn: np.ndarray,
    Xa: np.ndarray,
    normal_block_len: int,
    attack_block_len: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    idx_n = rng.permutation(len(Xn))
    idx_a = rng.permutation(len(Xa))
    Xn = Xn[idx_n]
    Xa = Xa[idx_a]

    seq_X, seq_y = [], []
    i = j = 0
    take_normal = True

    while i < len(Xn) or j < len(Xa):
        if take_normal and i < len(Xn):
            e = min(len(Xn), i + normal_block_len)
            seq_X.append(Xn[i:e])
            seq_y.append(np.zeros(e - i, dtype=int))
            i = e
        elif (not take_normal) and j < len(Xa):
            e = min(len(Xa), j + attack_block_len)
            seq_X.append(Xa[j:e])
            seq_y.append(np.ones(e - j, dtype=int))
            j = e

        if i >= len(Xn):
            take_normal = False
        elif j >= len(Xa):
            take_normal = True
        else:
            take_normal = not take_normal

    X_seq = np.vstack(seq_X) if seq_X else np.zeros((0, Xn.shape[1]), dtype=np.float32)
    y_seq = np.concatenate(seq_y) if seq_y else np.zeros((0,), dtype=int)
    return X_seq, y_seq


# =========================================================
# EXTRACT VAE LATENT FEATURES
# =========================================================
@torch.no_grad()
def extract_vae_latent_simple(X_scaled: np.ndarray, vae, window_size: int, layer_type: str, device: str) -> np.ndarray:
    if X_scaled is None or len(X_scaled) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    X_windows, _ = create_windows_for_vae(X_scaled, np.zeros(len(X_scaled)), window_size=window_size, mode="classification")
    if len(X_windows) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    X_input = X_windows.reshape(len(X_windows), -1) if layer_type == "dense" else X_windows
    Z, _, _, _ = extract_latent_features(vae, X_input, device=device)
    return Z


# =========================================================
# SYNTHETIC TRAIN SET FOR SCENARIO 2/3
# =========================================================
def generate_synthetic_training_set(
    train_idx: np.ndarray,
    y_real: np.ndarray,
    G_normal,
    G_attack,
    gan_scaler,
    device: str,
    *,
    use_blocks: bool,
    normal_block_len: int,
    attack_block_len: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_train_real = y_real[train_idx]
    n_normal = int(np.sum(y_train_real == 0))
    n_attack = int(len(train_idx) - n_normal)

    print("    Generating synthetic training data:")
    print(f"      - {n_normal} normal samples")
    print(f"      - {n_attack} attack samples")

    Xn_norm = generate_synthetic(G_normal, n_normal, device)
    Xn = gan_scaler.inverse_transform(Xn_norm)
    Xn = fix_synthetic_data(Xn)

    if n_attack > 0:
        Xa_norm = generate_synthetic(G_attack, n_attack, device)
        Xa = gan_scaler.inverse_transform(Xa_norm)
        Xa = fix_synthetic_data(Xa)
    else:
        Xa = np.zeros((0, Xn.shape[1]), dtype=Xn.dtype)

    if n_attack == 0:
        return Xn, np.zeros(n_normal, dtype=int)

    if use_blocks:
        return interleave_blocks(Xn, Xa, normal_block_len, attack_block_len, seed)

    X_seq = np.vstack([Xn, Xa])
    y_seq = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_attack, dtype=int)])
    return X_seq, y_seq


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n" + "=" * 70)
    print("FOLD DATA GENERATION FOR TASK SHEET 4")
    print("=" * 70)

    parser = argparse.ArgumentParser(description="Generate fold data for Task Sheet 4")
    parser.add_argument("-sc", "--scenario", type=int, required=True, choices=[1, 2, 3], help="Scenario number")
    parser.add_argument("-k", "--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("-M", "--window-size", type=int, default=20, help="Window size")
    parser.add_argument("--vae-checkpoint", type=str, required=True, help="Path to trained VAE checkpoint")

    parser.add_argument("--no-blocks", action="store_true", help="Disable continuous block mixing (Scenario 2/3)")
    parser.add_argument("--normal-block-len", type=int, default=0, help="Normal block length (0=auto)")
    parser.add_argument("--attack-block-len", type=int, default=0, help="Attack block length (0=auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    sc = args.scenario
    k = args.folds
    M = args.window_size
    F = 86

    print("\n" + "=" * 70)
    print("LOADING PRE-TRAINED MODELS")
    print("=" * 70)

    G_normal, G_attack, vae, gan_scaler, vae_scaler, vae_ckpt = load_pretrained_models(
        vae_checkpoint=args.vae_checkpoint, device=device, F=F
    )
    layer_type = vae_ckpt["layer_type"]

    print("\n" + "=" * 70)
    print("LOADING REAL HAI-22.04 DATA")
    print("=" * 70)

    train_files = sorted(glob("../datasets/hai-22.04/train1.csv"))
    test_files = sorted(glob("../datasets/hai-22.04/test1.csv"))
    if not train_files or not test_files:
        print("\n❌ ERROR: HAI-22.04 dataset not found!")
        return

    X_raw, y = load_data(train_files, test_files)
    print("\nReal HAI-22.04 data loaded:")
    print(f"  Shape: {X_raw.shape}")
    print(f"  Normal samples: {int(np.sum(y == 0))}")
    print(f"  Attack samples: {int(np.sum(y == 1))}")

    # real min/max for clipping synthetic raw features (Scenario 2/3 only)
    real_min = X_raw.min(axis=0)
    real_max = X_raw.max(axis=0)

    X_real = vae_scaler.transform(X_raw)
    y_real = y
    print("✓ Normalized using VAE scaler")

    if sc == 1:
        scenario_fn = scenario_1_split
    elif sc == 2:
        scenario_fn = scenario_2_split
    else:
        scenario_fn = scenario_3_split

    normal_block_len = args.normal_block_len if args.normal_block_len > 0 else max(10 * M, 200)
    attack_block_len = args.attack_block_len if args.attack_block_len > 0 else max(10 * M, 200)
    use_blocks = (not args.no_blocks)

    fold_data_dir = f"exports_sheet4/Scenario{sc}"
    os.makedirs(fold_data_dir, exist_ok=True)

    print("\nGenerating synthetic latent features per fold...")

    for fold_result in scenario_fn(pd.DataFrame(X_real), pd.Series(y_real), k):
        if sc == 1:
            fold_idx, train_idx, test_idx = fold_result
        else:
            fold_idx, _attack_id, train_idx, test_idx = fold_result

        print(f"\nFold {fold_idx + 1}:")

        # -----------------------
        # STEP 1: Synthetic train (raw)
        # -----------------------
        if sc == 1:
            y_train_real = y_real[train_idx]
            n_normal = int(np.sum(y_train_real == 0))
            print(f"    Generating {n_normal} NORMAL synthetic samples (Scenario 1)")

            Xn_norm = generate_synthetic(G_normal, n_normal, device)
            X_train_raw = gan_scaler.inverse_transform(Xn_norm)
            X_train_raw = fix_synthetic_data(X_train_raw)
            y_train_ts = np.zeros(n_normal, dtype=int)

            # shuffle timesteps is improving window diversity
            perm_ts = np.random.permutation(len(X_train_raw))
            X_train_raw = X_train_raw[perm_ts]
            y_train_ts = y_train_ts[perm_ts]
        else:
            X_train_raw, y_train_ts = generate_synthetic_training_set(
                train_idx=train_idx,
                y_real=y_real,
                G_normal=G_normal,
                G_attack=G_attack,
                gan_scaler=gan_scaler,
                device=device,
                use_blocks=use_blocks,
                normal_block_len=normal_block_len,
                attack_block_len=attack_block_len,
                seed=args.seed + fold_idx,
            )
            # clip to reduce domain shift
            if len(X_train_raw) > 0:
                X_train_raw = np.clip(X_train_raw, real_min, real_max)

        # Scale synthetic (or empty)
        X_test_fold = X_real[test_idx]
        y_test_fold = y_real[test_idx]
        
        if (sc == 1):
            # Save raw synthetic train and real test fold before VAE for Scenario 1
            raw_dir = f"{fold_data_dir}/syn_fold{fold_idx}"
            os.makedirs(raw_dir, exist_ok=True)
            np.save(f"{raw_dir}/train_raw.npy", X_train_raw.astype(np.float32))
            np.save(f"{raw_dir}/train_raw_labels.npy", y_train_ts.astype(np.int32))
            np.save(f"{raw_dir}/test_raw.npy", X_test_fold.astype(np.float32))
            np.save(f"{raw_dir}/test_raw_labels.npy", y_test_fold.astype(np.int32))

        X_train_scaled = vae_scaler.transform(X_train_raw) if len(X_train_raw) > 0 else X_train_raw

        # -----------------------
        # STEP 2: Latent features
        # -----------------------
        print("    Extracting latent features...")

        Z_train = extract_vae_latent_simple(X_train_scaled, vae, M, layer_type, device)

        Z_test = extract_vae_latent_simple(X_test_fold, vae, M, layer_type, device)

        # Window labels
        _, y_train_windows = create_windows_for_vae(X_train_scaled, y_train_ts, M, "classification")
        _, y_test_windows = create_windows_for_vae(X_test_fold, y_test_fold, M, "classification")

        # Shuffle at WINDOW level (safe for all scenarios)
        if len(y_train_windows) > 0:
            perm = np.random.permutation(len(y_train_windows))
            Z_train = Z_train[perm]
            y_train_windows = y_train_windows[perm]

        print(f"      Train (synthetic): {Z_train.shape}  labels={y_train_windows.shape}")
        print(f"      Test (REAL):       {Z_test.shape}   labels={y_test_windows.shape}")

        # -----------------------
        # STEP 3: Save fold
        # -----------------------
        fold_dir = f"{fold_data_dir}/fold{fold_idx}"
        os.makedirs(fold_dir, exist_ok=True)

        np.save(f"{fold_dir}/train_latent.npy", Z_train.astype(np.float32))
        np.save(f"{fold_dir}/train_labels.npy", y_train_windows.astype(np.int32))
        np.save(f"{fold_dir}/test_latent.npy", Z_test.astype(np.float32))
        np.save(f"{fold_dir}/test_labels.npy", y_test_windows.astype(np.int32))

        print(f"    ✓ Saved to {fold_dir}/")

    print("\n" + "=" * 70)
    print("FOLD DATA GENERATION COMPLETE!")
    print("=" * 70)
    print(f"Fold data saved to: {fold_data_dir}/\n")


if __name__ == "__main__":
    main()
