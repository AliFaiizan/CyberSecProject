#!/usr/bin/env python3
"""
Generate fold data specifically for Task 1 CNN experiments
This script ONLY generates RAW (unnormalized) data for CNN


Usage:
  python generate_folds_cnn.py -sc 2 -k 5
  python generate_folds_cnn.py -sc 3 -k 10
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import pickle
from glob import glob


from gan import Generator
from utils import load_data
from scenarios import scenario_2_split, scenario_3_split


# =========================================================
# SYNTHETIC FIXUPS (constants / binary / categorical)
# =========================================================
def fix_synthetic_data(X_synth: np.ndarray) -> np.ndarray:
    """Apply categorical/binary constraints to synthetic data"""
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
# LOAD GAN MODELS
# =========================================================
def load_gan_models(device: str, F: int = 86):
    """Load pre-trained GAN generators"""
    print("Loading GAN generators...")
    G_normal = Generator(F).to(device)
    G_attack = Generator(F).to(device)

    G_normal.load_state_dict(torch.load("G_normal.pt", map_location=device))
    G_attack.load_state_dict(torch.load("G_attack.pt", map_location=device))

    G_normal.eval()
    G_attack.eval()
    print("✓ Loaded GAN generators")

    print("Loading GAN scaler...")
    with open("gan_scaler.pkl", "rb") as f:
        gan_scaler = pickle.load(f)
    print("✓ Loaded GAN scaler")

    return G_normal, G_attack, gan_scaler


# =========================================================
# GENERATE SYNTHETIC DATA
# =========================================================
@torch.no_grad()
def generate_synthetic(G, n: int, device: str, batch_size: int = 128) -> np.ndarray:
    """Generate n synthetic samples using generator G"""
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
    """Create realistic attack sequences by interleaving normal and attack blocks"""
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
# GENERATE SYNTHETIC TRAINING SET
# =========================================================
def generate_synthetic_training_set(
    train_idx: np.ndarray,
    y_real: np.ndarray,
    G_normal,
    G_attack,
    gan_scaler,
    device: str,
    real_min: np.ndarray,
    real_max: np.ndarray,
    *,
    use_blocks: bool,
    normal_block_len: int,
    attack_block_len: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data matching the distribution of real training data"""
    
    y_train_real = y_real[train_idx]
    n_normal = int(np.sum(y_train_real == 0))
    n_attack = int(len(train_idx) - n_normal)

    print(f"    Generating {n_normal} NORMAL + {n_attack} ATTACK synthetic samples")

    # Generate normal samples
    Xn_norm = generate_synthetic(G_normal, n_normal, device)
    Xn = gan_scaler.inverse_transform(Xn_norm)
    Xn = fix_synthetic_data(Xn)

    # Generate attack samples
    if n_attack > 0:
        Xa_norm = generate_synthetic(G_attack, n_attack, device)
        Xa = gan_scaler.inverse_transform(Xa_norm)
        Xa = fix_synthetic_data(Xa)
    else:
        Xa = np.zeros((0, Xn.shape[1]), dtype=Xn.dtype)

    if n_attack == 0:
        return Xn, np.zeros(n_normal, dtype=int)

    # Interleave or concatenate
    if use_blocks:
        X_seq, y_seq = interleave_blocks(Xn, Xa, normal_block_len, attack_block_len, seed)
    else:
        X_seq = np.vstack([Xn, Xa])
        y_seq = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_attack, dtype=int)])

    # Clip to real data range to reduce domain shift
    if len(X_seq) > 0:
        X_seq = np.clip(X_seq, real_min, real_max)

    return X_seq, y_seq


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n" + "=" * 80)
    print("FOLD DATA GENERATION FOR CNN (Task 1)")
    print("Generates RAW (unnormalized) data only")
    print("=" * 80)

    parser = argparse.ArgumentParser(description="Generate CNN fold data")
    parser.add_argument("-sc", "--scenario", type=int, required=True, choices=[2, 3], 
                       help="Scenario number (2 or 3)")
    parser.add_argument("-k", "--folds", type=int, default=5, 
                       help="Number of CV folds (default: 5)")
    parser.add_argument("-F", "--feature-dim", type=int, default=86, 
                       help="Number of features (default: 86)")
    
    parser.add_argument("--no-blocks", action="store_true", 
                       help="Disable continuous block mixing for synthetic data")
    parser.add_argument("--normal-block-len", type=int, default=200, 
                       help="Normal block length (default: 200)")
    parser.add_argument("--attack-block-len", type=int, default=200, 
                       help="Attack block length (default: 200)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed (default: 42)")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    sc = args.scenario
    k = args.folds
    F = args.feature_dim

    # -----------------------
    # LOAD GAN MODELS
    # -----------------------
    print("\n" + "=" * 80)
    print("LOADING GAN MODELS")
    print("=" * 80)
    
    G_normal, G_attack, gan_scaler = load_gan_models(device=device, F=F)

    # -----------------------
    # LOAD REAL DATA
    # -----------------------
    print("\n" + "=" * 80)
    print("LOADING REAL HAI-22.04 DATA")
    print("=" * 80)

    train_files = sorted(glob("../datasets/hai-22.04/train1.csv"))
    test_files = sorted(glob("../datasets/hai-22.04/test1.csv"))
    
    if not train_files or not test_files:
        print("\nERROR: HAI-22.04 dataset not found!")
        print("Expected location: ../datasets/hai-22.04/")
        return

    X_raw, y = load_data(train_files, test_files)
    
    print(f"\nReal HAI-22.04 data loaded:")
    print(f"  Shape: {X_raw.shape}")
    print(f"  Normal samples: {int(np.sum(y == 0))}")
    print(f"  Attack samples: {int(np.sum(y == 1))}")
    print(f"  Data range: [{X_raw.min():.2f}, {X_raw.max():.2f}]")
    

    # Store min/max for clipping synthetic data
    real_min = X_raw.min(axis=0)
    real_max = X_raw.max(axis=0)

    # -----------------------
    # SELECT SCENARIO
    # -----------------------
    if sc == 2:
        scenario_fn = scenario_2_split
        print(f"\nUsing Scenario 2 split")
    else:
        scenario_fn = scenario_3_split
        print(f"\nUsing Scenario 3 split")

    use_blocks = not args.no_blocks
    
    fold_data_dir = f"cnn_folds/Scenario{sc}_k{k}"
    os.makedirs(fold_data_dir, exist_ok=True)

    # -----------------------
    # GENERATE FOLDS
    # -----------------------
    print("\n" + "=" * 80)
    print(f"GENERATING {k} FOLDS")
    print("=" * 80)

    for fold_result in scenario_fn(pd.DataFrame(X_raw), pd.Series(y), k):
        fold_idx, _attack_id, train_idx, test_idx = fold_result

        print(f"\n{'─'*80}")
        print(f"Fold {fold_idx + 1}/{k}")
        print(f"{'─'*80}")

        # -----------------------
        # REAL DATA (RAW)
        # -----------------------
        X_train_real = X_raw[train_idx]
        y_train_real = y[train_idx]
        X_test_real = X_raw[test_idx]
        y_test_real = y[test_idx]

        print(f"Real data:")
        print(f"  Train: {X_train_real.shape}, Normal={np.sum(y_train_real==0)}, Attack={np.sum(y_train_real==1)}")
        print(f"  Test:  {X_test_real.shape}, Normal={np.sum(y_test_real==0)}, Attack={np.sum(y_test_real==1)}")

        # Save REAL data
        real_fold_dir = f"{fold_data_dir}/real_fold{fold_idx}"
        os.makedirs(real_fold_dir, exist_ok=True)
        
        np.save(f"{real_fold_dir}/train_raw.npy", X_train_real.astype(np.float32))
        np.save(f"{real_fold_dir}/train_raw_labels.npy", y_train_real.astype(np.int32))
        np.save(f"{real_fold_dir}/test_raw.npy", X_test_real.astype(np.float32))
        np.save(f"{real_fold_dir}/test_raw_labels.npy", y_test_real.astype(np.int32))
        
        print(f"  ✓ Saved to {real_fold_dir}/")

        # -----------------------
        # SYNTHETIC DATA (RAW)
        # -----------------------
        X_train_synth, y_train_synth = generate_synthetic_training_set(
            train_idx=train_idx,
            y_real=y,
            G_normal=G_normal,
            G_attack=G_attack,
            gan_scaler=gan_scaler,
            device=device,
            real_min=real_min,
            real_max=real_max,
            use_blocks=use_blocks,
            normal_block_len=args.normal_block_len,
            attack_block_len=args.attack_block_len,
            seed=args.seed + fold_idx,
        )

        print(f"Synthetic data:")
        print(f"  Train: {X_train_synth.shape}, Normal={np.sum(y_train_synth==0)}, Attack={np.sum(y_train_synth==1)}")
        print(f"  Test:  Using same test set as real data")

        # Save SYNTHETIC data
        syn_fold_dir = f"{fold_data_dir}/syn_fold{fold_idx}"
        os.makedirs(syn_fold_dir, exist_ok=True)
        
        np.save(f"{syn_fold_dir}/train_raw.npy", X_train_synth.astype(np.float32))
        np.save(f"{syn_fold_dir}/train_raw_labels.npy", y_train_synth.astype(np.int32))
        np.save(f"{syn_fold_dir}/test_raw.npy", X_test_real.astype(np.float32))  # Same test set
        np.save(f"{syn_fold_dir}/test_raw_labels.npy", y_test_real.astype(np.int32))
        
        print(f"  ✓ Saved to {syn_fold_dir}/")

    # -----------------------
    # VERIFICATION
    # -----------------------
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    # Load and verify one fold
    verify_fold = f"{fold_data_dir}/real_fold0/train_raw.npy"
    if os.path.exists(verify_fold):
        X_verify = np.load(verify_fold)
        print(f"\nVerifying: {verify_fold}")
        print(f"  Shape: {X_verify.shape}")
        print(f"  Range: [{X_verify.min():.2f}, {X_verify.max():.2f}]")
        print(f"  Mean:  {X_verify.mean():.2f}")
        print(f"  Std:   {X_verify.std():.2f}")
        
        if X_verify.max() > 10:
            print("  ✓ Data appears to be RAW (unnormalized)")
        else:
            print(" Warning: Data might be normalized (unexpected)")

    print("\n" + "=" * 80)
    print("FOLD GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: {fold_data_dir}/")
    print(f"\nFolder structure:")
    print(f"  {fold_data_dir}/")
    print(f"    ├── real_fold0/")
    print(f"    │   ├── train_raw.npy")
    print(f"    │   ├── train_raw_labels.npy")
    print(f"    │   ├── test_raw.npy")
    print(f"    │   └── test_raw_labels.npy")
    print(f"    ├── syn_fold0/")
    print(f"    │   └── (same structure)")
    print(f"    ├── real_fold1/")
    print(f"    └── ...")
    
    print(f"\n✓ Ready for CNN training!")
    print(f"\nNext step:")
    print(f"  python run_cnn_task1.py --scenario {sc} --k {k} --M 50 --data-type real")


if __name__ == "__main__":
    main()
