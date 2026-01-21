#This file coresponds to tasksheet5 task e,f,g - experiments with VAE feature space

#!/usr/bin/env python3
import argparse, os, time, psutil
import numpy as np
import pandas as pd
from joblib import dump

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

process = psutil.Process()


# =====================================================================
#  Generate proper window labels from raw data
# =====================================================================
def load_latent_features_and_labels(latent_file, M=20, train_files=None, test_files=None):
    """
    Load latent features and generate corresponding window labels.
    
    Args:
        latent_file: Path to .npy file with latent features
        M: Window size (must match Task 1)
        train_files: List of training CSV files
        test_files: List of test CSV files
    
    Returns:
        Z: Latent features [N_windows, latent_dim]
        y_window: Window-level labels [N_windows]
    """
    print("\n" + "="*70)
    print("LOADING LATENT FEATURES AND GENERATING WINDOW LABELS")
    print("="*70)
    
    # Load latent features
    print(f"Loading latent features from: {latent_file}")
    Z = np.load(latent_file)
    print(f"  ✓ Loaded latent features: {Z.shape}")
    
    # Try to load pre-saved labels first (if Ali added them later)
    label_file = latent_file.replace('.npy', '_labels.npy')
    if os.path.exists(label_file):
        print(f"Found pre-saved window labels: {label_file}")
        y_window = np.load(label_file)
        print(f"Loaded window labels: {y_window.shape}")
    else:
        print(f"No pre-saved labels found. Regenerating from raw data...")
        
        # Load raw data
        print(f"Loading raw data...")
        X_raw, y_raw = load_data(train_files, test_files)
        print(f"    Raw data shape: X={X_raw.shape}, y={y_raw.shape}")
        
        # Generate window labels using SAME logic as Task 1
        print(f"  → Generating window labels (M={M}, mode=classification)...")
        _, y_window = create_windows_for_vae(
            X_raw,
            y_raw,
            window_size=M,
            mode="classification"  # Aggregates: label=1 if ANY row in window is attack
        )
        print(f"Generated window labels: {y_window.shape}")
    
    # Verify alignment
    print("\n" + "-"*70)
    print("VERIFICATION:")
    print("-"*70)
    print(f"  Latent features: {Z.shape[0]} windows")
    print(f"  Window labels:   {len(y_window)} labels")
    
    if len(y_window) != Z.shape[0]:
        raise ValueError(
            f"ALIGNMENT ERROR: {len(y_window)} labels vs {Z.shape[0]} features!\n"
            f"Make sure window size M={M} matches what was used in Task 1."
        )
    
    print(f" ALIGNMENT VERIFIED!")
    print(f"\n  Normal windows:  {(y_window == 0).sum()} ({(y_window == 0).sum()/len(y_window)*100:.1f}%)")
    print(f"  Attack windows:  {(y_window == 1).sum()} ({(y_window == 1).sum()/len(y_window)*100:.1f}%)")
    print("="*70 + "\n")
    
    return Z, y_window


# =====================================================================
# Save trained model
# =====================================================================
def save_model(model, scenario_id, model_name, fold_idx):
    out_dir = f"saved_models/Scenario{scenario_id}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{model_name}_Fold{fold_idx+1}.joblib"
    dump(model, path)
    print(f"  → Saved model: {path}")


# =====================================================================
# Window → Row Mapping (CNN only)
# =====================================================================
def map_windows_to_rows(window_preds, N, M):
    """
    Convert window-level predictions to row-level predictions using voting.
    
    Args:
        window_preds: Predictions for windows [N_windows]
        N: Number of rows in original data
        M: Window size
    
    Returns:
        row_preds: Predictions for rows [N_rows]
    """
    row_votes = np.zeros(N)
    row_counts = np.zeros(N)

    for w in range(len(window_preds)):
        start, end = w, min(w + M, N)  # ★ FIX: Ensure we don't exceed N
        row_votes[start:end] += window_preds[w]
        row_counts[start:end] += 1

    row_counts[row_counts == 0] = 1
    row_preds = (row_votes / row_counts) >= 0.5
    return row_preds.astype(int)


# =====================================================================
# ML Training + Prediction Saving
# =====================================================================
def run_and_save(model_name, run_fn, X_df, y_series, scenario_fn, k, scenario_id, out_base):

    model_dir = os.path.join(out_base, model_name)
    os.makedirs(model_dir, exist_ok=True)
    rows = []

    print(f"\n{'='*70}")
    print(f"Running {model_name} for Scenario {scenario_id}")
    print(f"{'='*70}")

    for res in run_fn(X_df, y_series, k, scenario_fn):
        
        if scenario_id == 1:
            fold_idx, test_idx, y_pred, y_true, model, fe_rt, fe_mem, clf_rt, clf_mem = res
            attack_id = None
        else:
            fold_idx, attack_id, test_idx, y_pred, y_true, model, fe_rt, fe_mem, clf_rt, clf_mem = res

        # Precision / Recall
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)

        # Save predictions
        pd.DataFrame({
            "predicted_label": y_pred,
            "Attack": y_true
        }).to_csv(f"{model_dir}/Predictions_Fold{fold_idx+1}.csv", index=False)

        save_model(model, scenario_id, model_name, fold_idx)

        rows.append({
            "fold": fold_idx+1,
            "attack_id": attack_id,
            "precision": precision,
            "recall": recall,
            "feature_runtime_sec": fe_rt,
            "feature_memory_bytes": fe_mem,
            "runtime_sec": clf_rt,
            "memory_bytes": clf_mem
        })

        print(f"  Fold {fold_idx+1}: precision={precision:.4f}, recall={recall:.4f}")

    pd.DataFrame(rows).to_csv(f"{model_dir}/metrics_summary.csv", index=False)
    print(f"Saved metrics summary")


# =====================================================================
# MAIN
# =====================================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-sc", "--scenario", required=True, type=int, choices=[1,2,3])
    parser.add_argument("--latent-file", required=True)
    parser.add_argument("-k", "--folds", type=int, default=5)
    parser.add_argument("-M", "--window-size", type=int, default=20,
                        help="Window size M (must match Task 1)")
    args = parser.parse_args()

    sc = args.scenario
    k = args.folds
    M = args.window_size

    # ---------------------------------------------------------
    # Load latent features AND generate window labels
    # ---------------------------------------------------------
    train_files = ["../datasets/hai-22.04/train1.csv"]
    test_files = ["../datasets/hai-22.04/test1.csv"]
    
    Z, y_window = load_latent_features_and_labels(
        latent_file=args.latent_file,
        M=M,
        train_files=train_files,
        test_files=test_files
    )
    
    # Convert to pandas for compatibility with existing code
    y_series = pd.Series(y_window).astype(int)
    X_df = pd.DataFrame(Z)

    
    print(f"Ready for experiments: X={X_df.shape}, y={y_series.shape}")

    # ---------------------------------------------------------
    # Select models
    # ---------------------------------------------------------
    if sc == 1:
        scenario_fn = scenario_1_split
        model_map = {
            "OCSVM": run_OneClassSVM,
            "LOF": run_LOF,
            "EllipticEnvelope": run_EllipticEnvelope
        }
    else:
        scenario_fn = scenario_2_split if sc == 2 else scenario_3_split
        model_map = {
            "SVM": run_binary_svm,
            "kNN": run_knn,
            "RandomForest": run_random_forest
        }

    out_base = f"exports/Scenario{sc}"
    os.makedirs(out_base, exist_ok=True)

    # ---------------------------------------------------------
    # Train ML models
    # ---------------------------------------------------------
    for name, fn in model_map.items():
        run_and_save(name, fn, X_df, y_series, scenario_fn, k, sc, out_base)



if __name__ == "__main__":
    main()
