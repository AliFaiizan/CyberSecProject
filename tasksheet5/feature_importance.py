#!/usr/bin/env python3
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from scenarios import scenario_1_split, scenario_2_split, scenario_3_split
from utils import load_data, create_windows_for_vae


# ---------------------------------------------------------
# Load latent features and proper window labels
# ---------------------------------------------------------
def load_latent_and_labels(latent_file, M=20):
    """
    Load latent features and generate proper window labels.
    This matches the logic from task2_FIXED.py
    """
    print(f"Loading latent features from: {latent_file}")
    Z = np.load(latent_file)
    print(f"  Latent features: {Z.shape}")
    
    # Try to load pre-saved labels
    label_file = latent_file.replace('.npy', '_labels.npy')
    if os.path.exists(label_file):
        print(f"  Loading pre-saved labels: {label_file}")
        y_window = np.load(label_file)
    else:
        print(f"  Regenerating window labels from raw data...")
        # Load raw data
        X_raw, y_raw = load_data(
            ["../datasets/hai-22.04/train1.csv"],
            ["../datasets/hai-22.04/test1.csv"]
        )
        
        # Generate window labels using SAME logic as Task 1
        _, y_window = create_windows_for_vae(
            X_raw, y_raw,
            window_size=M,
            mode="classification"
        )
    
    # Verify alignment
    if len(y_window) != Z.shape[0]:
        raise ValueError(
            f"Alignment error: {len(y_window)} labels vs {Z.shape[0]} features"
        )
    
    print(f"  Window labels: {y_window.shape}")
    print(f"  ✓ Alignment verified!\n")
    
    return Z, y_window


# ---------------------------------------------------------
# Plot helper
# ---------------------------------------------------------
def plot_importance(importances, feature_names, title, out_file):
    """Plot feature importance as bar chart"""
    idx = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[idx], color='steelblue')
    plt.xticks(range(len(importances)), feature_names[idx], rotation=45, ha='right')
    plt.xlabel('Latent Features', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [SAVED] {out_file}")


# ---------------------------------------------------------
# Compute feature importance
# ---------------------------------------------------------
def compute_importance(model, X_test, y_test, model_name, scenario):
    """
    Compute feature importance using model-specific methods
    """
    # Tree-based models have built-in feature_importances_
    if hasattr(model, "feature_importances_"):
        print(f"    Using built-in feature_importances_ for {model_name}")
        return model.feature_importances_
    
    # Linear models have coefficients
    if hasattr(model, "coef_"):
        print(f"    Using coefficient magnitudes for {model_name}")
        coef = model.coef_
        if coef.ndim > 1:
            # Multi-class: take mean absolute value across classes
            return np.abs(coef).mean(axis=0)
        return np.abs(coef)
    
    # For other models, use permutation importance
    print(f"    Computing permutation importance for {model_name}...")
    
    # Choose scoring metric based on scenario
    if scenario == 1:
        # One-class classification: use accuracy
        scoring = "accuracy"
    else:
        # Binary classification: use f1 score
        scoring = "f1"
    
    try:
        perm = permutation_importance(
            model, X_test, y_test,
            scoring=scoring,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        return perm.importances_mean
    except Exception as e:
        print(f"    [WARNING] Permutation importance failed: {e}")
        # Return uniform importance as fallback
        return np.ones(X_test.shape[1]) / X_test.shape[1]


# ---------------------------------------------------------
# Task 2(a) - Feature Importance Analysis
# ---------------------------------------------------------
def run_task2a(scenario, latent_file=None, M=20):
    """
    Compute and plot feature importance for all models in a scenario
    
    Args:
        scenario: Evaluation scenario (1, 2, or 3)
        latent_file: Path to latent features file (auto-detected if None)
        M: Window size used in Task 1 (default: 20)
    """
    print(f"\n{'='*70}")
    print(f"TASK 3A: FEATURE IMPORTANCE ANALYSIS - SCENARIO {scenario}")
    print(f"{'='*70}\n")
    
    # Auto-detect latent file if not provided
    if latent_file is None:
        vae_dir = "vae_features"
        files = [f for f in os.listdir(vae_dir) if f.endswith('.npy')]
        
        # For scenario 1, prefer reconstruction; for 2/3, use classification
        if scenario == 1:
            candidates = [f for f in files if "reconstruction" in f or "classification" in f]
        else:
            candidates = [f for f in files if "classification" in f]
        
        if not candidates:
            raise FileNotFoundError(f"No latent feature files found in {vae_dir}")
        
        latent_file = os.path.join(vae_dir, candidates[0])
        print(f"Auto-detected latent file: {latent_file}\n")
    
    # Load latent features and window labels
    Z, y_window = load_latent_and_labels(latent_file, M)
    
    # Feature names
    feature_names = np.array([f"z{i}" for i in range(Z.shape[1])])
    
    # Select scenario and models
    if scenario == 1:
        scenario_fn = scenario_1_split
        models = ["OCSVM", "LOF", "EllipticEnvelope"]
    else:
        scenario_fn = scenario_2_split if scenario == 2 else scenario_3_split
        models = ["SVM", "kNN", "RandomForest"]
    
    # Create output directory
    out_dir = f"feature_importance/Scenario{scenario}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Store all importances for aggregation
    all_importances = {model: [] for model in models}
    
    print(f"Processing {len(models)} models across 5 folds...\n")
    
    # Iterate through folds
    for split in scenario_fn(pd.DataFrame(Z), pd.Series(y_window), k=5):
        
        # Unpack split based on scenario
        if scenario == 1:
            fold_idx, train_idx, test_idx = split
        else:
            fold_idx, _, train_idx, test_idx = split
        
        print(f"{'─'*70}")
        print(f"FOLD {fold_idx + 1}")
        print(f"{'─'*70}")
        
        X_test = Z[test_idx]
        y_test = y_window[test_idx]
        
        print(f"  Test set size: {len(X_test)}")
        
        # Process each model
        for model_name in models:
            model_path = f"saved_models/Scenario{scenario}/{model_name}_Fold{fold_idx+1}.joblib"
            
            if not os.path.exists(model_path):
                print(f"  [WARNING] Model not found: {model_path}")
                continue
            
            print(f"\n  {model_name}:")
            
            # Load model
            model = joblib.load(model_path)
            
            # Compute importance
            importances = compute_importance(model, X_test, y_test, model_name, scenario)
            
            # Store for aggregation
            all_importances[model_name].append(importances)
            
            # Plot per-fold importance
            out_file = f"{out_dir}/{model_name}_Fold{fold_idx+1}_importance.png"
            title = f"Scenario {scenario} - {model_name} Feature Importance (Fold {fold_idx+1})"
            plot_importance(importances, feature_names, title, out_file)
        
        print()
    
    # ---------------------------------------------------------
    # Aggregate importances across folds
    # ---------------------------------------------------------
    print(f"{'='*70}")
    print(f"GENERATING AGGREGATED IMPORTANCE PLOTS")
    print(f"{'='*70}\n")
    
    for model_name in models:
        if not all_importances[model_name]:
            print(f"  [WARNING] No importance data for {model_name}")
            continue
        
        # Compute mean importance across folds
        mean_importance = np.mean(all_importances[model_name], axis=0)
        std_importance = np.std(all_importances[model_name], axis=0)
        
        # Plot aggregated importance with error bars
        idx = np.argsort(mean_importance)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(mean_importance)), mean_importance[idx], 
                yerr=std_importance[idx], capsize=5, color='steelblue', 
                alpha=0.8, error_kw={'linewidth': 2})
        plt.xticks(range(len(mean_importance)), feature_names[idx], 
                   rotation=45, ha='right')
        plt.xlabel('Latent Features', fontsize=12)
        plt.ylabel('Mean Importance Score', fontsize=12)
        plt.title(f"Scenario {scenario} - {model_name} Aggregated Feature Importance (Mean ± Std)", 
                  fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        out_file = f"{out_dir}/{model_name}_aggregated_importance.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  [SAVED] {out_file}")
        
        # Print top features
        print(f"\n  {model_name} - Top 3 Most Important Features:")
        for i in range(min(3, len(idx))):
            feat_idx = idx[i]
            print(f"    {i+1}. {feature_names[feat_idx]}: {mean_importance[feat_idx]:.4f} ± {std_importance[feat_idx]:.4f}")
    
    print(f"\n{'='*70}")
    print(f"✓ COMPLETED SCENARIO {scenario}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Task 2(a): Feature Importance Analysis"
    )
    parser.add_argument(
        "--scenario", 
        type=int, 
        required=True, 
        choices=[1, 2, 3],
        help="Evaluation scenario (1, 2, or 3)"
    )
    parser.add_argument(
        "--latent-file",
        type=str,
        default=None,
        help="Path to latent features .npy file (auto-detected if not provided)"
    )
    parser.add_argument(
        "-M", "--window-size",
        type=int,
        default=20,
        help="Window size M used in Task 1 (default: 20)"
    )
    
    args = parser.parse_args()
    
    run_task2a(args.scenario, args.latent_file, args.window_size)