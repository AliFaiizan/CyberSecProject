#!/usr/bin/env python3
from glob import glob
import os
import numpy as np
import pandas as pd
import joblib
from utils import load_data
import re
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

from task2_cnn_latent import create_windows
from scenarios import scenario_1_split, scenario_2_split, scenario_3_split
from utils import create_windows_for_vae

# ========================================================================
# CNN reshape wrapper for LIME
# ========================================================================
def make_cnn_predict_fn(model, M, latent_dim):
    def predict_fn(x_flat):
        x = x_flat.reshape((x_flat.shape[0], M, latent_dim))
        return model.predict(x)
    return predict_fn



# ========================================================================
# Run LIME for ML or CNN classifier — ALL SAMPLES + AGGREGATED PLOT
# ========================================================================
def run_lime_for_model(model_name, model, X_train, X_test, output_dir, predict_fn=None, flatten=False, save_individuals=False):
    if flatten:
        X_train = X_train.reshape(len(X_train), -1)
        X_test  = X_test.reshape(len(X_test), -1)
    os.makedirs(output_dir, exist_ok=True)
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=[f"f{i}" for i in range(X_train.shape[1])],
        class_names=["normal", "attack"],
        discretize_continuous=True
    )

    # Automatic probability wrapper
    if predict_fn is None:
        if hasattr(model, "predict_proba"):
            def predict_fn(x): return model.predict_proba(x)

        elif hasattr(model, "decision_function"):
            print(f"[INFO] {model_name}: Using decision_function")
            def predict_fn(x):
                scores = model.decision_function(x)
                if scores.ndim == 1:
                    scores = np.vstack([-scores, scores]).T
                return scores

        else:
            raise RuntimeError(f"{model_name} has no probability or decision_function!")

    # Run LIME on ALL samples
    num_features = min(8, X_test.shape[1])  # Use up to 8 features
    feature_importance_dict = {f"f{i}": 0.0 for i in range(X_test.shape[1])}
    
    print(f"[LIME] Running on {len(X_test)} samples for {model_name}...")
    
    for idx in range(len(X_test)):
        exp = explainer.explain_instance(
            X_test[idx],
            predict_fn,
            num_features=num_features
        )
        
        # Extract feature weights (for class 1 = attack)
        for feature, weight in exp.as_list():
            # Parse feature name (e.g., "f5 <= 0.5" → "f5")
            feat_name = feature.split()[0]
            if feat_name in feature_importance_dict:
                feature_importance_dict[feat_name] += abs(weight)
        
        # Optionally save individual samples (first 3)
        if save_individuals and idx < 3:
            fig = exp.as_pyplot_figure()
            out_file = f"{output_dir}/LIME_{model_name}_sample{idx}.png"
            fig.savefig(out_file, dpi=150, bbox_inches='tight')
            print(f"[LIME] Saved → {out_file}")
            plt.close(fig)
        
        # Progress tracking every 1000 samples
        if (idx + 1) % 1000 == 0:
            print(f"[LIME] Processed {idx + 1}/{len(X_test)} samples...")
    
    # Normalize by number of samples
    for key in feature_importance_dict:
        feature_importance_dict[key] /= len(X_test)
    
    # Create aggregated plot
    features = list(feature_importance_dict.keys())
    importances = list(feature_importance_dict.values())
    
    # Sort by importance
    sorted_pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    features_sorted = [p[0] for p in sorted_pairs]
    importances_sorted = [p[1] for p in sorted_pairs]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(features_sorted, importances_sorted, color='steelblue')
    ax.set_xlabel('Average Absolute Feature Weight', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'LIME Feature Importance — {model_name} (Aggregated over {len(X_test)} samples)', fontsize=13)
    plt.tight_layout()
    
    agg_file = f"{output_dir}/LIME_{model_name}_aggregated_importance.png"
    plt.savefig(agg_file, dpi=150, bbox_inches='tight')
    print(f"[LIME] Aggregated plot saved → {agg_file}")
    plt.close(fig)
    


# ========================================================================
# MAIN — Task 3(c)
# ========================================================================
def run_task3_c(scenario):

    print(f"\n=== RUNNING TASK 3(c) — LIME EXPLANATIONS (Scenario {scenario}) ===")

    # Load labels from dataset
    # train_files = sorted(glob("../datasets/hai-22.04/train1.csv"))
    # test_files  = sorted(glob("../datasets/hai-22.04/test1.csv"))
    # X, y = load_data(train_files, test_files)   # X: [T,F], y: [T]
    train_data = np.load("synthetic_train.npy")  # shape: [N_train, F]
    test_data = np.load("synthetic_test.npy")    # shape: [N_test, F]
    test_labels = np.load("synthetic_test_labels.npy")  # shape: [N_test,] or [N_test, 1]

    # Ensure test_labels is a column vector
    if test_labels.ndim == 1:
        test_labels = test_labels[:, None]

    # Add label column to train (all zeros)
    train_labels = np.zeros((train_data.shape[0], 1))
    train_data_with_label = np.hstack([train_data, train_labels])
    test_data_with_label = np.hstack([test_data, test_labels])

    # Combine
    all_data = np.vstack([train_data_with_label, test_data_with_label])

    # Now, features and labels:
    X = all_data[:, :-1]  # all columns except last
    y = all_data[:, -1]   # last column

    latent_path = "vae_features/task1_dense_relu_ld8_classification_M20.npy"

    
    match = re.search(r'ld(\d+).*_M(\d+)', latent_path)
    if match:
        latent_dim = int(match.group(1))
        M = int(match.group(2))
        print("latent_dim:", latent_dim)
        print("M:", M)
    else:
        print("Could not extract latent_dim and M from path!")

    if not os.path.exists(latent_path):
        print("[FATAL] Missing latent file:", latent_path)
        return

    _, y_window = create_windows_for_vae(
        X,
        y,
        window_size=M,
        mode="classification"  # Aggregates: label=1 if ANY row in window is attack
    )
    print(f"Generated window labels: {y_window.shape}")

    Z = np.load(latent_path)
    # Trim labels exactly like Task 2
    if len(y) != len(Z):
        print(f"[INFO] Adjusting labels: Z={len(Z)}, y={len(y)} → trimming")
        y = y[:len(Z)]
    print("[INFO] Loaded latent features:", Z.shape)

    # Choose split functions and model directories
    if scenario == 1:
        split_fn = scenario_1_split
        model_dir = "saved_models/Scenario1"
    elif scenario == 2:
        split_fn = scenario_2_split
        model_dir = "saved_models/Scenario2"
    else:
        split_fn = scenario_3_split
        model_dir = "saved_models/Scenario3"

    # Get only FIRST fold
    folds = list(split_fn(Z, pd.Series(y_window), 2))
    
    if scenario == 1:
        fold_idx, train_idx, test_idx = folds[0]
    else:
        fold_idx, attack_id, train_idx, test_idx = folds[0]

    fold = fold_idx + 1
    Z_train, y_train = Z[train_idx], y_window[train_idx]
    Z_test,  y_test  = Z[test_idx],  y_window[test_idx]

    # =====================================================
    # SCENARIO 1 — One-class models
    # =====================================================
    if scenario == 1:
        models = {
            "OCSVM": f"{model_dir}/OCSVM_Fold{fold}.joblib",
            "LOF":   f"{model_dir}/LOF_Fold{fold}.joblib",
            "EllipticEnvelope":    f"{model_dir}/EllipticEnvelope_Fold{fold}.joblib"
        }

        for name, path in models.items():
            if not os.path.exists(path):
                print(f"[WARN] Missing {name}: {path}")
                continue

            model = joblib.load(path)
            out_dir = f"Task3_Results/Scenario1/LIME/Fold{fold}/{name}"
            run_lime_for_model(name, model, Z_train, Z_test, out_dir)

    # =====================================================
    # SCENARIO 2 & 3 — ML models + CNN
    # =====================================================
    else:
        ml_models = {
            "SVM": f"{model_dir}/SVM_Fold{fold}.joblib",
            "kNN": f"{model_dir}/kNN_Fold{fold}.joblib",
            "RF":  f"{model_dir}/RandomForest_Fold{fold}.joblib"
        }

        # ML MODELS
        for name, path in ml_models.items():
            if not os.path.exists(path):
                print(f"[WARN] Missing ML model: {path}")
                continue

            model = joblib.load(path)
            out_dir = f"Task3_Results/Scenario{scenario}/LIME/Fold{fold}/{name}"
            run_lime_for_model(name, model, Z_train, Z_test, out_dir)

        # CNN MODEL
        print(f"[INFO] M={M}, latent_dim={latent_dim}")
        cnn_path = f"exports/Scenario{scenario}/CNN/CNN_Fold{fold}.h5"
        if not os.path.exists(cnn_path):
            print(f"[WARN] Missing CNN model: {cnn_path}")
        else:
            X_train_w, y_train_w = create_windows(Z_train, y_train, M)
            X_test_w,  y_test_w  = create_windows(Z_test,  y_test,  M)

            cnn_model = load_model(cnn_path)
            cnn_predict_fn = make_cnn_predict_fn(cnn_model, M, latent_dim)

            out_dir = f"Task3_Results/Scenario{scenario}/LIME/Fold{fold}/CNN"

            run_lime_for_model(
                "CNN",
                cnn_model,
                X_train_w,
                X_test_w,
                out_dir,
                predict_fn=cnn_predict_fn,
                flatten=True
            )
       
    print("\n=== TASK 3(c) COMPLETED ===")


# ========================================================================
# CLI
# ========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Task 3(c) — LIME explanations")
    parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--M", type=int, default=20)

    args = parser.parse_args()
    run_task3_c(args.scenario)
