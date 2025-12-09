#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras.models import load_model

from task2_cnn_latent import create_windows
from scenarios_util import scenario_1_split, scenario_2_split, scenario_3_split


# ========================================================================
# CNN reshape wrapper for LIME
# ========================================================================
def make_cnn_predict_fn(model, M, latent_dim):
    def predict_fn(x_flat):
        x = x_flat.reshape((x_flat.shape[0], M, latent_dim))
        return model.predict(x)
    return predict_fn


# ========================================================================
# Load labels EXACTLY like Task 2
# ========================================================================
def load_labels_for_task2():
    print("[INFO] Loading labels from datasets (Task 2 method) ...")

    df_train = pd.read_csv("../../datasets/hai-22.04/train1.csv")
    df_test  = pd.read_csv("../../datasets/hai-22.04/test1.csv")

    full = pd.concat([df_train, df_test], ignore_index=True)
    labels = full["Attack"].values

    print(f"[INFO] Loaded {len(labels)} labels (Attack).")
    return labels


# ========================================================================
# Run LIME for ML or CNN classifier
# ========================================================================
def run_lime_for_model(model_name, model, X_train, X_test, fold, scenario, output_dir, predict_fn=None):
    
    os.makedirs(output_dir, exist_ok=True)

    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat  = X_test.reshape(len(X_test), -1)

    explainer = LimeTabularExplainer(
        training_data=X_train_flat,
        feature_names=[f"f{i}" for i in range(X_train_flat.shape[1])],
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

    # Pick 5 samples
    indices = np.linspace(0, len(X_test_flat)-1, 5, dtype=int)

    for idx in indices:
        exp = explainer.explain_instance(
            X_test_flat[idx],
            predict_fn,
            num_features=15
        )

        out_file = f"{output_dir}/LIME_{model_name}_sample{idx}.html"
        exp.save_to_file(out_file)
        print(f"[LIME] Saved → {out_file}")


# ========================================================================
# MAIN — Task 3(c)
# ========================================================================
def run_task3_c(scenario, latent_dim=8, M=20):

    print(f"\n=== RUNNING TASK 3(c) — LIME EXPLANATIONS (Scenario {scenario}) ===")

    # Load labels from dataset
    y = load_labels_for_task2()

    # Load latent features from Task 1
    if scenario == 1:
        latent_path = "vae_features/task1_dense_relu_ld8_reconstruction.npy"
    else:
        latent_path = "vae_features/task1_dense_relu_ld8_classification.npy"

    if not os.path.exists(latent_path):
        print("[FATAL] Missing latent file:", latent_path)
        return

    Z = np.load(latent_path)
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

    # Iterate folds — FIXED SCENARIO 1 UNPACKING
    if scenario == 1:
        fold_iterator = (
            (fold_idx, None, train_idx, test_idx)
            for fold_idx, train_idx, test_idx in split_fn(Z, pd.Series(y), 5)
        )
    else:
        fold_iterator = split_fn(Z, pd.Series(y), 5)

    for fold_idx, attack_id, train_idx, test_idx in fold_iterator:

        fold = fold_idx + 1
        print(f"\n[ Fold {fold} ]")

        # Extract latent vectors
        Z_train, y_train = Z[train_idx], y[train_idx]
        Z_test,  y_test  = Z[test_idx],  y[test_idx]

        # Create windows (same for CNN & ML)
        X_train_w, y_train_w = create_windows(Z_train, y_train, M)
        X_test_w,  y_test_w  = create_windows(Z_test,  y_test,  M)

        # ==============================
        # SCENARIO 1 — OCSVM, LOF, EE
        # ==============================
        if scenario == 1:

            models = {
                "OCSVM": f"{model_dir}/OCSVM_Fold{fold}.joblib",
                "LOF":   f"{model_dir}/LOF_Fold{fold}.joblib",
                "EE":    f"{model_dir}/EE_Fold{fold}.joblib"
            }

            for name, path in models.items():
                if not os.path.exists(path):
                    print(f"[WARN] Missing {name}: {path}")
                    continue

                model = joblib.load(path)
                out_dir = f"Task3_Results/Scenario1/LIME/Fold{fold}/{name}"

                run_lime_for_model(name, model, X_train_w, X_test_w, fold, scenario, out_dir)

            continue  # Skip CNN for scenario 1

        # ==============================
        # SCENARIO 2 & 3 — ML + CNN
        # ==============================
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

            run_lime_for_model(name, model, X_train_w, X_test_w, fold, scenario, out_dir)

        # CNN MODEL
        cnn_path = f"{model_dir}/CNN_Model_Fold{fold}.h5"
        if not os.path.exists(cnn_path):
            print(f"[WARN] Missing CNN model: {cnn_path}")
            continue

        cnn_model = load_model(cnn_path)
        cnn_predict_fn = make_cnn_predict_fn(cnn_model, M, latent_dim)

        out_dir = f"Task3_Results/Scenario{scenario}/LIME/Fold{fold}/CNN"

        run_lime_for_model(
            "CNN",
            cnn_model,
            X_train_w,
            X_test_w,
            fold,
            scenario,
            out_dir,
            predict_fn=cnn_predict_fn
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
    run_task3_c(args.scenario, args.latent_dim, args.M)
