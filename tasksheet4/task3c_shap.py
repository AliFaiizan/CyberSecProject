

from glob import glob
import numpy as np
import pandas as pd
import os
import re
import joblib
import shap
import matplotlib.pyplot as plt
from utils import load_data
from scenarios import scenario_1_split, scenario_2_split, scenario_3_split
from task2_cnn_latent import create_windows
from tensorflow.keras.models import load_model

def run_shap_for_model(
    model_name,
    model,
    X_train,
    X_test,
    output_dir,
    predict_fn=None,
    flatten=False,
    M=None,
    latent_dim=None
):
    """
    SHAP explainer.
    Works for:
      - RF → TreeExplainer
      - SVM, kNN → KernelExplainer
      - OCSVM → KernelExplainer w/ wrapper
      - CNN → DeepExplainer
    """

    os.makedirs(output_dir, exist_ok=True)

    # ============================
    # Proper input formatting
    # ============================
    # ML models use raw Z (no flatten)
    # CNN uses flatten=True (your LIME logic)
    if flatten:
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_test_flat  = X_test.reshape(len(X_test), -1)
    else:
        X_train_flat = X_train
        X_test_flat  = X_test

    print(X_test_flat.shape, X_train_flat.shape)

    feature_names = [f"f{i}" for i in range(X_train_flat.shape[1])]

    # ============================
    # SELECT SHAP EXPLAINER
    # ============================

    # -----------------------------------------------------
    # 1. RandomForest → TreeExplainer
    # -----------------------------------------------------
    if model_name == "RF":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_flat)
        print(f"[DEBUG RF] SHAP values type: {type(shap_values)}")
        print(f"[DEBUG RF] SHAP values shape: {shap_values[0].shape if isinstance(shap_values, list) else shap_values.shape}")
        
        # TreeExplainer can return list [class0, class1] or 3D array
        # Select attack class (index 1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # Select attack class
        
        print(f"[DEBUG RF] After selection: {shap_values.shape}")

    # -----------------------------------------------------
    # 2. SVM / kNN → KernelExplainer
    # -----------------------------------------------------
    elif model_name in ["SVM", "kNN"]:
        # Use more background samples for kNN
        n_background = 100 if model_name == "kNN" else 50
        background = shap.sample(X_train_flat, n_background)
        print(f"[DEBUG] Background shape: {background.shape}, X_train_flat: {X_train_flat.shape}")
        
        # Use predict_proba if available, otherwise decision_function for SVM
        if hasattr(model, "predict_proba"):
            predict_fn = model.predict_proba
        else:
            # SVM without probability=True, use decision_function
            def predict_fn(x):
                scores = model.decision_function(x)
                # Convert to probability-like format
                if scores.ndim == 1:
                    scores = scores.reshape(-1, 1)
                    probs = np.hstack([1 / (1 + np.exp(scores)), 1 / (1 + np.exp(-scores))])
                else:
                    probs = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
                return probs
        
        # For kNN, use more samples to evaluate (nsamples parameter)
        print()
        if model_name == "kNN":
            explainer = shap.KernelExplainer(predict_fn, background, link="identity")
            shap_values = explainer.shap_values(X_test_flat, nsamples=500)  # More samples for kNN
        else:
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_test_flat)
        print(f"[DEBUG] SHAP values shape: {shap_values[0].shape if isinstance(shap_values, list) else shap_values.shape}")
        
        # For binary classification, select attack class (index 1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # Select attack class
        
        print(f"[DEBUG] After selection: {shap_values.shape}")

    # -----------------------------------------------------
    # 3. OCSVM → KernelExplainer + decision function wrapper
    # -----------------------------------------------------
    elif model_name == "OCSVM":

        def anomaly_to_prob(clf, X):
            scores = clf.decision_function(X)
            norm   = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            return np.vstack([1-norm, norm]).T

        background = shap.sample(X_train_flat, 50)
        print(f"[DEBUG OCSVM] Background shape: {background.shape}, X_train_flat: {X_train_flat.shape}")
        explainer  = shap.KernelExplainer(lambda x: anomaly_to_prob(model, x), background)
        shap_values = explainer.shap_values(X_test_flat)
        print(f"[DEBUG OCSVM] SHAP values shape: {shap_values[0].shape if isinstance(shap_values, list) else shap_values.shape}")
        
        # For binary classification, select attack class (index 1)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # Select attack class

    # -----------------------------------------------------
    # 4. CNN → DeepExplainer (windowed input)
    # -----------------------------------------------------
    elif model_name == "CNN":

        # reshape back into CNN shape
        X_train_cnn = X_train.reshape((-1, M, latent_dim))
        X_test_cnn  = X_test.reshape((-1, M, latent_dim))

        background = X_train_cnn[:20]
        explainer = shap.DeepExplainer(model, background)

        shap_vals = explainer.shap_values(X_test_cnn[:1000])
        
        print(f"[DEBUG CNN] SHAP values type: {type(shap_vals)}")
        print(f"[DEBUG CNN] SHAP values length/shape: {len(shap_vals) if isinstance(shap_vals, list) else shap_vals.shape}")
        

        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 4:
            
            shap_values = shap_vals[:, :, :, 1].reshape(shap_vals.shape[0], -1)
        elif isinstance(shap_vals, list) and len(shap_vals) > 1:
            shap_values = shap_vals[1].reshape(shap_vals[1].shape[0], -1)
        else:
            shap_values = shap_vals[0].reshape(shap_vals[0].shape[0], -1) if isinstance(shap_vals, list) else shap_vals.reshape(shap_vals.shape[0], -1)
        
        X_test_flat = X_test_cnn.reshape(X_test_cnn.shape[0], -1)
        print(f"[DEBUG CNN] Final shapes - shap_values: {shap_values.shape}, X_test_flat: {X_test_flat.shape}")

    else:
        print(f"[SHAP] Unsupported model: {model_name}")
        return

    # ============================
    # PLOT RESULT
    # ============================
    try:
        # Use minimum of shap_values and X_test_flat to avoid index errors
        n_samples = min(len(shap_values) if isinstance(shap_values, list) else shap_values.shape[0], 
                       X_test_flat.shape[0])
        
        shap.summary_plot(
            shap_values[:n_samples] if isinstance(shap_values, np.ndarray) else [sv[:n_samples] for sv in shap_values],
            X_test_flat[:n_samples],
            feature_names=feature_names,
            show=False
        )
        out_file = f"{output_dir}/SHAP_{model_name}_summary.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Saved → {out_file}")
    except Exception as e:
        print(f"[SHAP ERROR] {model_name}: {e}")

def run_task3_d(scenario):
    print(f"\n=== RUNNING TASK 3(d) — SHAP EXPLANATIONS (Scenario {scenario}) ===")

    # # Load data
    # train_files = sorted(glob("../datasets/hai-22.04/train1.csv"))
    # test_files  = sorted(glob("../datasets/hai-22.04/test1.csv"))
    # X, y = load_data(train_files, test_files)
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

    # Load latent file
    if scenario == 1:
        latent_path = "vae_features/task1_dense_relu_ld8_reconstruction_M20.npy"
    else:
        latent_path = "vae_features/task1_dense_relu_ld8_classification_M20.npy"

    Z = np.load(latent_path)
    if len(Z) != len(y):
        y = y[:len(Z)]

    # Extract dims
    match = re.search(r'ld(\d+).*_M(\d+)', latent_path)
    latent_dim = int(match.group(1))
    M = int(match.group(2))

    # Scenario-specific details
    if scenario == 1:
        split_fn = scenario_1_split
        model_dir = "saved_models/Scenario1"
    elif scenario == 2:
        split_fn = scenario_2_split
        model_dir = "saved_models/Scenario2"
    else:
        split_fn = scenario_3_split
        model_dir = "saved_models/Scenario3"

    # PICK ONE FOLD (same as your LIME code currently does)
    folds = list(split_fn(Z, pd.Series(y), 2))
    if scenario == 1:
        fold_idx, train_idx, test_idx = folds[0]
    else:
        fold_idx, attack_id, train_idx, test_idx = folds[0]

    fold = fold_idx + 1

    Z_train, y_train = Z[train_idx], y[train_idx]
    Z_test,  y_test  = Z[test_idx],  y[test_idx]

    # =====================================================
    # SCENARIO 1 — One-class models
    # =====================================================
    if scenario == 1:

        models = {
            "OCSVM": f"{model_dir}/OCSVM_Fold{fold}.joblib",
            "LOF":   f"{model_dir}/LOF_Fold{fold}.joblib",  # SKIPPED IN SHAP
        }

        for name, path in models.items():
            if not os.path.exists(path):
                continue
            if name == "LOF":
                print("[SHAP] LOF not supported. Skipping.")
                continue

            model = joblib.load(path)
            out_dir = f"Task3_Results/Scenario1/SHAP/Fold{fold}/{name}"

            run_shap_for_model(name, model, Z_train, Z_test, out_dir)

        return

    # =====================================================
    # SCENARIO 2 & 3 — ML ON RAW Z + CNN ON WINDOWS
    # =====================================================
    ml_models = {
        #"SVM": f"{model_dir}/SVM_Fold{fold}.joblib",
        #"kNN": f"{model_dir}/kNN_Fold{fold}.joblib",
        #"RF":  f"{model_dir}/RandomForest_Fold{fold}.joblib",
    }

    # ML models use Z (same as your LIME)
    for name, path in ml_models.items():
        if not os.path.exists(path):
            continue
        model = joblib.load(path)
        out_dir = f"Task3_Results/Scenario{scenario}/SHAP/Fold{fold}/{name}"
        run_shap_for_model(name, model, Z_train, Z_test, out_dir)

    # CNN
    cnn_path = f"{model_dir}/CNN_Fold{fold}.h5"
    if os.path.exists(cnn_path):
        X_train_w, y_train_w = create_windows(Z_train, y_train, M)
        X_test_w,  y_test_w  = create_windows(Z_test,  y_test,  M)

        cnn_model = load_model(cnn_path)

        out_dir = f"Task3_Results/Scenario{scenario}/SHAP/Fold{fold}/CNN"
        run_shap_for_model(
            "CNN",
            cnn_model,
            X_train_w,
            X_test_w,
            out_dir,
            flatten=True,
            M=M,
            latent_dim=latent_dim
        )

    print("\n=== TASK 3(c) COMPLETED ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Task 3(c) — SHAP explanations")
    parser.add_argument("--scenario", "-sc", type=int, required=True, choices=[1, 2, 3], help="Scenario number (1, 2, or 3)")
    args = parser.parse_args()
    run_task3_d(args.scenario)
