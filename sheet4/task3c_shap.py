

import numpy as np
import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import argparse

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
        shap_values = explainer.shap_values(X_test_flat[:5000])  # REMOVE Limit to 100 samples for speed

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
            shap_values = explainer.shap_values(X_test_flat[:5000], nsamples=500)  # More samples for kNN
        else:
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_test_flat[:5000])  # REMOVE Limit to 100 samples for speed
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
        shap_values = explainer.shap_values(X_test_flat[:5000])  # REMOVE Limit to 1000 samples for speed
        print(f"[DEBUG OCSVM] SHAP values shape: {shap_values[0].shape if isinstance(shap_values, list) else shap_values.shape}")
        
        # For binary classification, select attack class (index 1)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # Select attack class

    # -----------------------------------------------------
    # 4. CNN → DeepExplainer 
    # -----------------------------------------------------
    elif model_name == "CNN":

        # Reshape to (batch, n_features, 1) for CNN
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_cnn  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        background = X_train_cnn[:20]
        explainer = shap.DeepExplainer(model, background)

        shap_vals = explainer.shap_values(X_test_cnn[:10000])  #REMOVE  Limit to 1000 samples for speed
        

        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 4:
            
            shap_values = shap_vals[:, :, :, 1].reshape(shap_vals.shape[0], -1)
        elif isinstance(shap_vals, list) and len(shap_vals) > 1:
            shap_values = shap_vals[1].reshape(shap_vals[1].shape[0], -1)
        else:
            shap_values = shap_vals[0].reshape(shap_vals[0].shape[0], -1) if isinstance(shap_vals, list) else shap_vals.reshape(shap_vals.shape[0], -1)
        
        

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

def run_task3_c(scenario, fold_idx=0):
    """
    Load pre-generated fold data and trained models.
    Run SHAP explanations on fold data only.
    
    Parameters:
    -----------
    scenario : int (1, 2, or 3)
    fold_idx : int (0-based fold index, default 0 for first fold)
    """
    
    # Find all available folds
    base_dir = f"exports_sheet4/Scenario{scenario}"
    available_folds = []
    
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.startswith("fold"):
                try:
                    fold_num = int(item.replace("fold", ""))
                    available_folds.append(fold_num)
                except ValueError:
                    pass
    
    available_folds.sort()
    
    if not available_folds:
        print(f"[ERROR] No fold directories found in {base_dir}")
        return
    
    print(f"[INFO] Available folds: {available_folds}")
    print(f"[INFO] Using fold: {fold_idx}")
    
    if fold_idx not in available_folds:
        print(f"[ERROR] Fold {fold_idx} not available. Choose from: {available_folds}")
        return
    
    print(f"\n=== TASK 3(c) — SHAP EXPLANATIONS (Scenario {scenario}, Fold {fold_idx}) ===")
    
    # Load fold data
    fold_data_dir = f"{base_dir}/fold{fold_idx}"
    
    print(f"[INFO] Loading fold data from: {fold_data_dir}")
    
    Z_train = np.load(f"{fold_data_dir}/train_latent.npy")
    y_train = np.load(f"{fold_data_dir}/train_labels.npy")
    Z_test = np.load(f"{fold_data_dir}/test_latent.npy")
    y_test = np.load(f"{fold_data_dir}/test_labels.npy")
    
    # Flatten labels if needed
    if y_train.ndim > 1:
        y_train = y_train.ravel()
    if y_test.ndim > 1:
        y_test = y_test.ravel()
    
    print(f"[INFO] Z_train: {Z_train.shape}, y_train: {y_train.shape}")
    print(f"[INFO] Z_test: {Z_test.shape}, y_test: {y_test.shape}")
    
    model_dir = f"saved_models_sheet4/Scenario{scenario}"
    fold_num = fold_idx + 1

    # =====================================================
    # SCENARIO 1 — One-class models
    # =====================================================
    if scenario == 1:
        models = {
            "OCSVM": f"{model_dir}/OCSVM_Fold{fold_num}.joblib",
        }

        for name, path in models.items():
            if not os.path.exists(path):
                print(f"[WARN] Missing {name}: {path}")
                continue

            print(f"\n[INFO] Loading {name} from {path}")
            model = joblib.load(path)
            out_dir = f"exports_sheet4/Task3_Results/Scenario{scenario}/SHAP/Fold{fold_num}/{name}"
            run_shap_for_model(name, model, Z_train, Z_test, out_dir)

        print(f"\n=== TASK 3(c) COMPLETED FOR FOLD {fold_idx} ===")
        return

    # =====================================================
    # SCENARIO 2 & 3 — ML models + CNN
    # =====================================================
    ml_models = {
        "SVM": f"{model_dir}/SVM_Fold{fold_num}.joblib",
        "kNN": f"{model_dir}/kNN_Fold{fold_num}.joblib",
        "RF":  f"{model_dir}/RandomForest_Fold{fold_num}.joblib",
    }

    # ML models
    for name, path in ml_models.items():
        if not os.path.exists(path):
            print(f"[WARN] Missing {name}: {path}")
            continue
        
        print(f"\n[INFO] Loading {name} from {path}")
        model = joblib.load(path)
        out_dir = f"exports_sheet4/Task3_Results/Scenario{scenario}/SHAP/Fold{fold_num}/{name}"
        run_shap_for_model(name, model, Z_train, Z_test, out_dir)

    # CNN
    cnn_path = f"{model_dir}/CNN_Fold{fold_num}.h5"
    if not os.path.exists(cnn_path):
        print(f"[WARN] Missing CNN: {cnn_path}")
    else:
        print(f"\n[INFO] Loading CNN from {cnn_path}")
        cnn_model = load_model(cnn_path)
        
        # CNN operates directly on latent features (not windowed!)
        out_dir = f"exports_sheet4/Task3_Results/Scenario{scenario}/SHAP/Fold{fold_num}/CNN"
        
        run_shap_for_model(
            "CNN",
            cnn_model,
            Z_train,  # Use flat latent features directly
            Z_test,   # Use flat latent features directly
            out_dir,
            flatten=False
        )

    print(f"\n=== TASK 3(c) COMPLETED FOR FOLD {fold_idx} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Task 3(c) — SHAP explanations on fold data")
    parser.add_argument("-sc", "--scenario", type=int, required=True, 
                        choices=[1, 2, 3], help="Scenario number")
    parser.add_argument("-f", "--fold", type=int, default=0,
                        help="Fold index (0-based, default 0 for first available fold)")

    args = parser.parse_args()
    run_task3_c(args.scenario, args.fold)
