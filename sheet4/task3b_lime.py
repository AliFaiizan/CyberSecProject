#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import warnings
import argparse

warnings.filterwarnings('ignore')



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
    
    for idx in range(len(X_test[:1000])): # REMOVE Limit to 1000 samples for speed
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
# MAIN — Task 3(b) with Fold Data
# ========================================================================
def run_task3_b(scenario, fold_idx=0):
    """
    Load pre-generated fold data and trained models.
    Run LIME explanations on fold data only.
    
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

    
    if fold_idx not in available_folds:
        print(f"[ERROR] Fold {fold_idx} not available.")
        return
    
    print(f"\n=== TASK 3(b) — LIME EXPLANATIONS (Scenario {scenario}, Fold {fold_idx}) ===")
    
    # Load fold data
    fold_data_dir = f"exports_sheet4/Scenario{scenario}/fold{fold_idx}"
    
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
            "LOF":   f"{model_dir}/LOF_Fold{fold_num}.joblib",
            "EllipticEnvelope":    f"{model_dir}/EllipticEnvelope_Fold{fold_num}.joblib"
        }

        for name, path in models.items():
            if not os.path.exists(path):
                print(f"[WARN] Missing {name}: {path}")
                continue

            print(f"\n[INFO] Loading {name} from {path}")
            model = joblib.load(path)
            out_dir = f"exports_sheet4/Task3_Results/Scenario{scenario}/LIME/Fold{fold_num}/{name}"
            run_lime_for_model(name, model, Z_train, Z_test, out_dir)

    # =====================================================
    # SCENARIO 2 & 3 — ML models + CNN
    # =====================================================
    else:
        ml_models = {
            "SVM": f"{model_dir}/SVM_Fold{fold_num}.joblib",
            "kNN": f"{model_dir}/kNN_Fold{fold_num}.joblib",
            "RandomForest":  f"{model_dir}/RandomForest_Fold{fold_num}.joblib"
        }

        # ML MODELS
        for name, path in ml_models.items():
            if not os.path.exists(path):
                print(f"[WARN] Missing ML model: {path}")
                continue

            print(f"\n[INFO] Loading {name} from {path}")
            model = joblib.load(path)
            out_dir = f"exports_sheet4/Task3_Results/Scenario{scenario}/LIME/Fold{fold_num}/{name}"
            run_lime_for_model(name, model, Z_train, Z_test, out_dir)

        # CNN MODEL
        cnn_path = f"{model_dir}/CNN_Fold{fold_num}.h5"
        if not os.path.exists(cnn_path):
            print(f"[WARN] Missing CNN model: {cnn_path}")
        else:
            print(f"\n[INFO] Loading CNN from {cnn_path}")
            cnn_model = load_model(cnn_path)
            
            # CNN operates on raw latent features
            out_dir = f"exports_sheet4/Task3_Results/Scenario{scenario}/LIME/Fold{fold_num}/CNN"
            
            n_features = Z_train.shape[1]

            def cnn_predict_fn(x_flat):
                # Reshape to (batch, n_features, 1) as expected by CNN
                x_reshaped = x_flat.reshape((x_flat.shape[0], n_features, 1))
                predictions = cnn_model.predict(x_reshaped, verbose=0)
                # Return probabilities for both classes
                return predictions

            run_lime_for_model(
            "CNN",
            cnn_model,
            Z_train,
            Z_test,
            out_dir,
            predict_fn=cnn_predict_fn,
            flatten=False
        )
       
    print(f"\n=== TASK 3(b) COMPLETED FOR FOLD {fold_idx} ===\n")


# ========================================================================
# CLI
# ========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Task 3(b) — LIME explanations on fold data")
    parser.add_argument("-sc", "--scenario", type=int, required=True, 
                        choices=[1, 2, 3], help="Scenario number")
    parser.add_argument("-f", "--fold", type=int, default=0,
                        help="Fold index (0-based, default 0 for first available fold)")

    args = parser.parse_args()
    run_task3_b(args.scenario, args.fold)
