#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA


# ================================================================
# FEATURE IMPORTANCE HELPERS
# ================================================================

def perturbation_importance(model, X, score_type="prediction"):
    n = X.shape[1]
    importance = np.zeros(n)

    if score_type == "prediction":
        base = model.predict(X)
    else:
        base = model.decision_function(X)

    for i in range(n):
        Xp = X.copy()
        Xp[:, i] += np.random.normal(0, 0.1, size=len(X))

        if score_type == "prediction":
            new = model.predict(Xp)
            importance[i] = (base != new).sum()
        else:
            new = model.decision_function(Xp)
            importance[i] = np.abs(base - new).sum()

    return importance


def lof_importance(model, X):
    base_pred = model.predict(X)
    n = X.shape[1]
    imp = np.zeros(n)

    for i in range(n):
        Xp = X.copy()
        Xp[:, i] += np.random.normal(0, 0.1, size=len(X))

        new_pred = model.predict(Xp)
        imp[i] = (base_pred != new_pred).sum()

    return imp


def compute_importance_pca_space(model, model_name, X_pca, y):
    name = model_name.lower()

    if "randomforest" in name:
        return model.feature_importances_

    if "svm" in name and "oc" not in name:
        result = permutation_importance(model, X_pca, y, n_repeats=10)
        return result.importances_mean

    if "knn" in name:
        return perturbation_importance(model, X_pca, score_type="prediction")

    if "ocsvm" in name:
        return perturbation_importance(model, X_pca, score_type="decision")

    if "elliptic" in name:
        return np.abs(np.diag(model.precision_))

    if "lof" in name:
        return lof_importance(model, X_pca)

    raise ValueError(f"No FI method for model {model_name}")


def pca_importance_to_raw(pca, fi_pca):
    """
    Back-project FI from PCA space into original latent features.
    raw_FI = | PCA.components_.T × FI_pca |
    """
    components = pca.components_         # shape: (num_pcs, 8)
    return np.abs(components.T @ fi_pca)  # → shape (8,)


# ================================================================
# PLOTTING
# ================================================================

def plot_importance(importance, names, model_name, out_path):
    idx = np.argsort(importance)[::-1]
    imp_sorted = importance[idx]
    names_sorted = [names[i] for i in idx]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(imp_sorted)), imp_sorted)
    plt.xticks(range(len(imp_sorted)), names_sorted, rotation=90)
    plt.title(f"Feature Importance — {model_name}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ================================================================
# MAIN PIPELINE
# ================================================================

def run_task3_partA(latent_path, model_dir, scenario_id):

    print("\n=== Loading latent features ===")
    Z = np.load(latent_path)
    print("Latent shape:", Z.shape)

    # ------------------------------------------------------------
    # Load TRUE LABELS directly from original dataset (correct!)
    # ------------------------------------------------------------
    print("Loading labels from original dataset (train1 + test1)...")
    from utils import load_data

    _, y = load_data(
        ["../../datasets/hai-22.04/train1.csv"],
        ["../../datasets/hai-22.04/test1.csv"]
    )
    y = np.array(y).astype(int)

    feature_names = [f"f{i}" for i in range(Z.shape[1])]

    out_dir = f"Task3_Results/Scenario{scenario_id}/FeatureImportance"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Using models from {model_dir} ===")
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".joblib")])

    # ------------------------------------------------------------
    # For each saved model (each fold)
    # ------------------------------------------------------------
    for mfile in model_files:
        model_name = mfile.replace(".joblib", "")
        fold = int(model_name.split("_Fold")[-1])
        model = load(os.path.join(model_dir, mfile))

        print(f"\n[+] Processing {model_name}")

        # Load Task2 prediction file to recover test indices
        pred_path = f"exports/Scenario{scenario_id}/{model_name.split('_Fold')[0]}/Predictions_Fold{fold}.csv"
        df_pred = pd.read_csv(pred_path)
        test_idx = df_pred.index.values
        train_idx = np.setdiff1d(np.arange(len(Z)), test_idx)

        # ----------------------------------------------------
        # Rebuild PCA for that fold (same as Task2)
        # ----------------------------------------------------
        pca = PCA(n_components=0.95)
        pca.fit(Z[train_idx])
        Z_pca = pca.transform(Z)

        # ----------------------------------------------------
        # Compute feature importance in PCA space
        # ----------------------------------------------------
        fi_pca = compute_importance_pca_space(model, model_name, Z_pca, y)

        # ----------------------------------------------------
        # Convert PCA FI → raw latent FI
        # ----------------------------------------------------
        fi_raw = pca_importance_to_raw(pca, fi_pca)

        # ----------------------------------------------------
        # Save plot for f0–f7
        # ----------------------------------------------------
        out_path = f"{out_dir}/FI_{model_name}.png"
        plot_importance(fi_raw, feature_names, model_name, out_path)

        print(f"   Saved: {out_path}")

    print("\n=== Task 3(a) Complete — PCA Back-Projected FI ===")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 3(a) Feature Importance with PCA Back-Projection")
    parser.add_argument("--latent", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    run_task3_partA(args.latent, args.models, args.scenario)
