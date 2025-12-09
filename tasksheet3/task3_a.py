#!/usr/bin/env python3
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

from utils import load_data
from scenarios import scenario_1_split, scenario_2_split, scenario_3_split


# =====================================================================
# Ensure output folder exists
# =====================================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =====================================================================
# Compute feature importance on latent features (no PCA)
# =====================================================================
def compute_importance(model, model_name, X_test, y_test, scenario_id):
    """
    Computes feature importance directly on latent features.
    """

    # Scenario 1 (OC models) → use f1_macro
    scoring_method = "f1_macro" if scenario_id == 1 else "f1"

    # RandomForest supports native feature_importances_
    if "RandomForest" in model_name:
        return model.feature_importances_

    # Linear SVM supports coef_
    if "SVM" in model_name and hasattr(model, "coef_"):
        return np.abs(model.coef_).ravel()

    # All other models → permutation importance
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        scoring=scoring_method,
        random_state=42
    )

    return result.importances_mean


# =====================================================================
# Plot importance
# =====================================================================
def plot_importance(values, scenario_id, model_name, fold_idx, out_dir):
    sorted_idx = np.argsort(values)[::-1]
    sorted_vals = values[sorted_idx]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(sorted_vals)), sorted_vals)
    plt.title(f"Scenario {scenario_id} — {model_name} — Fold {fold_idx+1}")
    plt.xlabel("Latent Feature Index (sorted)")
    plt.ylabel("Importance Score")
    plt.tight_layout()

    path = f"{out_dir}/{model_name}_Fold{fold_idx+1}.png"
    plt.savefig(path)
    plt.close()


# =====================================================================
# Process a full scenario
# =====================================================================
def process_scenario(
    scenario_id,
    latent_file,
    y,
    scenario_fn,
    model_names
):
    print(f"\n=== Processing Scenario {scenario_id} ===")

    # Load latent features (no PCA anymore)
    Z = np.load(latent_file)
    X_df = pd.DataFrame(Z)

    # Output folder
    out_dir = f"Task3_Results/Scenario{scenario_id}/FeatureImportance"
    ensure_dir(out_dir)

    # Iterate ML models
    for model_name in model_names:
        print(f"\n→ Computing importance for model: {model_name}")

        # Iterate folds
        for split in scenario_fn(X_df, y, k=5):

            # Scenario 1 split
            if scenario_id == 1:
                fold_idx, train_idx, test_idx = split
            else:
                fold_idx, attack_id, train_idx, test_idx = split

            # Extract latent features for this fold
            X_train = Z[train_idx]
            X_test = Z[test_idx]
            y_test = y.iloc[test_idx].values

            # Load trained model
            model_path = f"saved_models/Scenario{scenario_id}/{model_name}_Fold{fold_idx+1}.joblib"
            model = joblib.load(model_path)

            # Compute latent feature importance
            importance_vals = compute_importance(model, model_name, X_test, y_test, scenario_id)

            # Save raw CSV sorted
            df = pd.DataFrame({
                "Latent_Feature": np.arange(len(importance_vals)),
                "Importance": importance_vals
            }).sort_values("Importance", ascending=False)

            df.to_csv(f"{out_dir}/{model_name}_Fold{fold_idx+1}.csv", index=False)

            # Plot importance
            plot_importance(importance_vals, scenario_id, model_name, fold_idx, out_dir)

    print(f"✓ Scenario {scenario_id} completed.")


# =====================================================================
# MAIN SCRIPT
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent_reconstruction",
        required=True,
        help="Latent features for Scenario 1 (reconstruction VAE)"
    )
    parser.add_argument(
        "--latent_classification",
        required=True,
        help="Latent features for Scenarios 2 & 3 (classification VAE)"
    )
    args = parser.parse_args()

    print("Loading labels... (same as Task 2)")
    X_raw, y = load_data(
        ["../../datasets/hai-22.04/train1.csv"],
        ["../../datasets/hai-22.04/test1.csv"]
    )
    y = pd.Series(y).astype(int)

    # Scenario 1
    process_scenario(
        scenario_id=1,
        latent_file=args.latent_reconstruction,
        y=y,
        scenario_fn=scenario_1_split,
        model_names=["OCSVM", "LOF", "EllipticEnvelope"]
    )

    # Scenario 2
    process_scenario(
        scenario_id=2,
        latent_file=args.latent_classification,
        y=y,
        scenario_fn=scenario_2_split,
        model_names=["SVM", "kNN", "RandomForest"]
    )

    # Scenario 3
    process_scenario(
        scenario_id=3,
        latent_file=args.latent_classification,
        y=y,
        scenario_fn=scenario_3_split,
        model_names=["SVM", "kNN", "RandomForest"]
    )

    print("\n=== Task 3(a) Completed Successfully ===")
