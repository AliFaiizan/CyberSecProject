#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from upsetplot import UpSet, from_memberships


# -------------------------------------------------------
# Load prediction errors (ML models)
# -------------------------------------------------------
def load_errors(pred_file):
    df = pd.read_csv(pred_file)
    y_true = df["Attack"].values
    y_pred = df["predicted_label"].values
    return set(np.where(y_pred != y_true)[0]), len(df)


# -------------------------------------------------------
# CNN window → sample mapping
# -------------------------------------------------------
def load_cnn_errors(pred_file, test_indices, M=20):
    df = pd.read_csv(pred_file)
    y_true = df["Attack"].values
    y_pred = df["predicted_label"].values

    win_errors = np.where(y_pred != y_true)[0]
    mapped = set()

    for w in win_errors:
        center = w + M//2
        if center in test_indices:
            mapped.add(center)

    return mapped


# -------------------------------------------------------
# Plot Venn for Scenario 1
# -------------------------------------------------------
def plot_venn_scenario1(fold, errors):
    ocsvm, lof, ee = errors

    plt.figure(figsize=(8, 8))
    venn3(
        subsets=(ocsvm, lof, ee),
        set_labels=("OCSVM", "LOF", "EE"),
        set_colors=("red", "green", "blue"),
        alpha=0.5
    )
    plt.title(f"Scenario 1 – Venn Diagram (Fold {fold})")

    out = "Task3_Results/Scenario1/Venn"
    os.makedirs(out, exist_ok=True)
    plt.savefig(f"{out}/Venn_Fold{fold}.png", dpi=300)
    plt.close()


# -------------------------------------------------------
# Plot UpSet for Scenario 2 & 3
# -------------------------------------------------------
def plot_upset_scenario23(fold, scenario, error_dict):
    # Create membership lists
    memberships = []
    max_index = max([max(s) if len(s) > 0 else 0 for s in error_dict.values()])

    for idx in range(max_index + 1):
        present = [model for model, errs in error_dict.items() if idx in errs]
        if present:
            memberships.append(present)

    if len(memberships) == 0:
        memberships = [[]]

    data = from_memberships(memberships)

    plt.figure(figsize=(12, 6))
    upset = UpSet(data, subset_size='count', show_counts=True)
    upset.plot()

    plt.suptitle(f"Scenario {scenario} – UpSet Plot (Fold {fold})")

    out = f"Task3_Results/Scenario{scenario}/UpSet"
    os.makedirs(out, exist_ok=True)
    plt.savefig(f"{out}/UpSet_Fold{fold}.png", dpi=300)
    plt.close()


# -------------------------------------------------------
# Main Task 3(b)
# -------------------------------------------------------
def run_task3_b(scenario, M=20):
    print(f"\n=== RUNNING TASK 3(b) FOR SCENARIO {scenario} ===")

    base = f"exports/Scenario{scenario}"
    num_folds = 5

    for fold in range(1, num_folds + 1):
        print(f"\n[ Fold {fold} ]")

        # ---------------- SCENARIO 1 --------------------
        if scenario == 1:
            ocsvm = f"{base}/OCSVM/Predictions_Fold{fold}.csv"
            lof   = f"{base}/LOF/Predictions_Fold{fold}.csv"
            ee    = f"{base}/EllipticEnvelope/Predictions_Fold{fold}.csv"

            err_oc, _ = load_errors(ocsvm)
            err_lo, _ = load_errors(lof)
            err_ee, _ = load_errors(ee)

            plot_venn_scenario1(fold, (err_oc, err_lo, err_ee))
            continue

        # ---------------- SCENARIO 2 & 3 --------------------
        svm = f"{base}/SVM/Predictions_Fold{fold}.csv"
        knn = f"{base}/kNN/Predictions_Fold{fold}.csv"
        rf  = f"{base}/RandomForest/Predictions_Fold{fold}.csv"
        cnn = f"{base}/CNN/Predictions_Fold{fold}.csv"

        # Load ML errors
        err_svm, N = load_errors(svm)
        err_knn, _ = load_errors(knn)
        err_rf, _  = load_errors(rf)

        # Test indices
        test_indices = set(range(N))

        # CNN errors
        err_cnn = load_cnn_errors(cnn, test_indices, M)

        error_dict = {
            "SVM": err_svm,
            "kNN": err_knn,
            "RF": err_rf,
            "CNN": err_cnn
        }

        plot_upset_scenario23(fold, scenario, error_dict)

    print(f"\n=== TASK 3(b) COMPLETED FOR SCENARIO {scenario} ===")


# -------------------------------------------------------
# CLI ENTRY
# -------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Task 3(b) – Venn + UpSet")
    parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument("--M", type=int, default=20)

    args = parser.parse_args()
    run_task3_b(args.scenario, args.M)
