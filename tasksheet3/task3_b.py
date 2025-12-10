#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from upsetplot import UpSet


# ---------------------------------------------------------
# Load true error indices (0..N_test-1) for a model fold
# ---------------------------------------------------------
def load_errors(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    y_true = df["Attack"].values
    y_pred = df["predicted_label"].values
    return set(np.where(y_pred != y_true)[0])


# ---------------------------------------------------------
# Find prediction files per model
# ---------------------------------------------------------
def find_prediction_files(folder):
    files = []
    for f in os.listdir(folder):
        if f.endswith(".csv") and "Predictions_Fold" in f and "metrics" not in f.lower():
            files.append(os.path.join(folder, f))
    return sorted(files)


# ---------------------------------------------------------
# Venn diagram (Scenario 1)
# ---------------------------------------------------------
def plot_venn3(err1, err2, err3, labels, out):
    plt.figure(figsize=(7, 7))
    venn3([err1, err2, err3], set_labels=labels)
    plt.title("Classification Error Overlap")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("[SAVED]", out)


# ---------------------------------------------------------
# Correct UpSet Plot using aligned indices
# ---------------------------------------------------------
def plot_upset(error_dict, fold_idx, out):
    import pandas as pd
    import matplotlib.pyplot as plt
    from upsetplot import UpSet

    models = list(error_dict.keys())

    # Build membership table
    rows = []
    for m in models:
        for idx in error_dict[m]:
            rows.append({"id": idx, m: True})

    df = pd.DataFrame(rows)
    df = df.pivot_table(index="id", aggfunc="max").fillna(False)

    # Convert to MultiIndex
    df.index = pd.MultiIndex.from_frame(df[models])
    df = df.drop(columns=models)

    # MUCH BIGGER FIGURE + tuning layout
    plt.figure(figsize=(14, 10))
    upset = UpSet(
        df,
        subset_size="count",
        show_counts=True,
        sort_by="cardinality",
        facecolor="black",
        with_lines=True,
        intersection_plot_elements=12,   # make intersections bigger
        element_size=40                 # enlarge dots
    )
    upset.plot()

    plt.suptitle(f"Classification Error Overlap – Fold {fold_idx}", fontsize=18, y=1.02)
    plt.savefig(out, bbox_inches="tight", dpi=200)
    plt.close()

    print("[SAVED]", out)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def run_task3b(scenario):

    base_dir = f"exports/Scenario{scenario}"
    out_dir = f"task3b_errors/Scenario{scenario}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Processing Scenario {scenario}")

    if scenario == 1:
        model_paths = {
            "OCSVM": os.path.join(base_dir, "OCSVM"),
            "LOF": os.path.join(base_dir, "LOF"),
            "EE": os.path.join(base_dir, "EllipticEnvelope")
        }
    else:
        model_paths = {
            "SVM": os.path.join(base_dir, "SVM"),
            "kNN": os.path.join(base_dir, "kNN"),
            "RF": os.path.join(base_dir, "RandomForest"),
            "CNN": os.path.join(base_dir, "CNN")
        }

    # Load folds
    folds = {m: find_prediction_files(folder) for m, folder in model_paths.items()}
    num_folds = len(next(iter(folds.values())))

    # ---------------------------------------------------------
    # Process each fold
    # ---------------------------------------------------------
    for f in range(num_folds):
        print(f"\n[INFO] FOLD {f+1}")

        errors = {}
        for model in model_paths:
            pred_file = folds[model][f]
            print(f"[LOADING] {model} → {pred_file}")
            errors[model] = load_errors(pred_file)

        # Scenario 1 → Venn
        if scenario == 1:
            out = f"{out_dir}/venn_scenario1_fold{f+1}.png"
            plot_venn3(errors["OCSVM"], errors["LOF"], errors["EE"],
                       labels=("OCSVM", "LOF", "EllipticEnvelope"),
                       out=out)

        # Scenario 2 & 3 → UpSet
        else:
            out = f"{out_dir}/upset_scenario{scenario}_fold{f+1}.png"
            plot_upset(errors, f+1, out)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 3(b): Error Overlap")
    parser.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()
    run_task3b(args.scenario)
