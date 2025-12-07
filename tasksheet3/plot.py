import os
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Helper for plotting Precision/Recall for one model
# ---------------------------------------------------------
def plot_model_metrics(model_name, df, scenario):
    plt.figure(figsize=(8, 5))
    plt.plot(df["fold"], df["precision"], marker="o", label="Precision")
    plt.plot(df["fold"], df["recall"], marker="o", label="Recall")
    plt.title(f"{model_name} – Scenario {scenario}")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.show()


# ---------------------------------------------------------
# Helper to load a model's metrics file
# ---------------------------------------------------------
def load_metrics(path):
    if not os.path.exists(path):
        print(f"[WARN] Metrics not found: {path}")
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------
# Plot ensemble metrics (random / majority / all)
# ---------------------------------------------------------
def plot_ensemble(base_dir, scenario):
    ensemble_dir = os.path.join(base_dir, "Ensemble")
    methods = ["random", "majority", "all"]

    for method in methods:
        rows = []
        for fold in range(1, 6):
            file = f"{ensemble_dir}/Ensemble_{method}_Fold{fold}.csv"
            if not os.path.exists(file):
                continue

            df = pd.read_csv(file)
            preds = df["predicted_label"].values
            true = df["Attack"].values

            tp = ((preds == 1) & (true == 1)).sum()
            fp = ((preds == 1) & (true == 0)).sum()
            fn = ((preds == 0) & (true == 1)).sum()

            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)

            rows.append({"fold": fold, "precision": precision, "recall": recall})

        if not rows:
            continue

        df = pd.DataFrame(rows)
        plot_model_metrics(f"Ensemble ({method})", df, scenario)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot Task 2 metrics for any scenario")
    parser.add_argument("-sc", "--scenario", required=True, type=int, choices=[1, 2, 3])
    args = parser.parse_args()

    scenario = args.scenario
    base_dir = f"exports/Scenario{scenario}"

    # -----------------------------------------------------
    # MODELS per scenario
    # -----------------------------------------------------
    if scenario == 1:
        models = ["OCSVM", "LOF", "EllipticEnvelope"]
    else:
        models = ["SVM", "kNN", "RandomForest", "CNN"]  # CNN only exists in 2 & 3

    # -----------------------------------------------------
    # Plot ML + CNN
    # -----------------------------------------------------
    for m in models:
        metrics_path = os.path.join(base_dir, m, "metrics_summary.csv")
        df = load_metrics(metrics_path)
        if df is not None:
            plot_model_metrics(m, df, scenario)

    # -----------------------------------------------------
    # Plot Ensemble (always available)
    # -----------------------------------------------------
    plot_ensemble(base_dir, scenario)


if __name__ == "__main__":
    main()
