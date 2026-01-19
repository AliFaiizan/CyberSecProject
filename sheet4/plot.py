import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ==========================================================================================
# 1) BAR PLOT: PRECISION & RECALL PER MODEL
# ==========================================================================================
def plot_comparative_metrics(scenario, base_dir="exports_sheet4"):

    if scenario == 1:
        models = ["OCSVM", "LOF", "EllipticEnvelope", "NGRAM_N2", "NGRAM_N5", "NGRAM_N8", "syn_NGRAM_N2", "syn_NGRAM_N5", "syn_NGRAM_N8"]
    else:
        models = ["SVM", "kNN", "RandomForest", "CNN"]

    ensemble_methods = ["random", "majority", "all"]
    model_labels = models + [f"Ensemble_{m}" for m in ensemble_methods]

    precisions, recalls = [], []

    for label in model_labels:

        # ===============================
        # ENSEMBLE MODELS
        # ===============================
        if label.startswith("Ensemble_"):
            method = label.split("_")[1]
            fold_prec, fold_rec = [], []

            for fold in range(1, 6):
                path = f"{base_dir}/Scenario{scenario}/Ensemble/Predictions_{method}_Fold{fold}.csv"
                if not os.path.exists(path):
                    continue

                df = pd.read_csv(path)
                pred, true = df["predicted_label"].values, df["Attack"].values

                tp = ((pred == 1) & (true == 1)).sum()
                fp = ((pred == 1) & (true == 0)).sum()
                fn = ((pred == 0) & (true == 1)).sum()

                fold_prec.append(tp / (tp + fp + 1e-9))
                fold_rec.append(tp / (tp + fn + 1e-9))

            precisions.append(np.mean(fold_prec))
            recalls.append(np.mean(fold_rec))
            continue

        # ===============================
        # NGRAM (explicit N)
        # ===============================
        if label.startswith("NGRAM_"):
            N = label.split("_")[1]  # N2 / N5 / N8
            fold_prec, fold_rec = [], []

            for fold in range(1, 6):
                path = f"{base_dir}/Scenario{scenario}/NGRAM/Predictions_{N}_Fold{fold}.csv"
                if not os.path.exists(path):
                    continue

                df = pd.read_csv(path)
                pred, true = df["predicted_label"].values, df["Attack"].values

                tp = ((pred == 1) & (true == 1)).sum()
                fp = ((pred == 1) & (true == 0)).sum()
                fn = ((pred == 0) & (true == 1)).sum()

                fold_prec.append(tp / (tp + fp + 1e-9))
                fold_rec.append(tp / (tp + fn + 1e-9))

            precisions.append(np.mean(fold_prec))
            recalls.append(np.mean(fold_rec))
            continue

        if label.startswith("syn_NGRAM_"):
            N = label.split("_")[2]  # N2 / N5 / N8
            fold_precisions, fold_recalls = [], []

            for fold_idx in range(1, 6):
                path = f"{base_dir}/Scenario{scenario}/syn_NGRAM/Predictions_{N}_Fold{fold_idx}.csv"
                if not os.path.exists(path):
                    continue

                df = pd.read_csv(path)
                pred, true = df["predicted_label"].values, df["Attack"].values

                tp = ((pred == 1) & (true == 1)).sum()
                fp = ((pred == 1) & (true == 0)).sum()
                fn = ((pred == 0) & (true == 1)).sum()

                precision = tp / (tp + fp + 1e-9)
                recall = tp / (tp + fn + 1e-9)

                fold_precisions.append(precision)
                fold_recalls.append(recall)

            precisions.append(np.mean(fold_precisions))
            recalls.append(np.mean(fold_recalls))
            continue
        # ===============================
        # ML / CNN MODELS
        # ===============================
        metrics_path = f"{base_dir}/Scenario{scenario}/{label}/metrics_summary.csv"

        if not os.path.exists(metrics_path):
            precisions.append(0)
            recalls.append(0)
            continue

        df = pd.read_csv(metrics_path)
        precisions.append(df["precision"].mean())
        recalls.append(df["recall"].mean())

    # ===============================
    # PLOT
    # ===============================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(model_labels))

    axes[0].bar(x, precisions)
    axes[0].set_title(f"Scenario {scenario} — Precision Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_labels, rotation=45, ha='right')
    axes[0].set_ylim(0, 1)

    axes[1].bar(x, recalls)
    axes[1].set_title(f"Scenario {scenario} — Recall Comparison")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_labels, rotation=45, ha='right')
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"Scenario{scenario}_comparative_metrics.png", dpi=300)
    plt.close()


# ==========================================================================================
# 2) RUNTIME / MEMORY
# ==========================================================================================
def plot_runtime_memory(scenario, base_dir="exports"):

    if scenario == 1:
        models = ["OCSVM", "LOF", "EllipticEnvelope"]
    else:
        models = ["SVM", "kNN", "RandomForest", "CNN"]

    runtimes, memories, names = [], [], []

    for model in models:
        path = f"{base_dir}/Scenario{scenario}/{model}/metrics_summary.csv"
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        runtimes.append(df["runtime_sec"].mean())
        memories.append(df["memory_bytes"].mean() / 1e6)
        names.append(model)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].bar(names, runtimes)
    axes[0].set_title(f"Scenario {scenario} — Runtime")

    axes[1].bar(names, memories)
    axes[1].set_title(f"Scenario {scenario} — Memory Usage (MB)")

    plt.tight_layout()
    plt.savefig(f"Scenario{scenario}_runtime_memory.png", dpi=300)
    plt.close()


# ==========================================================================================
# 3) FOLD-WISE PERFORMANCE
# ==========================================================================================
def plot_fold_performance(scenario, base_dir="exports"):

    if scenario == 1:
        models = ["OCSVM", "LOF", "EllipticEnvelope"]
    else:
        models = ["SVM", "kNN", "RandomForest", "CNN"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, model in enumerate(models):
        path = f"{base_dir}/Scenario{scenario}/{model}/metrics_summary.csv"
        if not os.path.exists(path):
            axes[idx].set_visible(False)
            continue

        df = pd.read_csv(path)
        axes[idx].plot(df["fold"], df["precision"], marker="o", label="Precision")
        axes[idx].plot(df["fold"], df["recall"], marker="s", label="Recall")
        axes[idx].set_title(model)
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(f"Scenario{scenario}_fold_performance.png", dpi=300)
    plt.close()


# ==========================================================================================
# 4) ENSEMBLE COMPARISON
# ==========================================================================================
def plot_ensemble_comparison(scenario, base_dir="exports"):

    ensemble_dir = f"{base_dir}/Scenario{scenario}/Ensemble"
    methods = ["random", "majority", "all"]

    data = []

    for method in methods:
        prec, rec = [], []

        for fold in range(1, 6):
            path = f"{ensemble_dir}/Predictions_{method}_Fold{fold}.csv"
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            pred, true = df["predicted_label"].values, df["Attack"].values

            tp = ((pred == 1) & (true == 1)).sum()
            fp = ((pred == 1) & (true == 0)).sum()
            fn = ((pred == 0) & (true == 1)).sum()

            prec.append(tp / (tp + fp + 1e-9))
            rec.append(tp / (tp + fn + 1e-9))

        data.append([method, np.mean(prec), np.mean(rec)])

    df = pd.DataFrame(data, columns=["Method", "Precision", "Recall"])
    df.plot(x="Method", y=["Precision", "Recall"], kind="bar")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"Scenario{scenario}_ensemble_comparison.png", dpi=300)
    plt.close()


# ==========================================================================================
# MASTER
# ==========================================================================================
def generate_all_plots(scenarios=[1, 2, 3], base_dir="exports_sheet4"):
    for sc in scenarios:
        plot_comparative_metrics(sc, base_dir)
        plot_fold_performance(sc, base_dir)
        plot_ensemble_comparison(sc, base_dir)
    print("All plots generated successfully!")


if __name__ == "__main__":
    generate_all_plots([1, 2, 3])
