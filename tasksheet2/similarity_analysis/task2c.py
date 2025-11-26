import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Load dataset and construct attack flag
# ---------------------------------------------------------
def load_dataset(folder_path):
    from haien_mapping import haien_mapping_dict

    dfs = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv") and not file.startswith("label"):
            print(f"  Loading {file}")
            dfs.append(pd.read_csv(os.path.join(folder_path, file)))
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined {len(dfs)} files → shape {df.shape}")

    # load continuous sensors
    sensor_file = os.path.join(folder_path, "continuous_sensors.txt")
    with open(sensor_file) as f:
        sensors = [s.strip() for s in f if s.strip()]

    # detect time column
    time_col = next((c for c in df.columns if "time" in c.lower()), None)

    # HAIEnd mapping
    if "haiend" in folder_path.lower():
        mapped_sensors = []
        missing = []
        for s in sensors:
            mapped = haien_mapping_dict.get(s, s)
            if mapped in df.columns:
                mapped_sensors.append(mapped)
            elif s in df.columns:
                mapped_sensors.append(s)
            else:
                missing.append(s)
        if missing:
            print(f"Warning: {len(missing)} sensors not found in haiend-23.05: {missing}")
        sensors = mapped_sensors

        # label handling
        label_files = [f for f in os.listdir(folder_path) if "label-test" in f]
        if label_files:
            label_dfs = []
            for lf in sorted(label_files):
                label_dfs.append(pd.read_csv(
                    os.path.join(folder_path, lf),
                    usecols=lambda c: "time" in c.lower() or "label" in c.lower(),
                    engine="python",
                    on_bad_lines="skip"
                ))
            label_df = pd.concat(label_dfs, ignore_index=True)
            label_time_col = next((c for c in label_df.columns if "time" in c.lower()), None)
            label_flag_col = next((c for c in label_df.columns if "label" in c.lower()), None)

            if time_col and label_time_col:
                df = df.merge(label_df[[label_time_col, label_flag_col]],
                              left_on=time_col, right_on=label_time_col, how="left")
                df["attack_flag"] = (df[label_flag_col].fillna(0).astype(int) == 1)
            else:
                min_len = min(len(df), len(label_df))
                df = df.iloc[:min_len].copy()
                df["attack_flag"] = (label_df[label_flag_col].iloc[:min_len].fillna(0).astype(int) == 1)
        else:
            print("No label-test files found; assuming all normal.")
            df["attack_flag"] = False

    elif "21.03" in folder_path:
        atk_cols = [c for c in df.columns if c.lower().startswith("attack")]
        df["attack_flag"] = df[atk_cols].sum(axis=1) > 0

    elif "22.04" in folder_path:
        if "Attack" in df.columns:
            df["attack_flag"] = (df["Attack"].astype(int) == 1)
        else:
            atk_cols = [c for c in df.columns if "attack" in c.lower()]
            df["attack_flag"] = df[atk_cols].sum(axis=1) > 0
    else:
        df["attack_flag"] = False

    df_sensors = df[sensors].fillna(0)
    return df_sensors, df["attack_flag"].to_numpy()


# ---------------------------------------------------------
# Run t-SNE with hyperparameter sweep
# ---------------------------------------------------------
def run_tsne(X, perplexities=(30, 50), lrs=(200, 400), exaggerations=(4, 8)):
    best_emb, best_kl = None, float("inf")
    for p in perplexities:
        for lr in lrs:
            for ex in exaggerations:
                print(f"Running t-SNE: perplexity={p}, lr={lr}, exaggeration={ex}")
                tsne = TSNE(
                    n_components=2,
                    perplexity=p,
                    early_exaggeration=ex,
                    learning_rate=lr,
                    max_iter=1000,
                    init="pca",
                    random_state=42,
                    n_jobs=8
                )
                emb = tsne.fit_transform(X)
                kl = getattr(tsne, "kl_divergence_", np.nan)
                if kl < best_kl:
                    best_kl, best_emb = kl, emb
    print(f"Best t-SNE KL divergence: {best_kl:.4f}")
    return best_emb


# ---------------------------------------------------------
# Plot t-SNE embeddings
# ---------------------------------------------------------
def plot_tsne(embs, labels, colors, color_map, title_suffix, filename_suffix):
    plt.figure(figsize=(10, 8))

    for name, color in color_map.items():
        mask = colors == color
        normal_mask = mask & (~labels)
        attack_mask = mask & labels

        # Filter according to filename_suffix
        if "normal" in filename_suffix:
            plt.scatter(embs[normal_mask, 0], embs[normal_mask, 1],
                        s=6, c=color, alpha=0.6, label=f"{name} (normal)")

        elif "attack" in filename_suffix:
            plt.scatter(embs[attack_mask, 0], embs[attack_mask, 1],
                        s=8, c=color, marker="x", alpha=0.8, label=f"{name} (attack)")

        else:  # "all"
            plt.scatter(embs[normal_mask, 0], embs[normal_mask, 1],
                        s=6, c=color, alpha=0.4, label=f"{name} (normal)")
            plt.scatter(embs[attack_mask, 0], embs[attack_mask, 1],
                        s=8, c=color, marker="x", alpha=0.8, label=f"{name} (attack)")

    plt.title(f"t-SNE Visualization of HAI Datasets ({title_suffix})")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.legend(markerscale=2, fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    filename = f"task2c_tsne_{filename_suffix}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.close()



# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))

    versions = ["hai-21.03"]
    color_map = {
        "hai-21.03": "tab:blue",
        "hai-22.04": "tab:green",
        "haiend-23.05": "tab:red"
    }

    all_embs, all_labels, all_colors = [], [], []
    for version in versions:
        print(f"\nProcessing {version} ...")
        folder = os.path.join(base, version)
        X, attack_flags = load_dataset(folder)

        if len(X) > 30000:
            idx = sorted(random.sample(range(len(X)), 30000))
            X, attack_flags = X.iloc[idx], attack_flags[idx]

        X_scaled = StandardScaler().fit_transform(X)
        emb = run_tsne(X_scaled)
        all_embs.append(emb)
        all_labels.append(attack_flags)
        all_colors.append(np.full(len(emb), color_map[version]))

    all_embs = np.vstack(all_embs)
    all_labels = np.concatenate(all_labels)
    all_colors = np.concatenate(all_colors)

    plot_tsne(all_embs, all_labels, all_colors, color_map, "Normal + Attack", "all")
    plot_tsne(all_embs, all_labels, all_colors, color_map, "Normal Only", "normal")
    plot_tsne(all_embs, all_labels, all_colors, color_map, "Attack Only", "attack")

    print("All t-SNE plots generated successfully.")

