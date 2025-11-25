import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Auto bin calculation (Freedman–Diaconis rule)
# ---------------------------------------------------------
def auto_bins(data):
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data) ** (1 / 3))
    if bin_width <= 0:
        return 50
    n_bins = int((data.max() - data.min()) / bin_width)
    return max(50, min(n_bins, 400))

# ---------------------------------------------------------
# Manual histogram (no np.histogram)
# ---------------------------------------------------------
def manual_hist(data, bins, range_=None):
    if range_ is None:
        min_v, max_v = float(data.min()), float(data.max())
    else:
        min_v, max_v = range_
    width = (max_v - min_v) / bins
    counts = np.zeros(bins, dtype=int)
    for val in data:
        idx = int((val - min_v) / width)
        if 0 <= idx < bins:
            counts[idx] += 1
    centers = np.linspace(min_v + width/2, max_v - width/2, bins)
    return centers, counts

# ---------------------------------------------------------
# Simple moving average smoothing
# ---------------------------------------------------------
def smooth(y, window=5):
    return np.convolve(y, np.ones(window) / window, mode='same')

# ---------------------------------------------------------
# Load computed Spearman distances
# ---------------------------------------------------------
def load_dataset_arrays(base):
    datasets = ["hai-21.03", "hai-22.04", "haiend-23.05"]
    data_dict = {}
    for d in datasets:
        normal_path = os.path.join(base, f"spearman_{d}_normal.npy")
        attack_path = os.path.join(base, f"spearman_{d}_attack.npy")
        if not os.path.exists(normal_path):
            continue
        data_dict[d] = {"normal": np.load(normal_path)}
        if os.path.exists(attack_path):
            data_dict[d]["attack"] = np.load(attack_path)
    return data_dict

# ---------------------------------------------------------
# Main plotting
# ---------------------------------------------------------
def main():
    base = (r"C:\Users\Aser\Documents\Study_Project\src"
            if os.name == "nt"
            else "/home/safiamed/Study_Project/src")

    data_dict = load_dataset_arrays(base)
    plt.figure(figsize=(10, 6))
    print("=== Optimized Spearman Distance Histogram ===")

    for name, parts in data_dict.items():
        normal = parts.get("normal")
        if normal is None or len(normal) == 0:
            continue

        # Shared range for consistent scaling between normal & attack
        data_range = (normal.min(), normal.max())
        bins = auto_bins(normal)

        c_n, h_n = manual_hist(normal, bins, range_=data_range)
        plt.plot(c_n, smooth(h_n / h_n.sum()), label=f"{name} (normal)")

        attack = parts.get("attack")
        if attack is not None and len(attack) > 0:
            c_a, h_a = manual_hist(attack, bins, range_=data_range)
            plt.plot(c_a, smooth(h_a / h_a.sum()), "--", label=f"{name} (attack)")

        print(f"{name}: {bins} bins | normal={len(normal)} | attack={len(attack) if attack is not None else 0}")

    plt.title("Distribution of Consecutive Spearman Distances")
    plt.xlabel("Spearman Distance (1 - ρ)")
    plt.ylabel("Normalized Frequency")
    plt.legend()
    plt.grid(True, alpha=0.4)

    # Logarithmic x-axis highlights tail behavior
    plt.xscale("log")
    plt.xlim(1e-4, 0.05)
    plt.ylim(bottom=0)
    plt.tight_layout()

    out_path = os.path.join(base, "spearman_histogram_all_datasets.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved optimized histogram → {out_path}")

    try:
        plt.show()
    except Exception:
        print("No GUI available (saved only).")

if __name__ == "__main__":
    main()
