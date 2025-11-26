#!/usr/bin/env python3
"""
Integrated ICS Security Toolbox
Includes:
- Practical Sheet 1 tasks
- Practical Sheet 2 Task 1 ML models
"""

import argparse
import sys
import os
import glob
import re


DATASET_CONFIG = {
    "hai-21.03": {
        "folder": "../datasets/hai-21.03",
        "train_pattern": "train*.csv",
        "test_pattern": "test*.csv",
        "script": "./statistical_analysis/task1.py",
        "has_labels": False
    },
    "hai-22.04": {
        "folder": "../datasets/hai-22.04",
        "train_pattern": "train*.csv",
        "test_pattern": "test*.csv",
        "script": "./statistical_analysis/task1-22.py",
        "has_labels": False
    },
    "haiend-23.05": {
        "folder": "../datasets/haiend-23.05",
        "train_pattern": "end-train*.csv",
        "test_pattern": "end-test*.csv",
        "label_pattern": "label-test*.csv",
        "script": "./statistical_analysis/task1-23.py",
        "has_labels": True
    }
}


# ==============================================================
# MASTER TOOLBOX ENTRY POINT
# ==============================================================
def main():
    # Check for help flag before parsing (to avoid requiring mode)
    if "-h" in sys.argv or "--help" in sys.argv:
        print_help()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="ICS Security Toolbox - Integrated Analysis and ML Detection",
        add_help=False
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Select operation mode")

    # ===== ANALYZE =====
    analyze_parser = subparsers.add_parser("analyze", help="Run dataset analysis (Task Sheet 1)")
    analyze_parser.add_argument("--dataset", choices=DATASET_CONFIG.keys(), required=True)

    # ===== SIMILARITY =====
    similarity_parser = subparsers.add_parser("similarity", help="Run similarity analysis")
    similarity_parser.add_argument("--dataset", choices=DATASET_CONFIG.keys(), required=True)

    # ===== NGRAM =====
    ngram_parser = subparsers.add_parser("ngram", help="Run n-gram based detection")
    ngram_parser.add_argument("--dataset", choices=DATASET_CONFIG.keys(), required=True)
    ngram_parser.add_argument("--ngram-order", type=int, default=2)

    # ===== ML =====
    ml_parser = subparsers.add_parser("ml", help="Run machine learning detection (Task Sheet 2)")
    ml_parser.add_argument("-m", "--model", choices=['ocsvm','lof','ee','knn','svm','rf'], required=True)
    ml_parser.add_argument("-sc", "--scenario", choices=['1','2','3'], required=True)
    ml_parser.add_argument("-k", "--kfold", type=int, default=5)
    ml_parser.add_argument("-e", "--export", choices=['1','2','3'])

    # ===== DL =====
    dl_parser = subparsers.add_parser("dl", help="Run deep learning CNN (Task Sheet 2)")
    dl_parser.add_argument("-sc", "--scenario", required=True, choices=['1','2','3'])
    dl_parser.add_argument("-M", "--window-size", required=True)
    dl_parser.add_argument("-e", "--epochs", default=5)
    dl_parser.add_argument("--stride", default=10)

    # ===== ENSEMBLE =====
    ensemble_parser = subparsers.add_parser("ensemble", help="Run Task 2 Ensemble Classifier")
    ensemble_parser.add_argument("-sc", "--scenario", required=True, choices=['1','2','3'])
    ensemble_parser.add_argument("-m", "--method", required=True, choices=["random","majority","all"])
    ensemble_parser.add_argument("-f", "--fold", required=True, type=int)

    args = parser.parse_args()
    print(args)

    # Handle modes
    if args.mode == "analyze":
        cfg = DATASET_CONFIG[args.dataset]
        run_task1(cfg["script"])

    elif args.mode == "similarity":
        run_similarity(args.dataset)

    elif args.mode == "ngram":
        run_ngram(args.dataset, args.ngram_order)

    elif args.mode == "ml":
        run_ml(args)

    elif args.mode == "dl":
        run_dl(args)

    elif args.mode == "ensemble":
        run_ensemble(args)

    else:
        print("Unknown mode")


# ==============================================================
# TASK SHEET 1 FUNCTIONS
# ==============================================================
def run_task1(script):
    print(f"Running dataset analysis: {script}")
    os.system(f"python {script}")


def run_similarity(dataset_name):
    os.system("python similarity_analysis/find_sensors.py")
    # modify files inside similarity_analysis/
    modify_spearman(dataset_name)
    modify_tsne(dataset_name)
    modify_pca(dataset_name)

    # run modules inside the folder
    os.system("python similarity_analysis/spearman_distance.py")
    os.system("python similarity_analysis/histogram.py")
    os.system("python similarity_analysis/task2c.py")
    os.system("python similarity_analysis/task2d.py")

def modify_spearman(version):
    path = "similarity_analysis/spearman_distance.py"
    with open(path, "r") as f:
        txt = f.read()
    txt = re.sub(r'version\s*=\s*".*"', f'version = \"{version}\"', txt)
    with open(path, "w") as f:
        f.write(txt)

def modify_tsne(version):
    path = "similarity_analysis/task2c.py"
    with open(path, "r") as f:
        txt = f.read()
    txt = re.sub(r'versions\s*=\s*\[.*?\]', f'versions = [\"{version}\"]', txt)
    with open(path, "w") as f:
        f.write(txt)

def modify_pca(version):
    path = "similarity_analysis/task2d.py"
    with open(path, "r") as f:
        txt = f.read()
    txt = re.sub(r'versions\s*=\s*\[.*?\]', f'versions = [\"{version}\"]', txt)
    with open(path, "w") as f:
        f.write(txt)


def run_ngram(version, n):
    with open("./similarity_analysis/task3.py", "r") as f:
        txt = f.read()

    txt = re.sub(r"N_GRAM_ORDER\s*=\s*\d+", f"N_GRAM_ORDER = {n}", txt)
    folder = DATASET_CONFIG[version]["folder"]
    txt = re.sub(r'DATA_DIR\s*=\s*r".*"', f'DATA_DIR = r"{folder}"', txt)

    with open("./similarity_analysis/task3.py", "w") as f:
        f.write(txt)

    os.system("python ./similarity_analysis/task3.py")

# ==============================================================
# TASK SHEET 2 – TASK 1 (ML)
# ==============================================================
def run_ml(args):
    print("\nRunning Task Sheet 2 – Machine Learning (Task 1)")

    if args.model is None:
        print("Error: --model (model) required for ML mode.")
        return
    if args.scenario is None:
        print("Error: --scenario (scenario) required for ML mode.")
        return
    if args.kfold is None:
        print("Error: --kfold (k) required for ML mode.")
        return

    import task1
    task1.run_from_toolbox(
        model=args.model,
        scenario=args.scenario,
        k=args.kfold,
        export=args.export
    )

def run_dl(args):
    print("\nRunning Task Sheet 2 – Deep Learning (Task 2)")
    import task2
    task2.run_from_toolbox(
        scenario=args.scenario,
        M=args.window_size,
        epochs=args.epochs,
        stride=args.stride
    )

def run_ensemble(args):
    print("\nRunning Task Sheet 2 – Ensemble Classifier")
    import task2_ensemble
    task2_ensemble.run_from_toolbox(
        scenario=args.scenario,
        method=args.method,
        fold=args.fold
    )


# ==============================================================
# HELP
# ==============================================================
def print_help():
    print("""
ICS Security Toolbox 
==========================================

Usage: python toolbox.py <mode> [options]

Modes:
  analyze     – Dataset statistical analysis
  similarity  – Spearman / PCA / t-SNE
  ngram       – N-gram anomaly detection
  ml          – Traditional ML (Task 2)
  dl          – Deep Learning CNN (Task 2)
  ensemble    – Ensemble ML Classifier (Task 2)

Examples:
  python toolbox.py ml -m ocsvm -sc 1 -k 5
  python toolbox.py dl -sc 2 -M 50 -e 5 --stride 10
  python toolbox.py ensemble -sc 2 -m majority -f 3
""")


if __name__ == "__main__":
    main()
