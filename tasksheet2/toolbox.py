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

    # ===== ANALYZE SUBPARSER =====
    analyze_parser = subparsers.add_parser("analyze", help="Run dataset analysis (Task Sheet 1)")
    analyze_parser.add_argument(
        "--dataset",
        choices=DATASET_CONFIG.keys(),
        required=True,
        help="Dataset version to analyze"
    )

    # ===== SIMILARITY SUBPARSER =====
    similarity_parser = subparsers.add_parser("similarity", help="Run similarity analysis (spearman, t-SNE, PCA)")
    similarity_parser.add_argument(
        "--dataset",
        choices=DATASET_CONFIG.keys(),
        required=True,
        help="Dataset version for similarity analysis"
    )

    # ===== NGRAM SUBPARSER =====
    ngram_parser = subparsers.add_parser("ngram", help="Run n-gram based detection")
    ngram_parser.add_argument(
        "--dataset",
        choices=DATASET_CONFIG.keys(),
        required=True,
        help="Dataset version for n-gram analysis"
    )
    ngram_parser.add_argument(
        "--ngram-order",
        type=int,
        default=2,
        help="N-gram order (default: 2)"
    )

    # ===== ML SUBPARSER =====
    ml_parser = subparsers.add_parser("ml", help="Run machine learning detection (Task Sheet 2)")
    ml_parser.add_argument(
        "-m", "--model",
        choices=['ocsvm', 'lof', 'ee', 'knn', 'svm', 'rf'],
        required=True,
        help="Machine learning model"
    )
    ml_parser.add_argument(
        "-sc", "--scenario",
        choices=['1', '2', '3'],
        required=True,
        help="Cross-validation scenario"
    )
    ml_parser.add_argument(
        "-k", "--kfold",
        type=int,
        default=5,
        help="Number of folds (default: 5)"
    )
    ml_parser.add_argument(
        "-e", "--export",
        choices=['1', '2', '3'],
        default='1',
        help="Export results (default: 1)"
    )

    # ===== DL SUBPARSER =====
    dl_parser = subparsers.add_parser("dl", help="Run deep learning detection (Task Sheet 2)")
    dl_parser.add_argument(
        "-sc",
        "--scenario",
        required=True,
        choices=['1', '2', '3'],
        help="Scenario number 1 | 2 | 3",
    )
    dl_parser.add_argument(
        "-M",
        "--window-size",
        required=True,
        help="Window size M (rows per CNN input)",
    )
    dl_parser.add_argument(
        "-e",
        "--epochs",
        default=5,
        help="Maximum training epochs",
    )
    dl_parser.add_argument(
        "--stride",
        default=10,
        help="Stride for training windows (default=10, use 1 for max overlap)",
    )

    args = parser.parse_args()

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

    else:
        print("Unknown mode")


# ==============================================================
# TASK SHEET 1 FUNCTIONS
# ==============================================================
def run_task1(script):
    print(f"Running dataset analysis: {script}")
    os.system(f"python {script}")


def run_similarity(dataset_name):
    modify_spearman(dataset_name)
    modify_tsne(dataset_name)
    modify_pca(dataset_name)

    os.system("python spearman_distance.py")
    os.system("python histogram.py")
    os.system("python task2c.py")
    os.system("python task2d.py")


def modify_spearman(version):
    with open("spearman_distance.py", "r") as f:
        txt = f.read()
    txt = re.sub(r'version\s*=\s*".*"', f'version = \"{version}\"', txt)
    with open("spearman_distance.py", "w") as f:
        f.write(txt)


def modify_tsne(version):
    with open("task2c.py", "r") as f:
        txt = f.read()
    txt = re.sub(r'versions\s*=\s*\[.*?\]', f'versions = [\"{version}\"]', txt)
    with open("task2c.py", "w") as f:
        f.write(txt)


def modify_pca(version):
    with open("task2d.py", "r") as f:
        txt = f.read()
    txt = re.sub(r'versions\s*=\s*\[.*?\]', f'versions = [\"{version}\"]', txt)
    with open("task2d.py", "w") as f:
        f.write(txt)


def run_ngram(version, n):
    with open("task3_21.03.py", "r") as f:
        txt = f.read()

    txt = re.sub(r"N_GRAM_ORDER\s*=\s*\d+", f"N_GRAM_ORDER = {n}", txt)
    folder = DATASET_CONFIG[version]["folder"]
    txt = re.sub(r'DATA_DIR\s*=\s*r".*"', f'DATA_DIR = r"{folder}"', txt)

    with open("task3_21.03.py", "w") as f:
        f.write(txt)

    os.system("python task3_21.03.py")


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

    import task1
    task1.run_from_toolbox(
        model=args.model,
        scenario=args.scenario,
        k=args.k,
        export=args.e
    )

def run_dl(args):
    print("\nRunning Task Sheet 2 – Deep Learning (Task 2)")

    if args.scenario is None:
        print("Error: --scenario (scenario) required for DL mode.")
        return

    import task2
    task2.run_from_toolbox(
        scenario=args.scenario,
        M=args.M,
        epochs=args.epochs,
        stride=args.stride
    )

# ==============================================================
# HELP
# ==============================================================
def print_help():
    print("""
ICS Security Toolbox 
==========================================

Usage: python toolbox.py <mode> [options]

Available Modes:

1. ANALYZE (Ks-statistic , ccdfs ):
   python toolbox.py analyze --dataset <name>

2. SIMILARITY (Spearman, t-SNE, PCA):
   python toolbox.py similarity --dataset <name>

3. NGRAM (N-gram anomly Detection):
   python toolbox.py ngram --dataset <name> --ngram-order <number>
          
4. ML (Traditional Classifiers - Task Sheet 2):
   python toolbox.py ml -m <model> -sc <scenario> -k <kfold>
   python toolbox.py ml -e <scenario_number> (exports scenario folds)

5. DL (Deep Learning CNN - Task Sheet 2):
    python toolbox.py dl -sc <scenario> -M <window_size> -e <epochs> --stride <stride>

Models: one class svm (ocsvm), Local outlier factor (lof), Elliptic Envelope (ee), k-nearest neighbors (knn), Support Vector Machine (svm), Random Forest (rf)
Scenarios: 1 only for one class models; 2 or 3 for supervised models


"""
)

if __name__ == "__main__":
    main()
