#!/usr/bin/env python3
"""
ICS Security Toolbox – Practical Sheet 1
"""

import argparse
import sys
import os
import re
import glob


DATASET_CONFIG = {
    "hai-21.03": {
        "folder": "../datasets/hai-21.03",
        "train_pattern": "train*.csv",
        "test_pattern": "test*.csv",
        "script": "task1.py",
        "has_labels": False
    },
    "hai-22.04": {
        "folder": "../datasets/hai-22.04",
        "train_pattern": "train*.csv",
        "test_pattern": "test*.csv",
        "script": "task1-22.py",
        "has_labels": False
    },
    "haiend-23.05": {
        "folder": "../datasets/haiend-23.05",
        "train_pattern": "end-train*.csv",
        "test_pattern": "end-test*.csv",
        "label_pattern": "label-test*.csv",
        "script": "task1-23.py",
        "has_labels": True
    }
}


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "-s", "--start",
        required=True,
        choices=["analyze", "similarity", "ngram"],
        help="Which task to run"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_CONFIG.keys(),
        help="Dataset version"
    )
    parser.add_argument(
        "--ngram-order",
        type=int,
        default=2,
        help="N-gram order"
    )
    parser.add_argument("-h", "--help", action="store_true")

    args = parser.parse_args()

    if args.help:
        print_help()
        sys.exit(0)

    cfg = DATASET_CONFIG[args.dataset]

    # Check dataset folder exists
    if not os.path.exists(cfg["folder"]):
        print(f"❌ Dataset folder not found: {cfg['folder']}")
        sys.exit(1)

    # Check files exist
    train_files = glob.glob(os.path.join(cfg["folder"], cfg["train_pattern"]))
    test_files  = glob.glob(os.path.join(cfg["folder"], cfg["test_pattern"]))

    if not train_files:
        print("❌ No train files found.")
        sys.exit(1)
    if not test_files:
        print("❌ No test files found.")
        sys.exit(1)

    print("ICS Security Toolbox")
    print("====================")
    print(f"Mode:    {args.start}")
    print(f"Dataset: {args.dataset}")
    print(f"Train files: {len(train_files)}")
    print(f"Test files:  {len(test_files)}\n")

    # Run the appropriate mode
    if args.start == "analyze":
        run_task1(cfg["script"])
    elif args.start == "similarity":
        run_similarity(args.dataset)
    elif args.start == "ngram":
        run_ngram(args.dataset, args.ngram_order)


# ---------------------------------------------------------
# TASK 1 – ANALYSIS
# ---------------------------------------------------------
def run_task1(script):
    print(f"Running {script} ...")
    os.system(f"python {script}")


# ---------------------------------------------------------
# TASK 2 – SIMILARITY
# ---------------------------------------------------------
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
    txt = re.sub(r'version\s*=\s*".*"', f'version = "{version}"', txt)
    with open("spearman_distance.py", "w") as f:
        f.write(txt)


def modify_tsne(version):
    with open("task2c.py", "r") as f:
        txt = f.read()
    txt = re.sub(r'versions\s*=\s*\[.*?\]', f'versions = ["{version}"]', txt)
    with open("task2c.py", "w") as f:
        f.write(txt)


def modify_pca(version):
    with open("task2d.py", "r") as f:
        txt = f.read()
    txt = re.sub(r'versions\s*=\s*\[.*?\]', f'versions = ["{version}"]', txt)
    with open("task2d.py", "w") as f:
        f.write(txt)


# ---------------------------------------------------------
# TASK 3 – N-GRAM
# ---------------------------------------------------------
def run_ngram(version, n):
    with open("task3_21.03.py", "r") as f:
        txt = f.read()

    txt = re.sub(r"N_GRAM_ORDER\s*=\s*\d+", f"N_GRAM_ORDER = {n}", txt)

    # Update dataset path
    folder = DATASET_CONFIG[version]["folder"]
    txt = re.sub(r'DATA_DIR\s*=\s*r".*"', f'DATA_DIR = r"{folder}"', txt)

    with open("task3_21.03.py", "w") as f:
        f.write(txt)

    os.system("python task3_21.03.py")


# ---------------------------------------------------------
# HELP
# ---------------------------------------------------------
def print_help():
    print("""
ICS Security Toolbox – Practical Sheet 1
========================================

Usage:
    python toolbox.py -s analyze --dataset hai-21.03
    python toolbox.py -s similarity --dataset hai-22.04
    python toolbox.py -s ngram --dataset haiend-23.05 --ngram-order 3
""")


if __name__ == "__main__":
    main()
