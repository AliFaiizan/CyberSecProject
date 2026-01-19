#!/usr/bin/env python3
"""
Task 3a: Venn/UpSet Plots for Classification Errors
Compares Real (Sheet 3) vs Synthetic (Sheet 4) training data
"""

import argparse
import os
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from upsetplot import UpSet, from_contents

# =================================================
# HELPER FUNCTIONS
# =================================================

def load_errors(csv_path):
    """Load prediction CSV and return set of error indices"""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    errors = set(df.index[df["predicted_label"] != df["Attack"]])
    return errors


def find_csv(base_dir, scenario, model, fold, suffix=""):
    """Find prediction CSV file - handles multiple naming conventions"""
    path = os.path.join(base_dir, f"Scenario{scenario}", model, f"Predictions_Fold{fold}{suffix}.csv")
    if os.path.exists(path):
        return path
    
    path_alt = os.path.join(base_dir, f"Scenario{scenario}", model, f"Predictions{suffix}_Fold{fold}.csv")
    if os.path.exists(path_alt):
        return path_alt
    
    if model == "NGRAM":
        path_ngram = os.path.join(base_dir, f"Scenario{scenario}", "NGRAM", f"Predictions{suffix}_Fold{fold}.csv")
        if os.path.exists(path_ngram):
            return path_ngram
    
    return None


# =================================================
# SCENARIO 1: VENN3 (NGRAM + 2 ML)
# =================================================

def plot_scenario1_venn3(base_dir, scenario, fold, ngram_n, data_type="synthetic"):
    """Scenario 1: Create 3 Venn3 diagrams with NGRAM + 2 ML models each"""
    print(f"\n{'='*60}")
    print(f"Scenario 1 Venn3 Plots ({data_type.upper()} data)")
    print(f"{'='*60}")
    
    ml_models = ["OCSVM", "LOF", "EllipticEnvelope"]
    ngram_label = f"NGRAM_N{ngram_n}"
    
    errors = {}
    for m in ml_models:
        path = find_csv(base_dir, scenario, m, fold)
        if path:
            err = load_errors(path)
            if err is not None:
                errors[m] = err
                print(f"  [OK] {m}: {len(errors[m])} errors")
        else:
            print(f"  [MISSING] {m}: Not found")
    
    ngram_path = find_csv(base_dir, scenario, "NGRAM", fold, suffix=f"_N{ngram_n}")
    if ngram_path:
        err = load_errors(ngram_path)
        if err is not None:
            errors[ngram_label] = err
            print(f"  [OK] {ngram_label}: {len(errors[ngram_label])} errors")
    else:
        print(f"  [MISSING] {ngram_label}: Not found")
    
    if len(errors) < 3:
        print(f"  [WARNING] Not enough methods found ({len(errors)}/4 needed)")
        return
    
    out_dir = os.path.join(base_dir, f"Scenario{scenario}", "Venn")
    os.makedirs(out_dir, exist_ok=True)
    
    combos = list(itertools.combinations(ml_models, 2))
    
    for ml_a, ml_b in combos:
        if ml_a not in errors or ml_b not in errors or ngram_label not in errors:
            print(f"  [SKIP] {ml_a} vs {ml_b} vs {ngram_label}")
            continue
            
        plt.figure(figsize=(8, 8))
        venn3([errors[ml_a], errors[ml_b], errors[ngram_label]], set_labels=(ml_a, ml_b, ngram_label))
        plt.title(f"Scenario {scenario} Error Overlap ({data_type} data)\nFold {fold} | N={ngram_n}", fontsize=14)
        
        filename = f"Venn3_{ml_a}_{ml_b}_{ngram_label}_Fold{fold}_{data_type}.png"
        filepath = os.path.join(out_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [SAVED] {filename}")


# =================================================
# SCENARIO 2 & 3: UPSET (4 METHODS)
# =================================================

def plot_scenario_upset(base_dir, scenario, fold, data_type="synthetic"):
    """Scenario 2 & 3: Create UpSet plot for 4 methods"""
    print(f"\n{'='*60}")
    print(f"Scenario {scenario} UpSet Plot ({data_type.upper()} data)")
    print(f"{'='*60}")
    
    models = ["SVM", "kNN", "RandomForest", "CNN"]
    
    contents = {}
    for m in models:
        path = find_csv(base_dir, scenario, m, fold)
        if path:
            err = load_errors(path)
            if err is not None:
                contents[m] = err
                print(f"  [OK] {m}: {len(contents[m])} errors")
        else:
            print(f"  [MISSING] {m}: Not found")
    
    total_errors = sum(len(errs) for errs in contents.values())
    
    if total_errors == 0:
        print(f"  [INFO] All classifiers achieved 100% accuracy - no errors to plot!")
        return
    
    if len(contents) < 2:
        print(f"  [WARNING] Not enough methods found")
        return
    
    out_dir = os.path.join(base_dir, f"Scenario{scenario}", "UpSet")
    os.makedirs(out_dir, exist_ok=True)
    
    upset_data = from_contents(contents)
    
    fig = plt.figure(figsize=(12, 8))
    upset = UpSet(upset_data, subset_size="count", show_counts=True, sort_by="cardinality")
    upset.plot(fig=fig)
    
    plt.suptitle(f"Scenario {scenario} Error Overlap ({data_type} data) - Fold {fold}", fontsize=16, y=0.98)
    
    filename = f"UpSet_Scenario{scenario}_Fold{fold}_{data_type}.png"
    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {filename}")


# =================================================
# COMPARISON FUNCTION
# =================================================

def generate_comparison_summary(sheet3_dir, sheet4_dir, scenario, fold, ngram_n=None):
    """Generate text summary comparing real vs synthetic errors"""
    print("\n" + "="*60)
    print(f"COMPARISON: Real vs Synthetic (Scenario {scenario}, Fold {fold})")
    print("="*60)
    
    if scenario == 1:
        models = ["OCSVM", "LOF", "EllipticEnvelope", "NGRAM"]
    else:
        models = ["SVM", "kNN", "RandomForest", "CNN"]
    
    for model in models:
        if model == "NGRAM":
            real_path = find_csv(sheet3_dir, scenario, "NGRAM", fold, suffix=f"_N{ngram_n}")
            synth_path = find_csv(sheet4_dir, scenario, "NGRAM", fold, suffix=f"_N{ngram_n}")
        else:
            real_path = find_csv(sheet3_dir, scenario, model, fold)
            synth_path = find_csv(sheet4_dir, scenario, model, fold)
        
        if real_path and synth_path:
            real_errors = load_errors(real_path)
            synth_errors = load_errors(synth_path)
            
            if real_errors is not None and synth_errors is not None:
                only_real = real_errors - synth_errors
                only_synth = synth_errors - real_errors
                shared = real_errors & synth_errors
                
                display_name = f"{model}_N{ngram_n}" if model == "NGRAM" else model
                print(f"\n{display_name}:")
                print(f"  Real-only errors:      {len(only_real):>6}")
                print(f"  Synthetic-only errors: {len(only_synth):>6}")
                print(f"  Shared errors:         {len(shared):>6}")
                print(f"  Total real:            {len(real_errors):>6}")
                print(f"  Total synthetic:       {len(synth_errors):>6}")
                
                if len(real_errors) > 0:
                    overlap_pct = len(shared) / len(real_errors) * 100
                    print(f"  Overlap percentage:    {overlap_pct:>5.1f}%")


# =================================================
# MAIN
# =================================================

def main():
    parser = argparse.ArgumentParser(description="Task 3a: Generate Venn/UpSet plots for error analysis")
    parser.add_argument("--sheet3_dir", default="../tasksheet3/exports", help="Path to Sheet 3 exports (real data)")
    parser.add_argument("--sheet4_dir", default="./exports_sheet4", help="Path to Sheet 4 exports (synthetic data)")
    parser.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3], help="Scenario number")
    parser.add_argument("--fold", type=int, required=True, help="Fold number")
    parser.add_argument("--ngram_n", type=int, default=2, choices=[2, 5, 8], help="N-gram size (Scenario 1 only)")
    parser.add_argument("--compare", action="store_true", help="Generate comparison summary")
    args = parser.parse_args()
    
    print("="*60)
    print(f"TASK 3a: Error Overlap Analysis")
    print(f"Scenario {args.scenario}, Fold {args.fold}")
    print("="*60)
    
    if os.path.exists(args.sheet3_dir):
        print(f"\nProcessing Sheet 3 (REAL data) from: {args.sheet3_dir}")
        if args.scenario == 1:
            plot_scenario1_venn3(args.sheet3_dir, args.scenario, args.fold, args.ngram_n, data_type="real")
        else:
            plot_scenario_upset(args.sheet3_dir, args.scenario, args.fold, data_type="real")
    else:
        print(f"\n[WARNING] Sheet 3 directory not found: {args.sheet3_dir}")
    
    if os.path.exists(args.sheet4_dir):
        print(f"\nProcessing Sheet 4 (SYNTHETIC data) from: {args.sheet4_dir}")
        if args.scenario == 1:
            plot_scenario1_venn3(args.sheet4_dir, args.scenario, args.fold, args.ngram_n, data_type="synthetic")
        else:
            plot_scenario_upset(args.sheet4_dir, args.scenario, args.fold, data_type="synthetic")
    else:
        print(f"\n[WARNING] Sheet 4 directory not found: {args.sheet4_dir}")
    
    if args.compare:
        ngram_n = args.ngram_n if args.scenario == 1 else None
        generate_comparison_summary(args.sheet3_dir, args.sheet4_dir, args.scenario, args.fold, ngram_n)
    
    print("\n" + "="*60)
    print("Task 3a completed successfully")
    print("="*60)


if __name__ == "__main__":
    main()
