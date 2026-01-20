#!/usr/bin/env python3
"""
Visualize Task 1 CNN results
Creates plots and summary tables
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def load_results(results_dir="task1_results"):
    """Load all JSON result files."""
    results = []
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return results
    
    for file in Path(results_dir).glob("*.json"):
        with open(file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    print(f"Loaded {len(results)} result files")
    return results


def create_summary_table(results):
    """Create summary DataFrame."""
    rows = []
    
    for r in results:
        rows.append({
            'Scenario': r['scenario'],
            'k': r['k_folds'],
            'M': r['M'],
            'Data': r['data_type'],
            'Precision': f"{r['mean_precision']:.4f} ± {r['std_precision']:.4f}",
            'Recall': f"{r['mean_recall']:.4f} ± {r['std_recall']:.4f}",
            'F1': f"{r['mean_f1']:.4f} ± {r['std_f1']:.4f}",
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['Scenario', 'k', 'M', 'Data'])
    return df


def plot_results(results, output_dir="task1_results"):
    """Create visualization plots."""
    
    if not results:
        print("No results to plot!")
        return
    
    # Plot: F1-Score comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Task 1: CNN F1-Score Results', fontsize=16, fontweight='bold')
    
    configs = [(5, 50), (5, 86), (10, 50), (10, 86)]
    
    for idx, (k, M) in enumerate(configs):
        ax = axes[idx // 2, idx % 2]
        
        # Filter results
        filtered = [r for r in results if r['k_folds'] == k and r['M'] == M]
        
        if not filtered:
            continue
        
        labels = []
        f1_scores = []
        colors = []
        
        for r in sorted(filtered, key=lambda x: (x['scenario'], x['data_type'])):
            label = f"Sc{r['scenario']}-{r['data_type'][:4]}"
            labels.append(label)
            f1_scores.append(r['mean_f1'])
            colors.append('steelblue' if r['data_type'] == 'real' else 'coral')
        
        bars = ax.bar(range(len(labels)), f1_scores, color=colors, alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_title(f'k={k}, M={M}', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.8, label='Real Data'),
        Patch(facecolor='coral', alpha=0.8, label='Synthetic Data')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/task1_f1_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/task1_f1_comparison.png")
    plt.close()
    
    # Plot: Precision vs Recall
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for r in results:
        marker = 'o' if r['data_type'] == 'real' else 's'
        color = 'blue' if r['scenario'] == 2 else 'red'
        label = f"Sc{r['scenario']}-{r['data_type'][:4]}-k{r['k_folds']}-M{r['M']}"
        
        ax.scatter(r['mean_recall'], r['mean_precision'], 
                  marker=marker, s=100, alpha=0.7, color=color, label=label)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Task 1: Precision vs Recall', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/task1_precision_recall.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/task1_precision_recall.png")
    plt.close()


def print_summary(results):
    """Print comprehensive summary."""
    print("\n" + "="*80)
    print("TASK 1: RESULTS SUMMARY")
    print("="*80)
    
    df = create_summary_table(results)
    print("\n" + df.to_string(index=False))
    
    # Statistics
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    best = max(results, key=lambda r: r['mean_f1'])
    print(f"\n Best F1-Score:")
    print(f"   Scenario {best['scenario']}, k={best['k_folds']}, M={best['M']}, {best['data_type']}")
    print(f"   F1: {best['mean_f1']:.4f} ± {best['std_f1']:.4f}")
    
    # Real vs Synthetic
    real_f1 = np.mean([r['mean_f1'] for r in results if r['data_type'] == 'real'])
    synth_f1 = np.mean([r['mean_f1'] for r in results if r['data_type'] == 'synthetic'])
    
    print(f"\n Average F1-Score:")
    print(f"   Real Data:      {real_f1:.4f}")
    print(f"   Synthetic Data: {synth_f1:.4f}")
    print(f"   Difference:     {abs(real_f1 - synth_f1):.4f}")
    
    # Scenario comparison
    sc2_f1 = np.mean([r['mean_f1'] for r in results if r['scenario'] == 2])
    sc3_f1 = np.mean([r['mean_f1'] for r in results if r['scenario'] == 3])
    
    print(f"\n Average F1 by Scenario:")
    print(f"   Scenario 2: {sc2_f1:.4f}")
    print(f"   Scenario 3: {sc3_f1:.4f}")
    
    print("\n" + "="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Task 1 Results")
    parser.add_argument("--results-dir", type=str, default="task1_results")
    args = parser.parse_args()
    
    print("Loading results...")
    results = load_results(args.results_dir)
    
    if not results:
        print(" No results found!")
        return
    
    print("\nGenerating plots...")
    plot_results(results, args.results_dir)
    
    print("\nGenerating summary...")
    print_summary(results)
    
    # Save summary table
    df = create_summary_table(results)
    summary_file = f"{args.results_dir}/task1_summary.csv"
    df.to_csv(summary_file, index=False)
    print(f"\n Summary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
