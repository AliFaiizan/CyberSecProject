import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ---------------------------------------------------------
# 1. COMPARATIVE BAR PLOTS (Main Requirement)
# ---------------------------------------------------------
def plot_comparative_metrics(scenario, base_dir="exports"):
    """
    Create bar plots comparing precision/recall across all models
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    if scenario == 1:
        models = ["OCSVM", "LOF", "EllipticEnvelope", "Ensemble"]
    else:
        models = ["SVM", "kNN", "RandomForest", "CNN", "Ensemble"]
    
    precision_data = []
    recall_data = []
    
    for model in models:
        if model == "Ensemble":
            # Calculate average for all ensemble methods
            ensemble_precisions = []
            ensemble_recalls = []
            ensemble_dir = os.path.join(base_dir, f"Scenario{scenario}", "Ensemble")
            methods = ["random", "majority", "all"]
            
            for method in methods:
                fold_metrics = []
                for fold in range(1, 6):
                    file = f"{ensemble_dir}/Ensemble_{method}_Fold{fold}.csv"
                    if os.path.exists(file):
                        df = pd.read_csv(file)
                        preds = df["predicted_label"].values
                        true = df["Attack"].values
                        
                        tp = ((preds == 1) & (true == 1)).sum()
                        fp = ((preds == 1) & (true == 0)).sum()
                        fn = ((preds == 0) & (true == 1)).sum()
                        
                        precision = tp / (tp + fp + 1e-9)
                        recall = tp / (tp + fn + 1e-9)
                        fold_metrics.append((precision, recall))
                
                if fold_metrics:
                    precisions, recalls = zip(*fold_metrics)
                    ensemble_precisions.append(np.mean(precisions))
                    ensemble_recalls.append(np.mean(recalls))
            
            if ensemble_precisions:
                precision_data.append(np.mean(ensemble_precisions))
                recall_data.append(np.mean(ensemble_recalls))
        else:
            metrics_path = os.path.join(base_dir, f"Scenario{scenario}", model, "metrics_summary.csv")
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                precision_data.append(df["precision"].mean())
                recall_data.append(df["recall"].mean())
            else:
                precision_data.append(0)
                recall_data.append(0)
    
    # Plot Precision
    x = np.arange(len(models))
    axes[0].bar(x, precision_data, color='skyblue', edgecolor='black')
    axes[0].set_title(f'Scenario {scenario}: Average Precision by Model', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Precision')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylim([0, 1])
    
    # Add value labels on bars
    for i, v in enumerate(precision_data):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    # Plot Recall
    axes[1].bar(x, recall_data, color='lightcoral', edgecolor='black')
    axes[1].set_title(f'Scenario {scenario}: Average Recall by Model', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Recall')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].set_ylim([0, 1])
    
    # Add value labels on bars
    for i, v in enumerate(recall_data):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'Scenario{scenario}_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return precision_data, recall_data

# ---------------------------------------------------------
# 2. RUNTIME & MEMORY ANALYSIS (Task 2(e) Requirement)
# ---------------------------------------------------------
def plot_runtime_memory(scenario, base_dir="exports"):
    """
    Plot runtime vs memory usage for all models
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    if scenario == 1:
        models = ["OCSVM", "LOF", "EllipticEnvelope"]
    else:
        models = ["SVM", "kNN", "RandomForest", "CNN"]
    
    runtime_data = []
    memory_data = []
    model_names = []
    
    for model in models:
        metrics_path = os.path.join(base_dir, f"Scenario{scenario}", model, "metrics_summary.csv")
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            runtime_data.append(df["runtime_sec"].mean())
            memory_data.append(df["memory_bytes"].mean() / 1e6)  # Convert to MB
            model_names.append(model)
    
    # Runtime Bar Plot
    x = np.arange(len(model_names))
    bars1 = axes[0].bar(x, runtime_data, color='lightgreen', edgecolor='black')
    axes[0].set_title(f'Scenario {scenario}: Average Runtime', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Runtime (seconds)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels
    for i, v in enumerate(runtime_data):
        axes[0].text(i, v + max(runtime_data)*0.01, f'{v:.1f}s', ha='center', fontsize=9)
    
    # Memory Bar Plot
    bars2 = axes[1].bar(x, memory_data, color='gold', edgecolor='black')
    axes[1].set_title(f'Scenario {scenario}: Average Memory Usage', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Memory (MB)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels
    for i, v in enumerate(memory_data):
        axes[1].text(i, v + max(memory_data)*0.01, f'{v:.1f}MB', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'Scenario{scenario}_runtime_memory.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also create scatter plot of runtime vs memory
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(runtime_data, memory_data, s=200, alpha=0.7, c=range(len(model_names)), cmap='viridis')
    
    # Annotate points
    for i, txt in enumerate(model_names):
        plt.annotate(txt, (runtime_data[i], memory_data[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.title(f'Scenario {scenario}: Runtime vs Memory Usage', fontsize=14, fontweight='bold')
    plt.xlabel('Runtime (seconds)')
    plt.ylabel('Memory (MB)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'Scenario{scenario}_runtime_vs_memory.png', dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------
# 3. FOLD-WISE PERFORMANCE (Line Plots)
# ---------------------------------------------------------
def plot_fold_performance(scenario, base_dir="exports"):
    """
    Plot precision and recall across folds for each model
    """
    if scenario == 1:
        models = ["OCSVM", "LOF", "EllipticEnvelope"]
    else:
        models = ["SVM", "kNN", "RandomForest", "CNN"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, model in enumerate(models):
        if idx >= len(axes):
            break
            
        metrics_path = os.path.join(base_dir, f"Scenario{scenario}", model, "metrics_summary.csv")
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            
            folds = df["fold"].values
            precision = df["precision"].values
            recall = df["recall"].values
            
            axes[idx].plot(folds, precision, marker='o', linewidth=2, markersize=8, label='Precision')
            axes[idx].plot(folds, recall, marker='s', linewidth=2, markersize=8, label='Recall')
            
            axes[idx].set_title(f'{model}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Fold')
            axes[idx].set_ylabel('Score')
            axes[idx].set_ylim([0, 1.1])
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend()
            axes[idx].set_xticks(folds)
    
    # Hide empty subplots
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Scenario {scenario}: Fold-wise Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'Scenario{scenario}_fold_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------
# 4. ENSEMBLE METHODS COMPARISON
# ---------------------------------------------------------
def plot_ensemble_comparison(scenario, base_dir="exports"):
    """
    Compare different ensemble methods
    """
    ensemble_dir = os.path.join(base_dir, f"Scenario{scenario}", "Ensemble")
    methods = ["random", "majority", "all"]
    
    if not os.path.exists(ensemble_dir):
        print(f"No ensemble results found for Scenario {scenario}")
        return
    
    data = []
    for method in methods:
        precisions = []
        recalls = []
        
        for fold in range(1, 6):
            file = f"{ensemble_dir}/Ensemble_{method}_Fold{fold}.csv"
            if os.path.exists(file):
                df = pd.read_csv(file)
                preds = df["predicted_label"].values
                true = df["Attack"].values
                
                tp = ((preds == 1) & (true == 1)).sum()
                fp = ((preds == 1) & (true == 0)).sum()
                fn = ((preds == 0) & (true == 1)).sum()
                
                precision = tp / (tp + fp + 1e-9)
                recall = tp / (tp + fn + 1e-9)
                
                precisions.append(precision)
                recalls.append(recall)
        
        if precisions:
            data.append({
                'Method': method,
                'Avg Precision': np.mean(precisions),
                'Avg Recall': np.mean(recalls),
                'Std Precision': np.std(precisions),
                'Std Recall': np.std(recalls)
            })
    
    if not data:
        return
    
    df_ensemble = pd.DataFrame(data)
    
    # Bar plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(methods))
    width = 0.35
    
    # Precision with error bars
    axes[0].bar(x - width/2, df_ensemble['Avg Precision'], width, 
                yerr=df_ensemble['Std Precision'], capsize=5, label='Precision', color='skyblue')
    axes[0].set_title('Ensemble: Precision by Method', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Method')
    axes[0].set_ylabel('Precision')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylim([0, 1])
    
    # Recall with error bars
    axes[1].bar(x + width/2, df_ensemble['Avg Recall'], width, 
                yerr=df_ensemble['Std Recall'], capsize=5, label='Recall', color='lightcoral')
    axes[1].set_title('Ensemble: Recall by Method', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Method')
    axes[1].set_ylabel('Recall')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylim([0, 1])
    
    plt.suptitle(f'Scenario {scenario}: Ensemble Methods Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'Scenario{scenario}_ensemble_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------
# 5. SUMMARY TABLE GENERATION
# ---------------------------------------------------------
def generate_summary_table(scenarios=[1, 2, 3], base_dir="exports"):
    """
    Generate a comprehensive summary table of all experiments
    """
    summary_data = []
    
    for scenario in scenarios:
        if scenario == 1:
            models = ["OCSVM", "LOF", "EllipticEnvelope"]
        else:
            models = ["SVM", "kNN", "RandomForest", "CNN"]
        
        for model in models:
            metrics_path = os.path.join(base_dir, f"Scenario{scenario}", model, "metrics_summary.csv")
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                summary_data.append({
                    'Scenario': scenario,
                    'Model': model,
                    'Avg Precision': df['precision'].mean(),
                    'Avg Recall': df['recall'].mean(),
                    'Avg Runtime (s)': df['runtime_sec'].mean(),
                    'Avg Memory (MB)': df['memory_bytes'].mean() / 1e6,
                    'Folds': len(df)
                })
    
    # Add ensemble results
    for scenario in scenarios:
        ensemble_dir = os.path.join(base_dir, f"Scenario{scenario}", "Ensemble")
        methods = ["random", "majority", "all"]
        
        for method in methods:
            precisions = []
            recalls = []
            
            for fold in range(1, 6):
                file = f"{ensemble_dir}/Ensemble_{method}_Fold{fold}.csv"
                if os.path.exists(file):
                    df = pd.read_csv(file)
                    preds = df["predicted_label"].values
                    true = df["Attack"].values
                    
                    tp = ((preds == 1) & (true == 1)).sum()
                    fp = ((preds == 1) & (true == 0)).sum()
                    fn = ((preds == 0) & (true == 1)).sum()
                    
                    precision = tp / (tp + fp + 1e-9)
                    recall = tp / (tp + fn + 1e-9)
                    
                    precisions.append(precision)
                    recalls.append(recall)
            
            if precisions:
                summary_data.append({
                    'Scenario': scenario,
                    'Model': f'Ensemble_{method}',
                    'Avg Precision': np.mean(precisions),
                    'Avg Recall': np.mean(recalls),
                    'Avg Runtime (s)': np.nan,  # Not measured for ensemble
                    'Avg Memory (MB)': np.nan,
                    'Folds': len(precisions)
                })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save to CSV
    df_summary.to_csv('experiment_summary.csv', index=False)
    
    # Create a formatted table for display
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY TABLE")
    print("="*80)
    print(df_summary.to_string())
    print("="*80)
    
    # Save as LaTeX table (optional for reports)
    with open('summary_table.tex', 'w') as f:
        f.write(df_summary.to_latex(index=False))
    
    return df_summary

# ---------------------------------------------------------
# 6. MAIN FUNCTION TO GENERATE ALL PLOTS
# ---------------------------------------------------------
def generate_all_plots(scenarios=[1, 2, 3], base_dir="exports"):
    """
    Generate all required plots for the task
    """
    print("Generating comprehensive visualizations for Task 2...")
    
    # Create output directory for plots
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Change working directory to save plots
    original_dir = os.getcwd()
    os.chdir(plot_dir)
    
    try:
        for scenario in scenarios:
            print(f"\n{'='*60}")
            print(f"Generating plots for Scenario {scenario}")
            print(f"{'='*60}")
            
            # 1. Comparative metrics (bar plots)
            print("1. Creating comparative bar plots...")
            plot_comparative_metrics(scenario, os.path.join('..', base_dir))
            
            # 2. Runtime and memory analysis
            print("2. Creating runtime/memory plots...")
            plot_runtime_memory(scenario, os.path.join('..', base_dir))
            
            # 3. Fold-wise performance
            print("3. Creating fold-wise performance plots...")
            plot_fold_performance(scenario, os.path.join('..', base_dir))
            
            # 4. Ensemble comparison
            print("4. Creating ensemble comparison plots...")
            plot_ensemble_comparison(scenario, os.path.join('..', base_dir))
        
        # 5. Generate summary table
        print("\n5. Generating comprehensive summary table...")
        summary_df = generate_summary_table(scenarios, os.path.join('..', base_dir))
        
        # 6. Create one master comparison plot across scenarios
        print("\n6. Creating master comparison across scenarios...")
        create_master_comparison(summary_df)
        
    finally:
        os.chdir(original_dir)
    
    print(f"\n{'='*60}")
    print("All plots generated successfully!")
    print(f"Saved to directory: {plot_dir}")
    print(f"Summary table saved as: experiment_summary.csv")
    print(f"{'='*60}")

# ---------------------------------------------------------
# 7. MASTER COMPARISON PLOT
# ---------------------------------------------------------
def create_master_comparison(summary_df):
    """
    Create a master plot comparing all scenarios and models
    """
    # Filter out ensemble for cleaner comparison
    main_models = summary_df[~summary_df['Model'].str.contains('Ensemble')]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Precision across scenarios
    for scenario in sorted(main_models['Scenario'].unique()):
        scenario_data = main_models[main_models['Scenario'] == scenario]
        axes[0, 0].bar(scenario_data['Model'], scenario_data['Avg Precision'], 
                      alpha=0.6, label=f'Scenario {scenario}')
    axes[0, 0].set_title('Precision Comparison Across Scenarios', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Recall across scenarios
    for scenario in sorted(main_models['Scenario'].unique()):
        scenario_data = main_models[main_models['Scenario'] == scenario]
        axes[0, 1].bar(scenario_data['Model'], scenario_data['Avg Recall'], 
                      alpha=0.6, label=f'Scenario {scenario}')
    axes[0, 1].set_title('Recall Comparison Across Scenarios', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Runtime vs Memory scatter (all scenarios)
    colors = plt.cm.tab10(range(len(main_models['Scenario'].unique())))
    for idx, scenario in enumerate(sorted(main_models['Scenario'].unique())):
        scenario_data = main_models[main_models['Scenario'] == scenario]
        axes[1, 0].scatter(scenario_data['Avg Runtime (s)'], scenario_data['Avg Memory (MB)'],
                          s=100, c=[colors[idx]], alpha=0.7, label=f'Scenario {scenario}')
        # Annotate points
        for _, row in scenario_data.iterrows():
            axes[1, 0].annotate(row['Model'], 
                              (row['Avg Runtime (s)'], row['Avg Memory (MB)']),
                              fontsize=8, alpha=0.8)
    axes[1, 0].set_title('Runtime vs Memory Usage', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Runtime (seconds)')
    axes[1, 0].set_ylabel('Memory (MB)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Best performing models per scenario
    best_models = []
    for scenario in sorted(main_models['Scenario'].unique()):
        scenario_data = main_models[main_models['Scenario'] == scenario]
        best_idx = scenario_data['Avg Precision'].idxmax()
        best_models.append(scenario_data.loc[best_idx])
    
    if best_models:
        best_df = pd.DataFrame(best_models)
        x = np.arange(len(best_df))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, best_df['Avg Precision'], width, label='Precision', color='skyblue')
        axes[1, 1].bar(x + width/2, best_df['Avg Recall'], width, label='Recall', color='lightcoral')
        axes[1, 1].set_title('Best Performing Model per Scenario', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Scenario')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f"Scenario {s}" for s in best_df['Scenario']])
        axes[1, 1].legend()
        axes[1, 1].set_ylim([0, 1])
    
    plt.suptitle('Task 2: Comprehensive Analysis of All Experiments', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('master_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comprehensive plots for Task 2')
    parser.add_argument('--scenarios', type=str, default='1,2,3',
                       help='Comma-separated list of scenarios to plot (default: 1,2,3)')
    parser.add_argument('--base-dir', type=str, default='exports',
                       help='Base directory containing experiment results')
    
    args = parser.parse_args()
    
    # Parse scenarios
    scenarios = [int(s.strip()) for s in args.scenarios.split(',')]
    
    # Generate all plots
    generate_all_plots(scenarios, args.base_dir)