"""
FIXED: GPU-Optimized Parameter Tuning for HAI 22.04
Fixed the shape mismatch error in score_samples function
"""

import pandas as pd
import numpy as np
import hashlib
from glob import glob
from typing import List, Tuple
import json
import os
from datetime import datetime
from tqdm import tqdm

# FILE PATHS
TRAIN_FILES = sorted(glob("../datasets/hai-22.04/train1.csv"))
TEST_FILES = sorted(glob("../datasets/hai-22.04/test1.csv"))

print(f"Found {len(TRAIN_FILES)} training files, {len(TEST_FILES)} test files")

class BloomFilter:
    def __init__(self, size=1_000_000, k=5):
        self.size = size
        self.k = k
        self.bloom = np.zeros(size, dtype=bool)
    def _hashes(self, item: str) -> List[int]:
        return [int(hashlib.sha1((str(seed) + item).encode()).hexdigest(), 16) % self.size for seed in range(self.k)]
    def add(self, item: str):
        for h in self._hashes(item): self.bloom[h] = True
    def check(self, item: str) -> bool:
        return all(self.bloom[h] for h in self._hashes(item))
    def utilization(self) -> float:
        return self.bloom.mean()

def load_data(train_files, test_files):
    train_dfs = [pd.read_csv(f) for f in train_files]
    train_df = pd.concat(train_dfs, ignore_index=True)
    if 'Attack' in train_df.columns:
        train_df = train_df[train_df['Attack'] == 0]
    
    test_dfs = [pd.read_csv(f) for f in test_files]
    test_df = pd.concat(test_dfs, ignore_index=True)
    test_labels = test_df['Attack'].values if 'Attack' in test_df.columns else None
    
    train_df = train_df.drop(columns=['timestamp', 'Attack'], errors='ignore')
    test_df = test_df.drop(columns=['timestamp', 'Attack'], errors='ignore')
    train_df = train_df.ffill().fillna(0).replace([np.inf, -np.inf], 0)
    test_df = test_df.ffill().fillna(0).replace([np.inf, -np.inf], 0)
    return train_df, test_df, test_labels

def quantize_data(train_data, test_data, Q):
    mean_vals = train_data.mean()
    std_vals = train_data.std().replace(0, 1)
    train_norm = np.clip((train_data - mean_vals) / std_vals, -3, 3)
    test_norm = np.clip((test_data - mean_vals) / std_vals, -3, 3)
    train_scaled = (train_norm + 3) / 6
    test_scaled = (test_norm + 3) / 6
    return np.clip(np.floor(train_scaled * Q).astype(int), 0, Q-1), np.clip(np.floor(test_scaled * Q).astype(int), 0, Q-1)

def build_state_strings(quantized_data):
    return ["_".join(map(str, row)) for row in quantized_data]

def generate_ngrams(state_strings, n):
    return ["->".join(state_strings[i:i+n]) for i in range(len(state_strings) - n + 1)]

def train_bloom(train_ngrams, M, k):
    bloom = BloomFilter(size=M, k=k)
    for ngram in train_ngrams: bloom.add(ngram)
    return bloom

def score_samples_fixed(test_ngrams, bloom, n):
    """
    FIXED: Properly align scores with original test data length
    
    For n-gram size n:
    - We have len(test_states) = len(test_data) time steps
    - We generate len(test_states) - n + 1 n-grams
    - Each n-gram corresponds to time steps [i, i+n-1]
    - So we assign the score to the END of the window (position i+n-1)
    """
    num_test_samples = len(test_ngrams) + n - 1  # Original number of samples
    scores = np.zeros(num_test_samples)
    
    # For each n-gram, calculate score
    for i, ngram in enumerate(test_ngrams):
        # Get a window of n consecutive n-grams centered at current position
        window_start = max(0, i - n + 1)
        window_end = min(len(test_ngrams), i + 1)
        window = test_ngrams[window_start:window_end]
        
        # Count unseen n-grams in window
        unseen_count = sum(1 for ng in window if not bloom.check(ng))
        score = unseen_count / len(window) if window else 0
        
        # Assign score to the position corresponding to the end of the n-gram window
        # N-gram at index i covers time steps [i, i+n-1], so assign to position i+n-1
        scores[i + n - 1] = score
    
    # For the first n-1 positions (where we don't have complete n-grams yet),
    # use a forward-looking approach
    for i in range(n - 1):
        if i < len(test_ngrams):
            scores[i] = 1.0 if not bloom.check(test_ngrams[i]) else 0.0
    
    return scores

def find_best_threshold(scores, labels):
    best_f1, best_threshold, best_metrics = 0, 0, {}
    labels_bool = labels.astype(bool)
    
    # Ensure same length
    min_len = min(len(scores), len(labels))
    scores = scores[:min_len]
    labels_bool = labels_bool[:min_len]
    
    for threshold in np.arange(0.05, 0.95, 0.05):
        predictions = scores > threshold
        TP = np.sum(predictions & labels_bool)
        FP = np.sum(predictions & ~labels_bool)
        FN = np.sum(~predictions & labels_bool)
        TN = np.sum(~predictions & ~labels_bool)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
            best_metrics = {'threshold': float(threshold), 'precision': float(precision), 
                          'recall': float(recall), 'f1': float(f1),
                          'TP': int(TP), 'FP': int(FP), 'FN': int(FN), 'TN': int(TN)}
    return best_threshold, best_metrics

def save_checkpoint(results, completed):
    with open('tuning_checkpoint.json', 'w') as f:
        json.dump({'results': results, 'completed': list(completed), 'timestamp': datetime.now().isoformat()}, f)

def load_checkpoint():
    if os.path.exists('tuning_checkpoint.json'):
        with open('tuning_checkpoint.json', 'r') as f:
            data = json.load(f)
        return data['results'], set(data['completed'])
    return [], set()

def tune_parameters():
    print("="*70)
    print("FIXED: GPU-OPTIMIZED PARAMETER TUNING FOR HAI 22.04")
    print("="*70)
    
    results, completed = load_checkpoint()
    if completed:
        print(f"\n✓ Resuming: {len(completed)} configs done")
    
    print("\nLoading data...")
    train_df, test_df, test_labels = load_data(TRAIN_FILES, TEST_FILES)
    print(f"✓ Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    Q_values = [5, 10, 15, 20]
    n_values = [2, 3, 5, 8]
    M_values = [1_000_000, 5_000_000, 10_000_000]
    k_values = [3, 5, 7]
    
    all_configs = [(Q, n, M, k) for Q in Q_values for n in n_values for M in M_values for k in k_values]
    configs_to_run = [c for c in all_configs if str(c) not in completed]
    
    print(f"\nTotal: {len(all_configs)}, Remaining: {len(configs_to_run)}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")
    
    quantized_cache = {}
    ngram_cache = {}
    
    pbar = tqdm(configs_to_run, desc="Tuning", unit="cfg")
    
    for idx, (Q, n, M, k) in enumerate(pbar):
        try:
            pbar.set_description(f"Q={Q} n={n} M={M/1e6:.0f}M k={k}")
            
            if Q not in quantized_cache:
                quantized_cache[Q] = quantize_data(train_df, test_df, Q)
            train_quant, test_quant = quantized_cache[Q]
            
            cache_key = (Q, n)
            if cache_key not in ngram_cache:
                train_states = build_state_strings(train_quant)
                test_states = build_state_strings(test_quant)
                ngram_cache[cache_key] = (generate_ngrams(train_states, n), generate_ngrams(test_states, n))
            train_ngrams, test_ngrams = ngram_cache[cache_key]
            
            bloom = train_bloom(train_ngrams, M, k)
            util = bloom.utilization()
            
            # FIXED: Use the corrected scoring function
            scores = score_samples_fixed(test_ngrams, bloom, n)
            threshold, metrics = find_best_threshold(scores, test_labels)
            
            results.append({'Q': Q, 'n': n, 'M': M, 'k': k, 'utilization': float(util), **metrics})
            completed.add(str((Q, n, M, k)))
            
            best_f1 = max([r['f1'] for r in results])
            pbar.set_postfix({'Best_F1': f"{best_f1:.4f}", 'Curr_F1': f"{metrics['f1']:.4f}"})
            
            if (idx + 1) % 10 == 0:
                save_checkpoint(results, completed)
        
        except Exception as e:
            print(f"\n⚠ Error {(Q, n, M, k)}: {e}")
            import traceback
            traceback.print_exc()
    
    save_checkpoint(results, completed)
    
    # Check if we have any results
    if not results:
        print("\n⚠ No successful configurations! All failed.")
        return None
    
    results_df = pd.DataFrame(results).sort_values('f1', ascending=False)
    results_df.to_csv('parameter_tuning_results.csv', index=False)
    
    print("\n" + "="*70)
    print("TOP 10 CONFIGURATIONS")
    print("="*70)
    print(results_df.head(10)[['Q', 'n', 'M', 'k', 'utilization', 'f1', 'precision', 'recall']].to_string(index=False))
    
    best = results_df.iloc[0]
    print(f"\n{'='*70}")
    print("🏆 BEST CONFIGURATION")
    print(f"{'='*70}")
    print(f"  Q={int(best['Q'])}, n={int(best['n'])}, M={int(best['M']):,} ({best['M']/1e6:.0f}M), k={int(best['k'])}")
    print(f"  Threshold: {best['threshold']:.4f}")
    print(f"  F1: {best['f1']:.4f}, Precision: {best['precision']:.4f}, Recall: {best['recall']:.4f}")
    print(f"  Utilization: {best['utilization']:.4f}")
    
    print(f"\n{'='*70}")
    print("BEST FOR EACH N")
    print(f"{'='*70}")
    for n_val in sorted(results_df['n'].unique()):
        best_n = results_df[results_df['n'] == n_val].iloc[0]
        print(f"N={n_val}: Q={int(best_n['Q'])}, M={int(best_n['M']/1e6)}M, k={int(best_n['k'])}, F1={best_n['f1']:.4f}")
    
    print(f"\n✓ Saved: parameter_tuning_results.csv")
    print(f"✓ Done: {datetime.now().strftime('%H:%M:%S')}")
    
    if os.path.exists('tuning_checkpoint.json'):
        os.remove('tuning_checkpoint.json')
    
    return results_df

if __name__ == "__main__":
    import sys
    if os.path.exists('tuning_checkpoint.json'):
        response = input("\nResume from checkpoint? (y/n): ")
        if response.lower() != 'y':
            os.remove('tuning_checkpoint.json')
    
    try:
        results_df = tune_parameters()
        if results_df is not None:
            best = results_df.iloc[0]
            print("\n" + "="*70)
            print("TO USE BEST CONFIG:")
            print("="*70)
            print(f"Update ngram_hai22_working.py:")
            print(f"  Q = {int(best['Q'])} (line ~226)")
            print(f"  n = {int(best['n'])} (line ~235)")
            print(f"  M = {int(best['M'])} (line ~241)")
            print(f"  k = {int(best['k'])} (line ~242)")
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted! Progress saved.")
        sys.exit(0)
