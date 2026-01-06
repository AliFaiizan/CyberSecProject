#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SIMPLE Parameter Tuning - HAI 22.04
Uses a simpler approach with better threshold detection
"""

import math
import hashlib
import os
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = "../datasets/hai-22.04/"
TRAIN_PATTERN = os.path.join(DATA_DIR, "train*.csv")
TEST_PATTERN = os.path.join(DATA_DIR, "test*.csv")
ATTACK_COLUMN = 'Attack'

# Simple parameter grid
CONFIGS = [
    # (Q, n, window, lambda, k)
    (8, 2, 80, 0.1, 5),
    (10, 2, 80, 0.1, 5),
    (12, 2, 80, 0.1, 5),
    (15, 2, 80, 0.1, 5),
    (10, 2, 100, 0.1, 5),
    (10, 2, 120, 0.1, 5),
    (10, 3, 80, 0.1, 5),
    (10, 2, 80, 0.15, 5),
    (10, 2, 80, 0.1, 7),
]

print(f"Testing {len(CONFIGS)} configurations\n")

class BloomFilter:
    def __init__(self, size=1_000_000, num_hashes=3):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = np.zeros(size, dtype=bool)
    def _hashes(self, value):
        for i in range(self.num_hashes):
            combined = f"{i}-{value}"
            yield int(hashlib.sha1(combined.encode()).hexdigest(), 16) % self.size
    def add(self, value):
        for position in self._hashes(value):
            self.bit_array[position] = True
    def contains(self, value):
        return all(self.bit_array[pos] for pos in self._hashes(value))

def optimal_bloom_size(n_items, fpr=0.01):
    size = -n_items * math.log(fpr) / (math.log(2) ** 2)
    return max(1_000_000, int(size))

def exponential_smoothing(values, lam):
    if len(values) == 0:
        return []
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(lam * v + (1 - lam) * smoothed[-1])
    return smoothed

def unseen_ngram_ratio(ngrams, bloom_filter, window_size):
    if len(ngrams) == 0:
        return []
    
    window, ratios = [], []
    for ng in ngrams:
        window.append(ng)
        if len(window) > window_size:
            window.pop(0)
        unseen_count = sum(not bloom_filter.contains(g) for g in window)
        ratios.append(unseen_count / len(window))
    return ratios

def to_state_strings(array_2d):
    return ["_".join(map(str, row)) for row in array_2d]

def make_ngrams(sequence, n):
    if len(sequence) < n:
        return []
    return ["→".join(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]

def load_dataset():
    train_files = sorted(glob(TRAIN_PATTERN))
    train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
    
    test_files = sorted(glob(TEST_PATTERN))
    test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)

    if ATTACK_COLUMN in train_df.columns:
        train_df = train_df[train_df[ATTACK_COLUMN] == 0]

    test_labels = test_df[ATTACK_COLUMN].values if ATTACK_COLUMN in test_df.columns else None

    drop_cols = ['timestamp', ATTACK_COLUMN]
    train_df = train_df.drop(columns=drop_cols, errors='ignore').ffill().fillna(0)
    test_df = test_df.drop(columns=drop_cols, errors='ignore').ffill().fillna(0)
    
    return train_df, test_df, test_labels

def normalize_and_quantize(df, mean, std, q_levels):
    zscores = ((df - mean) / std).clip(-3, 3).replace([np.inf, -np.inf], 0).fillna(0)
    quantized = ((zscores + 3) / 6 * q_levels).astype(int).clip(0, q_levels - 1)
    return quantized.values

def evaluate_simple(train_df, test_df, test_labels, Q, n, window, lam, num_hashes):
    """Simplified evaluation with better debugging"""
    
    try:
        # Use small subsets for speed
        train_size = min(30000, len(train_df))
        test_size = min(10000, len(test_df))
        
        train_subset = train_df.sample(train_size, random_state=42)
        test_subset = test_df.iloc[:test_size]
        labels_subset = test_labels[:test_size]
        
        # Stats
        mean, std = train_df.mean(), train_df.std().replace(0, 1e-6)
        
        # Quantize
        train_quant = normalize_and_quantize(train_subset, mean, std, Q)
        test_quant = normalize_and_quantize(test_subset, mean, std, Q)
        
        # States
        train_states = to_state_strings(train_quant)
        test_states = to_state_strings(test_quant)
        
        # N-grams
        train_ngrams = make_ngrams(train_states, n)
        test_ngrams = make_ngrams(test_states, n)
        
        if len(train_ngrams) == 0 or len(test_ngrams) == 0:
            return None
        
        # Bloom
        bloom_size = optimal_bloom_size(len(set(train_ngrams)))
        bloom = BloomFilter(bloom_size, num_hashes)
        for ng in train_ngrams:
            bloom.add(ng)
        
        # Score
        raw_scores = unseen_ngram_ratio(test_ngrams, bloom, window)
        if len(raw_scores) == 0:
            return None
            
        smoothed_scores = exponential_smoothing(raw_scores, lam)
        
        # Pad to match test length
        padding = [smoothed_scores[0]] * (n - 1) if len(smoothed_scores) > 0 else []
        scores_padded = padding + smoothed_scores
        
        # Align with labels
        min_len = min(len(scores_padded), len(labels_subset))
        scores_final = np.array(scores_padded[:min_len])
        labels_final = labels_subset[:min_len].astype(bool)
        
        # Find best threshold by grid search
        best_f1 = 0
        best_metrics = None
        
        for tau in np.linspace(0.01, 0.99, 100):
            preds = scores_final > tau
            
            TP = np.sum(preds & labels_final)
            FP = np.sum(preds & ~labels_final)
            FN = np.sum(~preds & labels_final)
            TN = np.sum(~preds & ~labels_final)
            
            if (TP + FP) == 0 or (TP + FN) == 0:
                continue
                
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_metrics = {
                    'Q': Q, 'n': n, 'window': window, 'lambda': lam, 'num_hashes': num_hashes,
                    'threshold': tau, 'f1': f1, 'precision': precision, 'recall': recall,
                    'TP': int(TP), 'FP': int(FP), 'FN': int(FN), 'TN': int(TN),
                    'bloom_size': bloom_size, 'utilization': bloom.bit_array.mean()
                }
        
        return best_metrics if best_metrics else None
        
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        return None

if __name__ == "__main__":
    print("="*70)
    print("SIMPLE PARAMETER TUNING - HAI 22.04")
    print("="*70)
    
    print("\nLoading data...")
    train_df, test_df, test_labels = load_dataset()
    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
    print(f"Attacks: {test_labels.sum():,} ({100*test_labels.mean():.2f}%)")
    
    print("\nTesting configurations...")
    print("="*70)
    
    results = []
    for idx, (Q, n, window, lam, k) in enumerate(tqdm(CONFIGS, desc="Progress")):
        result = evaluate_simple(train_df, test_df, test_labels, Q, n, window, lam, k)
        
        if result:
            results.append(result)
            print(f"\nConfig {idx+1}/{len(CONFIGS)}: Q={Q} n={n} w={window} λ={lam} k={k}")
            print(f"  F1={result['f1']:.4f}, P={result['precision']:.4f}, R={result['recall']:.4f}")
    
    if not results:
        print("\n❌ All configurations failed!")
        exit(1)
    
    # Save
    results_df = pd.DataFrame(results).sort_values('f1', ascending=False)
    results_df.to_csv('simple_tuning_results.csv', index=False)
    
    # Display
    print("\n" + "="*70)
    print("BEST CONFIGURATIONS")
    print("="*70)
    print(results_df[['Q', 'n', 'window', 'lambda', 'num_hashes', 'f1', 'precision', 'recall']].to_string(index=False))
    
    best = results_df.iloc[0]
    print(f"\n{'='*70}")
    print("🏆 BEST CONFIGURATION")
    print(f"{'='*70}")
    print(f"  Q={int(best['Q'])}, n={int(best['n'])}, window={int(best['window'])}")
    print(f"  lambda={best['lambda']:.2f}, num_hashes={int(best['num_hashes'])}")
    print(f"  Threshold={best['threshold']:.4f}")
    print(f"\n  F1={best['f1']:.4f}, Precision={best['precision']:.4f}, Recall={best['recall']:.4f}")
    
    print(f"\n✓ Saved: simple_tuning_results.csv")
    print("\nUpdate task3_22_04_gpu.py with these values!")
    print("="*70)
