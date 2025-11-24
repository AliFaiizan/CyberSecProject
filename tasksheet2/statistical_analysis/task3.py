import math
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from typing import List, Tuple

# File paths and column definitions
hai_21_train_files = sorted(glob("../hai-21.03/train1.csv"))
hai_21_test_files = sorted(glob("../hai-21.03/test1.csv"))
hai_21_attack_cols = ['attack', 'attack_P1', 'attack_P2', 'attack_P3']

print(f"Found {len(hai_21_train_files)} training files: {hai_21_train_files}")  
print(f"Found {len(hai_21_test_files)} test files: {hai_21_test_files}")

class BloomFilter:
    """Bloom filter implementation for n-gram anomaly detection"""
    
    def __init__(self, size=1_000_000, k=5):
        self.size = size
        self.k = k
        self.bloom = np.zeros(size, dtype=bool)
        
    def _hashes(self, item: str) -> List[int]:
        """Generate k hash values for an item"""
        return [
            int(hashlib.sha1((str(seed) + item).encode()).hexdigest(), 16) % self.size # c4a
            for seed in range(self.k)
        ]
    
    def add(self, item: str):
        """Add item to bloom filter"""
        for h in self._hashes(item):
            self.bloom[h] = True
    
    def check(self, item: str) -> bool:
        """Check if item might be in the filter"""
        return all(self.bloom[h] for h in self._hashes(item))

def load_and_clean_data(train_files: List[str], test_files: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 1: Load & Clean Data
    - Read all train CSVs for HAI 21.03
    - Drop timestamp and attack labels
    - Remove rows where Attack == 1 (use only normal data for training)
    """
    print("\n=== Step 1: Loading & Cleaning Data ===")
    
    # Load training data
    train_dfs = []
    for file in train_files:
        print(f"Loading {file}...")
        df = pd.read_csv(file)
        print(f"  Original shape: {df.shape}")
        
        # Remove attack rows (keep only normal data for training)
        if 'attack' in df.columns:
            normal_mask = df['attack'] == 0
            df = df[normal_mask]
            print(f"  After removing attacks: {df.shape}")
        
        train_dfs.append(df)
    
    # Load test data
    test_dfs = []
    for file in test_files:
        print(f"Loading {file}...")
        df = pd.read_csv(file)
        test_dfs.append(df)
    
    # Combine all data
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Drop timestamp and attack columns
    cols_to_drop = ['time'] + [col for col in hai_21_attack_cols if col in train_df.columns]
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"Final training data shape: {train_df.shape}")
    print(f"Final test data shape: {test_df.shape}") 
    
    # Handle NaN values
    train_df = train_df.fillna(method='ffill').fillna(0)
    test_df = test_df.fillna(method='ffill').fillna(0)

    return train_df, test_df

def normalize_and_quantize(train_data: pd.DataFrame, test_data: pd.DataFrame, Q: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 2: Normalize & Quantize
    - Use z-score normalization instead of min-max
    - Quantize into discrete bins (0 to Q-1)
    """
    print(f"\n=== Step 2: Normalizing & Quantizing (Q={Q}) ===")
    # Round all values before normalization

    # Calculate z-score normalization parameters from training data only
    mean_vals = train_data.mean()
    std_vals = train_data.std()
    
    print(f"Data ranges before normalization:")
    print(f"  Train: min={train_data.min().min():.2f}, max={train_data.max().max():.2f}")
    print(f"  Test: min={test_data.min().min():.2f}, max={test_data.max().max():.2f}")
    
    # Z-score normalization
    train_normalized = (train_data - mean_vals) / std_vals
    test_normalized = (test_data - mean_vals) / std_vals
    
    # Handle any remaining NaN/inf values
    train_normalized = train_normalized.fillna(0)
    test_normalized = test_normalized.fillna(0)
    train_normalized = train_normalized.replace([np.inf, -np.inf], 0)
    test_normalized = test_normalized.replace([np.inf, -np.inf], 0)
    
    # Clip extreme values to reasonable range (e.g., -3 to 3 standard deviations)
    train_normalized = np.clip(train_normalized, -3, 3)
    test_normalized = np.clip(test_normalized, -3, 3)
    
    # Rescale to [0, 1] range for quantization
    # Map [-3, 3] to [0, 1]
    train_scaled = (train_normalized + 3) / 6
    test_scaled = (test_normalized + 3) / 6
    
    # Quantize into discrete bins [0, Q-1]
    train_quantized = np.floor(train_scaled * Q).astype(int)
    test_quantized = np.floor(test_scaled * Q).astype(int)
    
    # Ensure values are in valid range
    train_quantized = np.clip(train_quantized, 0, Q-1)
    test_quantized = np.clip(test_quantized, 0, Q-1)
    
    print(f"Quantized ranges:")
    print(f"  Train: min={train_quantized.min()}, max={train_quantized.max()}")
    print(f"  Test: min={test_quantized.min()}, max={test_quantized.max()}")
    
    return train_quantized, test_quantized

def build_state_strings(quantized_data: np.ndarray) -> List[str]:
    """
    Step 3: Build State String per Time Step
    Convert each time row into a state string
    """
    print(f"\n=== Step 3: Building State Strings ===")
    
    state_strings = []
    for i, row in enumerate(quantized_data):
        state = "_".join(map(str, row))
        state_strings.append(state)
        
        if i < 3:  # Show first few examples
            print(f"  Row {i}: {state[:50]}...")
    
    print(f"Generated {len(state_strings)} state strings")
    return state_strings

def generate_ngrams(state_strings: List[str], n: int = 3) -> List[str]:
    """
    Generate N-grams
    Slide a window of size n over the sequence of state strings
    """
    print(f"\n=== Step 4: Generating {n}-grams ===")
    
    ngrams = []
    for i in range(len(state_strings) - n + 1):
        sequence = state_strings[i:i+n]
        ngram = "→".join(sequence)
        ngrams.append(ngram)
        
        if i < 3:  # Show first few examples
            print(f"  N-gram {i}: {ngram[:100]}...")
    
    print(f"Generated {len(ngrams)} n-grams")
    return ngrams

def train_bloom_filter(ngrams: List[str], M: int = 1_000_000, k: int = 5) -> BloomFilter:
    """
    Hash and Store in Bloom Filter
    Train the bloom filter with normal n-grams
    """
    print(f"\n=== Training Bloom Filter (M={M}, k={k}) ===")
    
    bloom = BloomFilter(size=M, k=k)
    
    for i, ngram in enumerate(ngrams):
        bloom.add(ngram)
        
        if i % 10000 == 0:
            print(f"  Added {i+1}/{len(ngrams)} n-grams")
    
    print(f"Bloom filter training completed!")
    return bloom

def detect_anomalies(
    test_ngrams: List[str],
    bloom_filter: BloomFilter,
    n: int,
    threshold: float = None
) -> Tuple[List[float], List[bool], float]:
    """
    Detect anomalies using trained Bloom filter
    Returns:
        scores: anomaly scores per n-gram window (N_new / n)
        labels: anomaly label (True = anomaly) based on threshold
        anomaly_rate: overall anomaly rate
    """
    print(f"\n=== Anomaly Detection ===")

    scores = []
    for i in range(len(test_ngrams) - n):
        window = test_ngrams[i:i+n]
        unseen = sum(not bloom_filter.check(g) for g in window)
        score = unseen / n
        scores.append(score)

    # Compute threshold if not provided (using 3σ rule from normal-like training behavior)
    if threshold is None:
        mu, sigma = np.mean(scores), np.std(scores)
        threshold = mu + 2 * sigma
        print(f"Auto threshold (mean + 3σ): {threshold:.4f}")

    labels = [s > threshold for s in scores]
    anomaly_rate = sum(labels) / len(labels)

    print(f"Anomaly rate: {anomaly_rate:.4f} ({sum(labels)}/{len(labels)})")
    return scores, labels, anomaly_rate

def optimal_params(n, p_fp): # number of elements, desired false positive rate
    m = -n * math.log(p_fp) / (math.log(2)**2) # the minimum number of bits needed to achieve your target false positive rate
    k = int(round((m/n) * math.log(2))) #the optimal number of hash functions that minimizes false positives for that filter size.
    return int(m), k

# Main execution
if __name__ == "__main__":
    print("=== HAI 21.03 N-gram Anomaly Detection ===")
    Q = 20
    #  Load and clean data
    train_df, test_df = load_and_clean_data(hai_21_train_files, hai_21_test_files)
    ngram_size = 2
    n = len(train_df) # number of training samples
    M, k = optimal_params(n, p_fp=0.01)
    print(f"Optimal Bloom filter size: {M} bits, Hash functions: {k}")
    # Normalize and quantize
    train_quantized, test_quantized = normalize_and_quantize(train_df, test_df, Q=Q)


    # # Build state strings
    train_states = build_state_strings(train_quantized.values) 
    test_states = build_state_strings(test_quantized.values)
    
    # # Generate n-grams

    train_ngrams = generate_ngrams(train_states, ngram_size)
    test_ngrams = generate_ngrams(test_states, ngram_size)
    
    m_values = [1_000_000, 5_000_000, 10_000_000]
    k_values = [2,3,4,5,7]
    q_values = [5,10,20]
    n_values = [2,3]

    results = []
    for M in m_values:
        for k in k_values:
            for Q in q_values:
                for ngram_size in n_values:
                    train_quant, test_quant = normalize_and_quantize(train_df, test_df, Q=Q)
                    train_states = build_state_strings(train_quant.values)
                    test_states  = build_state_strings(test_quant.values)
                    train_ngrams = generate_ngrams(train_states, ngram_size)
                    test_ngrams  = generate_ngrams(test_states, ngram_size)
                    bloom = BloomFilter(size=M, k=k)
                    for ng in train_ngrams: bloom.add(ng)
                    util = bloom.bloom.mean()
                    unseen = sum(not bloom.check(ng) for ng in test_ngrams)
                    anomaly_rate = unseen / len(test_ngrams)
                    results.append((M, k, Q, ngram_size, util, anomaly_rate))
                    print(f"M={M/1e6:.1f}M k={k} Q={Q} n={ngram_size} util={util:.3f} anomaly={anomaly_rate:.3f}")
    print(results)
    df_results = pd.DataFrame(results,
    columns=["M","k","Q","n","utilization","anomaly"])
    df_results.sort_values(by="anomaly").head(10)
    
        # Filter for utilization between 0.45 and 0.56
    util_range_df = df_results[(df_results['utilization'] >= 0.45) & (df_results['utilization'] <= 0.56)]

    # Find the row with the minimum anomaly in this range
    best_idx = util_range_df['anomaly'].idxmin()
    best_M, best_k, best_Q, best_n = util_range_df.loc[best_idx, ['M','k','Q','n']]
    print(f"Best config in util range → M={best_M}, k={best_k}, Q={best_Q}, n={best_n}")

    # # Train bloom filter
    bloom_filter = train_bloom_filter(train_ngrams, M=M, k=k)

    print(bloom_filter.bloom.mean())
    
    # # Detect anomalies
    test_scores, anomalies, anomaly_rate = detect_anomalies(test_ngrams, bloom_filter, n=ngram_size)
    
    print(f"\n=== Summary ===")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"N-gram order: {ngram_size}")
    print(f"Quantization levels:",Q)
    print(f"Bloom filter size: {M} bits")
    print(f"Hash functions:",k)
    print(f"Anomaly rate: {anomaly_rate:.4f}")
    
    # Save results
    results_df = pd.DataFrame({
        'anomaly_score': anomalies[:len(test_df)-n+1]  # Adjust length for n-grams
    })


