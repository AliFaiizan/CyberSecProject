#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 3 – N-gram Anomaly Detection (GAN-generated data)
-----------------------------------------------------
• Uses synthetic_train.npy as normal training data
• Uses synthetic_test.npy as evaluation stream
• Same N-gram, quantization, Bloom filter logic as original HAI experiment
"""

import numpy as np
import pandas as pd
import hashlib
from typing import List, Tuple

# ============================================================
# CONFIGURATION
# ============================================================

SYNTHETIC_TRAIN_PATH = "synthetic_train.npy"
SYNTHETIC_TEST_PATH  = "synthetic_test.npy"

N_GRAM_ORDER = 8        # change to {2,5,8}
Q_LEVELS = 20
BLOOM_SIZE = 100_000
NUM_HASHES = 5

# ============================================================
# BLOOM FILTER
# ============================================================

class BloomFilter:
    def __init__(self, size=1_000_000, k=5):
        self.size = size
        self.k = k
        self.bloom = np.zeros(size, dtype=bool)

    def _hashes(self, item: str):
        for seed in range(self.k):
            yield int(
                hashlib.sha1((str(seed) + item).encode()).hexdigest(),
                16
            ) % self.size

    def add(self, item: str):
        for h in self._hashes(item):
            self.bloom[h] = True

    def check(self, item: str) -> bool:
        return all(self.bloom[h] for h in self._hashes(item))


# ============================================================
# PREPROCESSING
# ============================================================

def normalize_and_quantize(
    train: np.ndarray,
    test: np.ndarray,
    Q: int
) -> Tuple[np.ndarray, np.ndarray]:

    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1e-6

    train_z = (train - mean) / std
    test_z  = (test - mean) / std

    train_z = np.clip(train_z, -3, 3)
    test_z  = np.clip(test_z, -3, 3)

    train_q = np.floor((train_z + 3) / 6 * Q).astype(int)
    test_q  = np.floor((test_z  + 3) / 6 * Q).astype(int)

    train_q = np.clip(train_q, 0, Q - 1)
    test_q  = np.clip(test_q, 0, Q - 1)

    return train_q, test_q


def build_state_strings(data: np.ndarray) -> List[str]:
    return ["_".join(map(str, row)) for row in data]


def generate_ngrams(states: List[str], n: int) -> List[str]:
    return [
        "→".join(states[i:i+n])
        for i in range(len(states) - n + 1)
    ]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n=== GAN N-gram Anomaly Detection ===")

    # Load GAN data
    train_data = np.load(SYNTHETIC_TRAIN_PATH)
    test_data  = np.load(SYNTHETIC_TEST_PATH)

    print(f"GAN train shape: {train_data.shape}")
    print(f"GAN test  shape: {test_data.shape}")

    # Normalize + quantize
    train_q, test_q = normalize_and_quantize(
        train_data,
        test_data,
        Q_LEVELS
    )

    # Build state strings
    train_states = build_state_strings(train_q)
    test_states  = build_state_strings(test_q)

    # Generate n-grams
    train_ngrams = generate_ngrams(train_states, N_GRAM_ORDER)
    test_ngrams  = generate_ngrams(test_states, N_GRAM_ORDER)

    # Train Bloom filter
    bloom = BloomFilter(size=BLOOM_SIZE, k=NUM_HASHES)
    for ng in train_ngrams:
        bloom.add(ng)

    # Detect anomalies
    anomaly_flags = []
    for ng in test_ngrams:
        anomaly_flags.append(not bloom.check(ng))

    anomaly_rate = np.mean(anomaly_flags)

    print("\n=== Summary ===")
    print(f"N-gram order : {N_GRAM_ORDER}")
    print(f"Q levels     : {Q_LEVELS}")
    print(f"Bloom size   : {BLOOM_SIZE}")
    print(f"Hash funcs  : {NUM_HASHES}")
    print(f"Anomaly rate: {anomaly_rate:.4f}")

    # Save results
    out = pd.DataFrame({
        "is_anomaly": anomaly_flags
    })
    out.to_csv(f"task3_gan_ngram_N{N_GRAM_ORDER}.csv", index=False)

    print(f"✓ Results saved to task3_gan_ngram_N{N_GRAM_ORDER}.csv")
