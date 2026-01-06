#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnostic Script - Figure out why detection is failing
"""

import numpy as np
import pandas as pd
from glob import glob
import os

DATA_DIR = "../datasets/hai-22.04/"
TRAIN_PATTERN = os.path.join(DATA_DIR, "train*.csv")
TEST_PATTERN = os.path.join(DATA_DIR, "test*.csv")

print("="*70)
print("DIAGNOSTIC ANALYSIS - HAI 22.04")
print("="*70)

# Load data
print("\n1. LOADING DATA...")
train_files = sorted(glob(TRAIN_PATTERN))
train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)

test_files = sorted(glob(TEST_PATTERN))
test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)

print(f"   Train: {len(train_df):,} rows")
print(f"   Test: {len(test_df):,} rows")

# Check columns
print("\n2. COLUMNS:")
print(f"   Train columns: {train_df.columns.tolist()[:5]}... ({len(train_df.columns)} total)")
print(f"   Test columns: {test_df.columns.tolist()[:5]}... ({len(test_df.columns)} total)")

# Check Attack column
print("\n3. ATTACK COLUMN:")
if 'Attack' in train_df.columns:
    print(f"   Train Attack values: {train_df['Attack'].value_counts().to_dict()}")
    print(f"   Train Attack rate: {train_df['Attack'].mean():.4f}")
else:
    print("   ⚠ NO 'Attack' column in train!")

if 'Attack' in test_df.columns:
    print(f"   Test Attack values: {test_df['Attack'].value_counts().to_dict()}")
    print(f"   Test Attack rate: {test_df['Attack'].mean():.4f}")
else:
    print("   ⚠ NO 'Attack' column in test!")

# Check data types
print("\n4. DATA TYPES:")
print(f"   Train dtypes: {train_df.dtypes.value_counts().to_dict()}")
print(f"   Test dtypes: {test_df.dtypes.value_counts().to_dict()}")

# Check for NaN
print("\n5. MISSING VALUES:")
print(f"   Train NaN count: {train_df.isna().sum().sum()}")
print(f"   Test NaN count: {test_df.isna().sum().sum()}")

# Check feature values
print("\n6. FEATURE VALUE RANGES:")
drop_cols = ['timestamp', 'Attack']
train_features = train_df.drop(columns=drop_cols, errors='ignore')
test_features = test_df.drop(columns=drop_cols, errors='ignore')

print(f"   Train feature count: {train_features.shape[1]}")
print(f"   Train min value: {train_features.min().min():.4f}")
print(f"   Train max value: {train_features.max().max():.4f}")
print(f"   Train mean: {train_features.mean().mean():.4f}")
print(f"   Train std: {train_features.std().mean():.4f}")

print(f"\n   Test feature count: {test_features.shape[1]}")
print(f"   Test min value: {test_features.min().min():.4f}")
print(f"   Test max value: {test_features.max().max():.4f}")
print(f"   Test mean: {test_features.mean().mean():.4f}")
print(f"   Test std: {test_features.std().mean():.4f}")

# Sample rows
print("\n7. SAMPLE DATA (first 3 rows of train):")
print(train_df.head(3))

print("\n8. SAMPLE DATA (first 3 rows of test with attacks):")
if 'Attack' in test_df.columns:
    attack_samples = test_df[test_df['Attack'] == 1].head(3)
    if len(attack_samples) > 0:
        print(attack_samples)
    else:
        print("   ⚠ NO ATTACKS FOUND IN TEST DATA!")
else:
    print("   ⚠ NO 'Attack' COLUMN!")

# Check timestamp
print("\n9. TIMESTAMP FORMAT:")
if 'timestamp' in train_df.columns:
    print(f"   Train timestamp sample: {train_df['timestamp'].iloc[0]}")
    print(f"   Test timestamp sample: {test_df['timestamp'].iloc[0]}")

# Quick anomaly test
print("\n10. QUICK ANOMALY SCORE TEST:")
print("    Testing if normal vs attack data looks different...")

if 'Attack' in test_df.columns and test_df['Attack'].sum() > 0:
    test_normal = test_features[test_df['Attack'] == 0].head(100)
    test_attack = test_features[test_df['Attack'] == 1].head(100)
    
    normal_mean = test_normal.mean().mean()
    attack_mean = test_attack.mean().mean()
    normal_std = test_normal.std().mean()
    attack_std = test_attack.std().mean()
    
    print(f"    Normal samples - Mean: {normal_mean:.4f}, Std: {normal_std:.4f}")
    print(f"    Attack samples - Mean: {attack_mean:.4f}, Std: {attack_std:.4f}")
    print(f"    Difference: {abs(attack_mean - normal_mean):.4f}")
    
    if abs(attack_mean - normal_mean) < 0.01:
        print("    ⚠ WARNING: Normal and attack samples look VERY SIMILAR!")
        print("    This could be why detection is failing.")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)

# Save detailed info
output = {
    'train_shape': train_df.shape,
    'test_shape': test_df.shape,
    'train_attack_rate': float(train_df['Attack'].mean()) if 'Attack' in train_df.columns else None,
    'test_attack_rate': float(test_df['Attack'].mean()) if 'Attack' in test_df.columns else None,
    'feature_count': train_features.shape[1],
}

import json
with open('diagnostic_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n✓ Saved: diagnostic_results.json")
