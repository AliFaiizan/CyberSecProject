python << 'EOF'
import numpy as np

# Check ORIGINAL GAN outputs
normal = np.load("synthetic_normal.npy")
attack = np.load("synthetic_attack.npy")

print("="*60)
print("ORIGINAL GAN OUTPUTS")
print("="*60)
print("\nNORMAL:")
print(f"  Shape: {normal.shape}")
print(f"  Mean: {normal.mean():.4f}")
print(f"  Std: {normal.std():.4f}")
print(f"  Min: {normal.min():.4f}")
print(f"  Max: {normal.max():.4f}")
print(f"  NaNs: {np.isnan(normal).sum()}")
print(f"  Infs: {np.isinf(normal).sum()}")

print("\nATTACK:")
print(f"  Shape: {attack.shape}")
print(f"  Mean: {attack.mean():.4f}")
print(f"  Std: {attack.std():.4f}")
print(f"  Min: {attack.min():.4f}")
print(f"  Max: {attack.max():.4f}")
print(f"  NaNs: {np.isnan(attack).sum()}")
print(f"  Infs: {np.isinf(attack).sum()}")

# Check for extreme outliers
print("\n" + "="*60)
print("OUTLIER CHECK")
print("="*60)
print(f"Normal values > 100k: {(np.abs(normal) > 100000).sum()}")
print(f"Attack values > 100k: {(np.abs(attack) > 100000).sum()}")
print(f"Normal values > 1M: {(np.abs(normal) > 1000000).sum()}")
print(f"Attack values > 1M: {(np.abs(attack) > 1000000).sum()}")

# Compare with expected HAI range
print("\n" + "="*60)
print("EXPECTED HAI RANGE: [-112, 54831]")
print("="*60)
print(f"Normal outside range: {((normal < -112) | (normal > 54831)).sum()}")
print(f"Attack outside range: {((attack < -112) | (attack > 54831)).sum()}")

EOF
