import numpy as np

rng = np.random.default_rng(42)

# Load existing data
X_normal = np.load("synthetic_normal.npy")  # (179116, 86)
X_attack = np.load("synthetic_attack.npy")  # (885, 86)

print(f"Original: Normal={X_normal.shape}, Attack={X_attack.shape}")

# Need: 93601 (train) + 85515 (test normal) = 179116
# Plus some buffer for positioning = ~20k more
NEEDED_TOTAL = 200000  # Safe buffer

shortage = NEEDED_TOTAL - len(X_normal)

if shortage > 0:
    print(f"Need {shortage} more samples. Duplicating with shuffle...")
    # Randomly duplicate samples
    extra_indices = rng.choice(len(X_normal), shortage, replace=True)
    X_extra = X_normal[extra_indices]
    
    # Add small noise to duplicates (optional - makes them slightly different)
    noise = np.random.normal(0, 0.01, X_extra.shape)
    X_extra = X_extra + noise
    
    X_normal_expanded = np.vstack([X_normal, X_extra])
else:
    X_normal_expanded = X_normal

# Shuffle all normals
rng.shuffle(X_normal_expanded)

# Save
np.save("synthetic_normal.npy", X_normal_expanded)
print(f"✓ Expanded synthetic_normal.npy: {X_normal_expanded.shape}")
