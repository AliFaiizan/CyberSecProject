import numpy as np

rng = np.random.default_rng(42)

X_normal = np.load("synthetic_normal.npy")  # (179116, 86)
X_attack = np.load("synthetic_attack.npy")  # (885, 86)

print(f"Original: Normal={X_normal.shape}, Attack={X_attack.shape}")

rng.shuffle(X_normal)
rng.shuffle(X_attack)

# =================================================
# MIMIC HAI-22.04 EXACTLY
# =================================================

# TRAIN: 93,601 samples (100% NORMAL - like train1.csv)
N_TRAIN = 93601
X_train = X_normal[:N_TRAIN]
y_train = np.zeros(N_TRAIN, dtype=int)

# TEST: 86,400 samples (98.98% normal, 1.02% attack - like test1.csv)
N_TEST = 86400
N_TEST_NORMAL = 85515
N_TEST_ATTACK = 885

X_test_normal = X_normal[N_TRAIN:N_TRAIN + N_TEST_NORMAL]
X_test_attack = X_attack[:N_TEST_ATTACK]

# Create 7 attack blocks (like real HAI)
X_test_list = []
y_test_list = []

K = 7
attack_per_block = N_TEST_ATTACK // K
remainder = N_TEST_ATTACK % K
block_sizes = np.full(K, attack_per_block, dtype=int)
block_sizes[:remainder] += 1

# Distribute attacks from 24% to 94% of test set
first_normal = int(N_TEST_NORMAL * 0.24)
last_normal = int(N_TEST_NORMAL * 0.06)
middle_normal = N_TEST_NORMAL - first_normal - last_normal
middle_block_size = middle_normal // (K - 1)

normal_block_sizes = [first_normal]
for i in range(K - 1):
    normal_block_sizes.append(middle_block_size)
normal_block_sizes.append(last_normal)

normal_idx = 0
attack_idx = 0

for k in range(K):
    # Normal block
    n_size = normal_block_sizes[k]
    X_test_list.append(X_test_normal[normal_idx:normal_idx + n_size])
    y_test_list.append(np.zeros(n_size, dtype=int))
    normal_idx += n_size
    
    # Attack block
    a_size = block_sizes[k]
    X_test_list.append(X_test_attack[attack_idx:attack_idx + a_size])
    y_test_list.append(np.ones(a_size, dtype=int))
    attack_idx += a_size

# Final normal block
X_test_list.append(X_test_normal[normal_idx:])
y_test_list.append(np.zeros(len(X_test_normal) - normal_idx, dtype=int))

X_test = np.vstack(X_test_list)
y_test = np.hstack(y_test_list)

# Verify
assert len(X_test) == N_TEST, f"Expected {N_TEST}, got {len(X_test)}"

# Save
np.save("synthetic_train.npy", X_train)
np.save("synthetic_train_labels.npy", y_train)
np.save("synthetic_test.npy", X_test)
np.save("synthetic_test_labels.npy", y_test)

print("\n" + "="*60)
print("HAI-22.04 STRUCTURE (FOR VAE)")
print("="*60)
print(f"✓ TRAIN: {X_train.shape}, attacks: {y_train.sum()} (0.00%)")
print(f"✓ TEST:  {X_test.shape}, attacks: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
print("\nThis matches real HAI structure!")
print("VAE will train on 100% normal data")
print("="*60)
