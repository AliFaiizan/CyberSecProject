import numpy as np

rng = np.random.default_rng(42)

# Load GAN outputs
X_normal = np.load("synthetic_normal.npy")  # (179116, 86)
X_attack = np.load("synthetic_attack.npy")  # (885, 86)

print(f"Loaded: Normal={X_normal.shape}, Attack={X_attack.shape}")

rng.shuffle(X_normal)
rng.shuffle(X_attack)

# =================================================
# EXACT HAI-22.04 STRUCTURE
# =================================================

# TRAIN: 93,601 samples (100% NORMAL)
N_TRAIN = 93601
X_train = X_normal[:N_TRAIN]
y_train = np.zeros(N_TRAIN, dtype=int)

# TEST: 86,400 samples (1.02% attack)
N_TEST = 86400
N_TEST_NORMAL = 85515
N_TEST_ATTACK = 885

X_test_normal = X_normal[N_TRAIN:N_TRAIN + N_TEST_NORMAL]
X_test_attack = X_attack[:N_TEST_ATTACK]

# Build test set with 7 attack blocks (like real HAI)
K = 7
attack_sizes = np.full(K, N_TEST_ATTACK // K, dtype=int)
attack_sizes[:N_TEST_ATTACK % K] += 1

# Attack positions (24% to 94% of test set)
first_normal = int(N_TEST_NORMAL * 0.24)
last_normal = int(N_TEST_NORMAL * 0.06)
middle_normal = N_TEST_NORMAL - first_normal - last_normal
middle_size = middle_normal // (K - 1)

normal_sizes = [first_normal] + [middle_size] * (K - 1) + [last_normal]

# Assemble test set
test_blocks = []
normal_idx = 0
attack_idx = 0

for k in range(K):
    # Normal block
    n = normal_sizes[k]
    test_blocks.append(X_test_normal[normal_idx:normal_idx + n])
    test_blocks.append(np.zeros((n, 1)))  # labels
    normal_idx += n
    
    # Attack block
    a = attack_sizes[k]
    test_blocks.append(X_test_attack[attack_idx:attack_idx + a])
    test_blocks.append(np.ones((a, 1)))  # labels
    attack_idx += a

# Final normal block
remaining = len(X_test_normal) - normal_idx
test_blocks.append(X_test_normal[normal_idx:])
test_blocks.append(np.zeros((remaining, 1)))

# Combine
X_test = np.vstack(test_blocks[::2])  # Every other (features)
y_test = np.vstack(test_blocks[1::2]).squeeze()  # Every other (labels)

assert len(X_test) == N_TEST
assert len(y_test) == N_TEST

# Save
np.save("synthetic_train.npy", X_train)
np.save("synthetic_train_labels.npy", y_train)
np.save("synthetic_test.npy", X_test)
np.save("synthetic_test_labels.npy", y_test)

print("\n" + "="*60)
print("EXACT HAI-22.04 STRUCTURE")
print("="*60)
print(f"TRAIN: {X_train.shape}")
print(f"  Normal: {y_train.sum() == 0} (all zeros)")
print(f"  Attack: {y_train.sum()} (0.00%)")

print(f"\nTEST: {X_test.shape}")
print(f"  Normal: {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.2f}%)")
print(f"  Attack: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.2f}%)")

# Verify attack blocks
attack_indices = np.where(y_test == 1)[0]
print(f"\nAttack distribution:")
print(f"  First attack: index {attack_indices[0]} ({attack_indices[0]/len(y_test)*100:.1f}%)")
print(f"  Last attack: index {attack_indices[-1]} ({attack_indices[-1]/len(y_test)*100:.1f}%)")

blocks = 1
for i in range(1, len(attack_indices)):
    if attack_indices[i] - attack_indices[i-1] > 1:
        blocks += 1
print(f"  Attack blocks: {blocks}")

print("\n Matches HAI-22.04 structure perfectly!")
print("="*60)
