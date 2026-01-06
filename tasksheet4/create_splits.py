import numpy as np

rng = np.random.default_rng(42)

# Load synthetic data
X_normal = np.load("synthetic_normal.npy")  # (200000, 86)
X_attack = np.load("synthetic_attack.npy")  # (885, 86)

print(f"Loaded: Normal={X_normal.shape}, Attack={X_attack.shape}")

# =================================================
# TRAIN SET (100% NORMAL - like train1.csv)
# =================================================
N_TRAIN = 93601

X_train = X_normal[:N_TRAIN]
y_train = np.zeros(N_TRAIN, dtype=int)

np.save("synthetic_train.npy", X_train)
np.save("synthetic_train_labels.npy", y_train)

print(f"✓ TRAIN: {X_train.shape}, attacks: {y_train.sum()} (0.00%)")

# =================================================
# TEST SET (NORMAL + 7 ATTACK BLOCKS - like test1.csv)
# =================================================
N_TEST = 86400
N_TEST_ATTACK = 885
N_TEST_NORMAL = N_TEST - N_TEST_ATTACK  # 85515

# Get test normals and attacks
X_test_normal = X_normal[N_TRAIN:N_TRAIN + N_TEST_NORMAL]
X_test_attack = X_attack[:N_TEST_ATTACK]

# Shuffle
rng.shuffle(X_test_normal)
rng.shuffle(X_test_attack)

# =================================================
# INSERT 7 ATTACK BLOCKS
# =================================================
K = 7

# Split attacks into K blocks
attack_per_block = N_TEST_ATTACK // K
remainder = N_TEST_ATTACK % K
block_sizes = np.full(K, attack_per_block, dtype=int)
block_sizes[:remainder] += 1

# Split normals into K+1 segments (normal blocks between/around attacks)
# Real HAI: attacks span from ~24% to ~94% of test set
# So: [normal_segment_0] [attack_0] [normal_1] [attack_1] ... [normal_K]

# Calculate normal block sizes
normal_blocks = K + 1
# First 24% is normal, last 6% is normal, middle 70% has alternating
first_normal = int(N_TEST_NORMAL * 0.24)
last_normal = int(N_TEST_NORMAL * 0.06)
middle_normal = N_TEST_NORMAL - first_normal - last_normal

# Distribute middle normals among K-1 blocks
middle_block_size = middle_normal // (K - 1)
middle_remainder = middle_normal % (K - 1)

normal_block_sizes = [first_normal]
for i in range(K - 1):
    size = middle_block_size + (1 if i < middle_remainder else 0)
    normal_block_sizes.append(size)
normal_block_sizes.append(last_normal)

# Build sequence
X_seq = []
y_seq = []
normal_idx = 0
attack_idx = 0

for k in range(K):
    # Add normal block
    n_size = normal_block_sizes[k]
    X_seq.append(X_test_normal[normal_idx:normal_idx + n_size])
    y_seq.append(np.zeros(n_size, dtype=int))
    normal_idx += n_size
    
    # Add attack block
    a_size = block_sizes[k]
    X_seq.append(X_test_attack[attack_idx:attack_idx + a_size])
    y_seq.append(np.ones(a_size, dtype=int))
    attack_idx += a_size

# Add final normal block
X_seq.append(X_test_normal[normal_idx:])
y_seq.append(np.zeros(len(X_test_normal) - normal_idx, dtype=int))

X_test = np.vstack(X_seq)
y_test = np.hstack(y_seq)

# Verify size
assert len(X_test) == N_TEST, f"Expected {N_TEST}, got {len(X_test)}"

np.save("synthetic_test.npy", X_test)
np.save("synthetic_test_labels.npy", y_test)

print(f"✓ TEST:  {X_test.shape}, attacks: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# =================================================
# VERIFY
# =================================================
print("\n" + "="*60)
print("VERIFICATION")
print("="*60)
print(f"TRAIN: {X_train.shape} | Normal: {len(y_train) - y_train.sum()} | Attack: {y_train.sum()}")
print(f"TEST:  {X_test.shape} | Normal: {len(y_test) - y_test.sum()} | Attack: {y_test.sum()}")
print(f"TOTAL: ({len(X_train) + len(X_test)}, 86)")

# Check attack blocks
attack_indices = np.where(y_test == 1)[0]
blocks = 1
for i in range(1, len(attack_indices)):
    if attack_indices[i] - attack_indices[i-1] > 1:
        blocks += 1
print(f"Attack blocks: {blocks}")
print(f"First attack: index {attack_indices[0]} ({attack_indices[0]/len(y_test)*100:.1f}%)")
print(f"Last attack: index {attack_indices[-1]} ({attack_indices[-1]/len(y_test)*100:.1f}%)")

print("\n✓ Matches HAI-22.04 structure!")
