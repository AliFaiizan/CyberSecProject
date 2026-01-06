import numpy as np

rng = np.random.default_rng(42)

# Load full synthetic pools
X_normal = np.load("synthetic_normal.npy")   # ~179k
X_attack = np.load("synthetic_attack.npy")   # 885

# Shuffle normals once
rng.shuffle(X_normal)

# ----------------------------
# 1) TRAIN SET (NORMAL ONLY)
# ----------------------------
N_TRAIN_NORMAL = 140_000

X_train = X_normal[:N_TRAIN_NORMAL]
y_train = np.zeros(len(X_train), dtype=int)

np.save("synthetic_train.npy", X_train)
np.save("synthetic_train_labels.npy", y_train)

# ----------------------------
# 2) TEST SET (NORMAL + ATTACK)
# ----------------------------
X_test_normal = X_normal[N_TRAIN_NORMAL:]  # remaining normals (~39k)
rng.shuffle(X_test_normal)
rng.shuffle(X_attack)

# build 7 contiguous attack intervals
K = 7
lengths = np.full(K, len(X_attack)//K, dtype=int)
lengths[:len(X_attack) % K] += 1

positions = np.linspace(0, len(X_test_normal), K+2, dtype=int)[1:-1]

X_seq, y_seq = [], []
n0, a0 = 0, 0

for k in range(K):
    n1 = positions[k]

    # normal block
    X_seq.append(X_test_normal[n0:n1])
    y_seq.append(np.zeros(n1 - n0, dtype=int))

    # attack block
    a1 = a0 + lengths[k]
    X_seq.append(X_attack[a0:a1])
    y_seq.append(np.ones(lengths[k], dtype=int))

    n0, a0 = n1, a1

# remaining normals
X_seq.append(X_test_normal[n0:])
y_seq.append(np.zeros(len(X_test_normal) - n0, dtype=int))

X_test = np.vstack(X_seq)
y_test = np.hstack(y_seq)

np.save("synthetic_test.npy", X_test)
np.save("synthetic_test_labels.npy", y_test)

# ----------------------------
# SUMMARY
# ----------------------------
print("TRAIN:", X_train.shape, "attacks:", y_train.sum())
print("TEST :", X_test.shape, "attacks:", y_test.sum())
print("TOTAL:", len(X_train) + len(X_test))
print("Attack ratio (test):", y_test.mean())
print("Attack blocks:", K)
