import numpy as np

rng = np.random.default_rng(42)

Xn = np.load("synthetic_test_normal.npy")   # test normals
Xa = np.load("synthetic_attack.npy")        # attacks

rng.shuffle(Xn)
rng.shuffle(Xa)

K = 7  # number of attack intervals

lengths = np.full(K, len(Xa)//K, dtype=int)
lengths[:len(Xa) % K] += 1

pos = np.linspace(0, len(Xn), K+2, dtype=int)[1:-1]

X_seq, y_seq = [], []
a0, n0 = 0, 0

for k in range(K):
    n1 = pos[k]

    # normal block
    X_seq.append(Xn[n0:n1])
    y_seq.append(np.zeros(n1 - n0, dtype=int))

    # attack block
    a1 = a0 + lengths[k]
    X_seq.append(Xa[a0:a1])
    y_seq.append(np.ones(lengths[k], dtype=int))

    n0, a0 = n1, a1

# remaining normals
X_seq.append(Xn[n0:])
y_seq.append(np.zeros(len(Xn) - n0, dtype=int))

X_test = np.vstack(X_seq)
y_test = np.hstack(y_seq)

np.save("synthetic_test_temporal.npy", X_test)
np.save("synthetic_test_temporal_labels.npy", y_test)

print("Final test shape:", X_test.shape)
print("Attack blocks:", K)
print("Attack ratio:", y_test.mean())
print("Total attacks:", y_test.sum())
