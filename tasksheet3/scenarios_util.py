import numpy as np
import pandas as pd


# -------------------------------------------------------------
# 1) Extract attack intervals and assign attack IDs
# -------------------------------------------------------------
def extract_attack_types(y: pd.Series):
    """
    Turns binary labels into attack IDs:
       0 = normal
       1 = first attack interval
       2 = second attack interval
       ...
    Returns: attack_type array + DataFrame of intervals.
    """
    y_bin = (y.astype(int) != 0).astype(int)

    prev = y_bin.shift(1, fill_value=0)
    change = y_bin - prev

    start_idx = y_bin.index[change == 1].tolist()
    end_idx   = y_bin.index[change == -1].tolist()

    # If last region is still under attack
    if len(end_idx) < len(start_idx):
        end_idx.append(y_bin.index[-1])

    intervals = pd.DataFrame({
        "attack_id": range(1, len(start_idx) + 1),
        "start": start_idx,
        "end": end_idx
    })

    attack_type = pd.Series(0, index=y.index, dtype=int)
    for _, row in intervals.iterrows():
        attack_type.loc[row.start:row.end] = row.attack_id

    return attack_type, intervals


# -------------------------------------------------------------
# 2) Helper: get BALANCED samples per attack type
# -------------------------------------------------------------
def get_balanced_per_type(attack_type, seed=42):
    """
    Returns a dictionary: {attack_id: np.array(indices)}
    where each attack_id has EXACTLY the same number of samples
    (the minimum across all attack types).

    This is required by the tasksheet.
    """
    np.random.seed(seed)

    attack_ids = sorted([a for a in attack_type.unique() if a != 0])

    # Count per attack type
    per_type = {a: np.where(attack_type == a)[0] for a in attack_ids}

    # Determine minimum size → enforce equal balancing
    min_count = min(len(v) for v in per_type.values())

    balanced = {}
    for a in attack_ids:
        idx = per_type[a]
        np.random.shuffle(idx)
        balanced[a] = idx[:min_count]

    return balanced, attack_ids, min_count


# -------------------------------------------------------------
# 3) Helper: k-fold splitting for NORMAL samples only
# -------------------------------------------------------------
def make_kfold_indices(n_samples, k=5, seed=42):
    np.random.seed(seed)
    idx = np.arange(n_samples)
    np.random.shuffle(idx)

    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1

    folds = []
    current = 0
    for size in fold_sizes:
        folds.append(idx[current:current + size])
        current += size
    return folds


# -------------------------------------------------------------
# 4) SCENARIO 1: Normal-only training
# -------------------------------------------------------------
def scenario_1_split(X, y, k=5, seed=42):

    attack_type, intervals = extract_attack_types(y)

    normal_idx = np.where(y == 0)[0]

    # Balanced attacks: same number per attack type
    balanced, attack_ids, min_count = get_balanced_per_type(attack_type, seed)
    balanced_attack_idx = np.concatenate(list(balanced.values()))

    folds = make_kfold_indices(len(normal_idx), k=k, seed=seed)

    for fold_idx in range(k):
        test_normal = normal_idx[folds[fold_idx]]
        train_normal = np.setdiff1d(normal_idx, test_normal)

        train_idx = train_normal           # ONLY normal samples
        test_idx  = np.concatenate([test_normal, balanced_attack_idx])

        yield fold_idx, train_idx, test_idx


# -------------------------------------------------------------
# 5) SCENARIO 2: Train on n−1 attack types, test on ALL
# -------------------------------------------------------------
def scenario_2_split(X, y, k=5, seed=42):

    attack_type, intervals = extract_attack_types(y)

    # Balanced per attack type
    balanced, attack_ids, min_count = get_balanced_per_type(attack_type, seed)

    # Choose 1 attack type to HOLD OUT
    np.random.seed(seed)
    held_out = int(np.random.choice(attack_ids))

    train_attack = []
    test_attack  = []

    for a in attack_ids:
        idx_balanced = balanced[a]
        # Use the SAME number of samples per type
        if a == held_out:
            # Held-out → only appears in test set
            test_attack.append(idx_balanced)
        else:
            # n−1 attack types → included in training
            # Split 50/50 just to create disjoint sets
            half = len(idx_balanced) // 2
            train_attack.append(idx_balanced[:half])
            test_attack.append(idx_balanced[half:])

    train_attack_idx = np.concatenate(train_attack) if len(train_attack) else np.array([], int)
    test_attack_idx  = np.concatenate(test_attack)

    normal_idx = np.where(y == 0)[0]
    folds = make_kfold_indices(len(normal_idx), k=k, seed=seed)

    for fold_idx in range(k):

        test_normal = normal_idx[folds[fold_idx]]
        train_normal = np.setdiff1d(normal_idx, test_normal)

        train_idx = np.concatenate([train_normal, train_attack_idx])
        test_idx  = np.concatenate([test_normal, test_attack_idx])

        yield fold_idx, held_out, train_idx, test_idx


# -------------------------------------------------------------
# 6) SCENARIO 3: Train on exactly ONE attack type
# -------------------------------------------------------------
def scenario_3_split(X, y, k=5, seed=42):

    attack_type, intervals = extract_attack_types(y)

    balanced, attack_ids, min_count = get_balanced_per_type(attack_type, seed)

    # Pick ONE attack type for training
    np.random.seed(seed)
    chosen = int(np.random.choice(attack_ids))

    train_attack_idx = balanced[chosen]

    # Test contains balanced samples from ALL types
    test_attack_idx = np.concatenate(list(balanced.values()))

    normal_idx = np.where(y == 0)[0]
    folds = make_kfold_indices(len(normal_idx), k=k, seed=seed)

    for fold_idx in range(k):
        test_normal = normal_idx[folds[fold_idx]]
        train_normal = np.setdiff1d(normal_idx, test_normal)

        train_idx = np.concatenate([train_normal, train_attack_idx])
        test_idx  = np.concatenate([test_normal, test_attack_idx])

        yield fold_idx, chosen, train_idx, test_idx
