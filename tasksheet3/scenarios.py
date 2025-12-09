import numpy as np
import pandas as pd



def extract_attack_types(y: pd.Series):
    """
    Given a binary label Series y (0=normal, 1=attack),
    returns:
      - attack_type: per-row integer attack ID (0=normal, 1..N=attack periods)
      - intervals: table listing each attack period
    """
    # Make sure y is a Series of ints 0/1
    y_bin = (y.astype(int) != 0).astype(int)
    
    # Detect transitions
    prev = y_bin.shift(1, fill_value=0)
    change = y_bin - prev
    
    # Attack start and end indices # 0 indicate no state change
    start_idx = y_bin.index[change == 1].tolist() 
    end_idx   = y_bin.index[change == -1].tolist()
    
    # If the last row is still attack, close it
    if len(end_idx) < len(start_idx):
        end_idx.append(y_bin.index[-1])
    
    # Build intervals table
    intervals = pd.DataFrame({
        "attack_id": range(1, len(start_idx)+1),
        "start_index": start_idx,
        "end_index": end_idx
    })
    
    # Create per-row attack_type
    attack_type = pd.Series(0, index=y_bin.index, dtype=int)
    for i, row in intervals.iterrows():
        attack_type.loc[row.start_index:row.end_index] = row.attack_id # attack_type.loc[2:4] = 1
    
    return attack_type, intervals

def get_balanced_attack_indices(attack_type, held_out=None, train_ratio=0.8, seed=42):
    """
    Balanced, disjoint attack sample split for Scenario 2.

    Parameters
    ----------
    attack_type : array-like
        attack_type[i] = 0 (normal) or attack-type ID.
    held_out : int or None
        Attack type to EXCLUDE from training (Scenario 2).
    train_ratio : float
        Percentage of samples per attack type to use for training.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_attack_idx : np.ndarray
        Balanced, disjoint indices for training (n-1 attack types).
    test_attack_idx : np.ndarray
        Balanced, disjoint indices for testing (all attack types).
    """

    import numpy as np
    np.random.seed(seed)

    # All attack types in dataset (ignore 0)
    attack_types = np.unique(attack_type[attack_type != 0])

    # smallest available per attack type
    min_count = min(len(np.where(attack_type == a)[0]) for a in attack_types)

    # train/test counts per attack type
    train_count = int(min_count * train_ratio)
    test_count  = min_count - train_count

    train_indices = []
    test_indices = []

    for a in attack_types:
        idx_list = np.where(attack_type == a)[0]
        np.random.shuffle(idx_list)

        # Split evenly
        train_part = idx_list[:train_count]
        test_part  = idx_list[train_count:train_count + test_count]

        # If this is the held-out type, exclude from training
        if held_out is not None and a == held_out:
            test_indices.append(test_part)      # Only test samples
        else:
            train_indices.append(train_part)
            test_indices.append(test_part)

    # Combine for output
    train_attack_idx = np.concatenate(train_indices) if train_indices else np.array([], dtype=int)
    test_attack_idx  = np.concatenate(test_indices)

    return train_attack_idx, test_attack_idx

def make_kfold_indices(n_samples, k=5, seed=42): # generates indices for k-fold cross-validation.
    np.random.seed(seed) # seed for reproducibility
    indices = np.arange(n_samples)
    np.random.shuffle(indices) # only normal rows
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1 # distribute the samples as evenly as possible
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    return folds

def scenario_1_split(X, y, k=5, seed=42):
    """
    Scenario 1:
      Train on normal data only.
      Test on normal (current fold) + all attack samples.
    """
    attack_type, attack_intervals = extract_attack_types(y)

    normal_idx = np.where(y == 0)[0] # tuple returns indices for normal ( taking normal data indices)
    _, balanced_attack_idx = get_balanced_attack_indices(attack_type)

    folds = make_kfold_indices(len(normal_idx), k, seed)

    for fold_idx in range(k):
        test_normal_idx = normal_idx[folds[fold_idx]] # pick normal test samples for current fold
        train_normal_idx = np.setdiff1d(normal_idx, test_normal_idx) # pick whatever normal samples are not in test set

        train_idx = train_normal_idx                           # no attacks in training
        test_idx  = np.concatenate([test_normal_idx,           # normal fold
                                    balanced_attack_idx])      # balanced attack samples
        yield fold_idx, train_idx, test_idx

def scenario_2_split(X, y, k=5, seed=42):

    attack_type, attack_intervals = extract_attack_types(y)

    np.random.seed(seed)

    normal_idx = np.where(y == 0)[0]

    # randomly select one attack type to hold out
    held_out = int(np.random.choice(attack_intervals["attack_id"].unique()))

    # CORRECT UNPACKING
    train_attack_idx, _ = get_balanced_attack_indices(
        attack_type,
        held_out=held_out
    )

    _, test_attack_idx = get_balanced_attack_indices(
        attack_type,
        held_out=None
    )

    folds = make_kfold_indices(len(normal_idx), k=k, seed=seed)

    for fold_idx in range(k):
        test_normal_idx  = normal_idx[folds[fold_idx]]
        train_normal_idx = np.setdiff1d(normal_idx, test_normal_idx)

        train_idx = np.concatenate([train_normal_idx, train_attack_idx])
        test_idx  = np.concatenate([test_normal_idx, test_attack_idx])

        yield fold_idx, held_out, train_idx, test_idx

def scenario_3_split(X, y, k=5, seed=42):
    """
    Scenario 3 (simplified):
      - Train on normal + ONE selected attack type
      - Test on normal fold + all attack types
    """
    attack_type, attack_intervals = extract_attack_types(y)
    np.random.seed(None)

    normal_idx = np.where(y == 0)[0]
    attack_ids = attack_intervals["attack_id"].unique()
    selected_type = np.random.choice(attack_ids)  # Pick one attack type

    folds = make_kfold_indices(len(normal_idx), k=k, seed=seed)

    for fold_idx in range(k):
        test_normal_idx = normal_idx[folds[fold_idx]]
        train_normal_idx = np.setdiff1d(normal_idx, test_normal_idx)

        train_attack_idx = np.where(attack_type == selected_type)[0]
        test_attack_idx = np.where(attack_type != 0)[0]

        train_idx = np.concatenate([train_normal_idx, train_attack_idx])
        test_idx = np.concatenate([test_normal_idx, test_attack_idx])

        yield fold_idx, selected_type, train_idx, test_idx
