from utils import load_and_clean_data
from glob import glob
import numpy as np
import pandas as pd


train_files = sorted(glob("../haiend-23.05/end-train1.csv"))
test_files = sorted(glob("../haiend-23.05/end-test1.csv"))
label_files = sorted(glob("../haiend-23.05/label-test1.csv"))

haiEnd_df = load_and_clean_data(train_files, test_files, attack_cols=None, label_files=label_files) # merge train and test data


X = haiEnd_df.drop(columns=['label', 'timestamp'], errors='ignore') # label here refers to attack label 0 or 1
y = haiEnd_df['label']

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
    
    # Attack start and end indices
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
        attack_type.loc[row.start_index:row.end_index] = row.attack_id
    
    return attack_type, intervals

attack_type, attack_intervals = extract_attack_types(y)


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

def scenario_1_split(X, y, k=5, seed=42, balance_attacks=False):
    """
    Scenario 1:
      Train on normal data only.
      Test on normal (current fold) + all attack samples.
    """
    normal_idx = np.where(y == 0)[0] # returns indices where condition is met ( taking normal data indices)
    attack_idx = np.where(y == 1)[0]
    folds = make_kfold_indices(len(normal_idx), k, seed)

    for fold_idx in range(k):
        test_normal_idx = normal_idx[folds[fold_idx]] # pick normal test samples for current fold
        train_normal_idx = np.setdiff1d(normal_idx, test_normal_idx) # pick whatever normal samples are not in test set

        # Optionally balance attack samples in test
        if balance_attacks:
            n_attack = len(test_normal_idx)
            attack_sample_idx = np.random.choice(attack_idx, n_attack, replace=False)
        else:
            attack_sample_idx = attack_idx

        test_idx = np.concatenate([test_normal_idx, attack_sample_idx]) # both normal and attack samples in test set
        train_idx = train_normal_idx # only normal samples in train set

        yield fold_idx, train_idx, test_idx

def scenario_2_split(X, y, attack_type, attack_intervals, k=5, seed=42):
    """
    Scenario 2:
      - Train on normal + (n−1) attack types (i.e., exclude one attack type)
      - Test on normal fold + all attack types
    """
    np.random.seed(seed)

    normal_idx = np.where(y == 0)[0]
    attack_ids = attack_intervals["attack_id"].unique()

    folds = make_kfold_indices(len(normal_idx), k=k, seed=seed)

    for fold_idx in range(k):
        # normal samples for this fold
        test_normal_idx = normal_idx[folds[fold_idx]]
        train_normal_idx = np.setdiff1d(normal_idx, test_normal_idx)

        # loop through each attack type to hold out
        for held_out in attack_ids:

            # training attack = all except the held_out type
            train_attack_idx = np.where(
                (attack_type != 0) & (attack_type != held_out)
            )[0]

            # test attack = all attack samples
            test_attack_idx = np.where(attack_type != 0)[0]

            train_idx = np.concatenate([train_normal_idx, train_attack_idx]) # train on normal + (n-1) attack types
            test_idx = np.concatenate([test_normal_idx, test_attack_idx]) # test on normal fold + all attack types

            yield fold_idx, held_out, train_idx, test_idx

def scenario_3_split(X, y, attack_type, attack_intervals, k=5, seed=42):
    """
    Scenario 3:
      - Train on normal + exactly ONE attack type
      - Test on normal fold + all attack types
    """
    np.random.seed(seed)

    normal_idx = np.where(y == 0)[0]
    attack_ids = attack_intervals["attack_id"].unique()
    
    folds = make_kfold_indices(len(normal_idx), k=k, seed=seed)

    for fold_idx in range(k):
        # normal fold for testing
        test_normal_idx = normal_idx[folds[fold_idx]]
        train_normal_idx = np.setdiff1d(normal_idx, test_normal_idx)

        for selected_type in attack_ids:
            # training attack = exactly this one type
            train_attack_idx = np.where(attack_type == selected_type)[0]

            # test attack = all attack samples
            test_attack_idx = np.where(attack_type != 0)[0]

            train_idx = np.concatenate([train_normal_idx, train_attack_idx])
            test_idx = np.concatenate([test_normal_idx, test_attack_idx])

            yield fold_idx, selected_type, train_idx, test_idx

from .models import run_OneClassSVM
results = run_OneClassSVM(X, y, scenario_1_split)
# Access example
for fold_idx, test_idx, y_pred, y_test in results:
    print(f"Fold {fold_idx}: detected {y_pred.sum()} attacks")

for idx, pred in zip(test_idx, y_pred):
    print(f"Row {idx}: {'ATTACK' if pred==1 else 'NORMAL'}")


from .models import run_EllipticEnvelope
results_ee = run_EllipticEnvelope(X, y, scenario_1_split)

from .models import run_LOF
results_lof = run_LOF(X, y, scenario_1_split)
