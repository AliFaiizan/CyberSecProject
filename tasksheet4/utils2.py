from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def load_and_clean_data(train_files: List[str], test_files: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Load & Clean Data
    - Read all train CSVs for HAIend 22.04
    - Drop timestamp and attack labels
    - Remove rows where Attack == 1 (use only normal data for training)
    """
    print("\n=== Loading & Cleaning Data ===")
    
    # Load training data
    train_dfs = []
    for file in train_files:
        print(f"Loading {file}...")
        df = pd.read_csv(file)
        print(f"  Original shape: {df.shape}")
        
        train_dfs.append(df)
    
    # Load test data
    test_dfs = []
    for file in test_files:
        print(f"Loading {file}...")
        df = pd.read_csv(file)
        print(f"  Original shape: {df.shape}")
        
        test_dfs.append(df)
    
    # Combine all data
    train_df = pd.concat(train_dfs, ignore_index=True) 
    test_df = pd.concat(test_dfs, ignore_index=True)

    # Drop timestamp and attack columns
    cols_to_drop = ['timestamp'] 

        
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"Final training data shape: {train_df.shape}")
    print(f"Final test data shape: {test_df.shape}") 
    
    merged_dataset = pd.concat([train_df, test_df], ignore_index=True)
    attack_count = (merged_dataset['Attack'] == 1).sum()
    print(f"Total attack rows in merged dataset: {attack_count}")
    return merged_dataset


def optimal_param_search(X, y, scenario_fn, model_builder, param_grid):
    """
    model_builder(params) → returns a model instance
    param_grid = dict of lists
    """

    best_params = None
    best_score = -1
    acc = None
    results = []

    # Create all parameter combinations manually
    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    print("Starting Parameter search...")

    pca = PCA(n_components=0.95)
    for combo in itertools.product(*values):
        
        params = dict(zip(keys, combo))

        fold_scores = []

        for res in scenario_fn(X, y):
            if(len(res) == 3):
                fold_idx, train_idx, test_idx = res
                X_train = X.iloc[train_idx]
                X_test  = X.iloc[test_idx]
                y_test  = y.iloc[test_idx]

                model = model_builder(params)
                print(f"Fold:{fold_idx} Fitting model with params: {params}")
                model.fit(X_train)
                print("Model fitting complete.")
                # One-class models output +1 (normal) / -1 (attack)
                raw_pred = model.predict(X_test)
                y_pred = (raw_pred == -1).astype(int)  # attack=1, normal=0

                acc = np.mean(y_pred == y_test)
                fold_scores.append(acc)
            elif(len(res) == 4):
                fold_idx, attack_id, train_idx, test_idx = res
                X_train = X.iloc[train_idx]
                X_test  = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test  = y.iloc[test_idx]
                
                X_train_reduced = pca.fit_transform(X_train)
                X_test_reduced = pca.transform(X_test)

                model = model_builder(params)

                model.fit(X_train_reduced, y_train)
                y_pred = model.predict(X_test_reduced)
                acc = (y_pred == y_test).mean()
                fold_scores.append(acc)

        avg_score = np.mean(fold_scores)
        results.append((params, avg_score))

        print(f"Params {params} → Avg Acc = {avg_score:.4f}")

        # Keep best
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    print("\nBEST PARAMS:", best_params, "Score:", best_score)
    return best_params, results

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