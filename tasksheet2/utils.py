from typing import List, Tuple
import pandas as pd
import numpy as np


def load_and_clean_data(train_files: List[str], test_files: List[str], attack_cols: List[str]=None, label_files: List[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:

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
    results = []

    # Create all parameter combinations manually
    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combo in itertools.product(*values):
        
        params = dict(zip(keys, combo))

        fold_scores = []

        for fold_idx, train_idx, test_idx in scenario_fn(X, y):
            X_train = X.iloc[train_idx]
            X_test  = X.iloc[test_idx]
            y_test  = y.iloc[test_idx]

            # Create model with current param combo
            model = model_builder(params)
            print(f"Fold:{fold_idx} Fitting model with params: {params}")
            model.fit(X_train)
            print("Model fitting complete.")
            # One-class models output +1 (normal) / -1 (attack)
            raw_pred = model.predict(X_test)
            y_pred = (raw_pred == -1).astype(int)  # attack=1, normal=0

            acc = np.mean(y_pred == y_test)
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

def get_balanced_attack_indices(attack_type):
    """
    Returns a balanced set of attack sample indices.
    Each attack type contributes the same number of samples.

    attack_type: array-like of shape (n_samples,)
        attack_type[i] = 0 (normal) or attack_typeID (1,2,3,...)

    Returns:
        balanced_attack_idx: np.array of indices
    """

    # attack type = 0 is normals --> get unique attack types [1,2,3,...]
    attack_types = np.unique(attack_type[attack_type != 0])

    attack_by_type = {
        a: np.where(attack_type == a)[0] for a in attack_types # find all indices where attack_type equals a
    }

    # find the smallest number of samples among all attack types
    min_count = min(len(idx_list) for idx_list in attack_by_type.values())

    balanced_list = []
    for a, idx_list in attack_by_type.items(): # idx_list contains all indices for attack type a
        chosen = np.random.choice(idx_list, min_count, replace=False) # randomly select min_count samples from each attack type without replacement
        balanced_list.append(chosen)

    # 5. Combine into a single array
    balanced_attack_idx = np.concatenate(balanced_list)

    return balanced_attack_idx