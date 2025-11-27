import os
from utils import load_and_clean_data, get_balanced_attack_indices
from exports import export_model_output
from glob import glob
import numpy as np
import pandas as pd
import argparse

train_files = sorted(glob("../datasets/hai-22.04/train1.csv"))
test_files = sorted(glob("../datasets/hai-22.04/test1.csv"))
# label_files = sorted(glob("../datasets/hai-22.04/label-test*.csv"))

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
    attack_type, attack_intervals = extract_attack_types(y)

    normal_idx = np.where(y == 0)[0] # tuple returns indices for normal ( taking normal data indices)
    balanced_attack_idx = get_balanced_attack_indices(attack_type)

    folds = make_kfold_indices(len(normal_idx), k, seed)

    for fold_idx in range(k):
        test_normal_idx = normal_idx[folds[fold_idx]] # pick normal test samples for current fold
        train_normal_idx = np.setdiff1d(normal_idx, test_normal_idx) # pick whatever normal samples are not in test set

        train_idx = train_normal_idx                           # no attacks in training
        test_idx  = np.concatenate([test_normal_idx,           # normal fold
                                    balanced_attack_idx])      # balanced attack samples
        yield fold_idx, train_idx, test_idx

def scenario_2_split(X, y, k=5, seed=42):
    """
    Scenario 2:
      - Train on normal + (n−1) attack types (i.e., exclude one attack type)
      - Test on normal fold + all attack types
    """
    
    attack_type, attack_intervals = extract_attack_types(y)

    np.random.seed(None)

    normal_idx = np.where(y == 0)[0]
    held_out = np.random.choice(attack_intervals["attack_id"].unique()) # randomly select one attack type to hold out
    train_attack_idx = get_balanced_attack_indices(attack_type, exclude_type=held_out)
    test_attack_idx = get_balanced_attack_indices(attack_type, exclude_type=None)

    folds = make_kfold_indices(len(normal_idx), k=k, seed=seed)

    for fold_idx in range(k):
        test_normal_idx = normal_idx[folds[fold_idx]]
        train_normal_idx = np.setdiff1d(normal_idx, test_normal_idx)

        train_idx = np.concatenate([train_normal_idx, train_attack_idx])
        test_idx = np.concatenate([test_normal_idx, test_attack_idx])

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

from models import run_OneClassSVM, run_LOF, run_EllipticEnvelope, run_knn, run_binary_svm, run_random_forest

def _internal_main(args):

    # Load data
    haiEnd_df = load_and_clean_data(train_files, test_files)
    X = haiEnd_df.drop(columns=['Attack', 'timestamp'], errors='ignore')
    y = haiEnd_df['Attack']

    # If export-only mode, just export and exit
    if args.export:
        from exports import export_scenario_1, export_scenario_2, export_scenario_3
        
        if args.export == '1':
            export_scenario_1(haiEnd_df, X, y, scenario_1_split, out_dir="exports/Scenario1")
        elif args.export == '2':
            export_scenario_2(haiEnd_df, X, y, scenario_2_split, out_dir="exports/Scenario2")
        elif args.export == '3':
            export_scenario_3(haiEnd_df, X, y, scenario_3_split, out_dir="exports/Scenario3")
        return

    # Validate: one-class models use only scenario 1, supervised models need scenario 2 or 3
    one_class_models = ['ocsvm', 'lof', 'ee']
    supervised_models = ['knn', 'svm', 'rf']

    if args.model in one_class_models:
        print("\n**This model only support scenario 1**")
        args.scenario = '1'  # Force scenario 1 for one-class models
    elif args.model in supervised_models:
        if args.scenario is None or args.scenario == '1':
            print("Error: Models 'knn', 'svm', 'rf' require --scenario 2 or 3")
            return

    # Choose scenario function
    if args.scenario == '1':
        scenario_fn = scenario_1_split
    elif args.scenario == '2':
        scenario_fn = scenario_2_split
    elif args.scenario == '3':
        scenario_fn = scenario_3_split

    # Run selected model
    #TODO pass kfold indices to model functions
    if args.model == 'ocsvm':
        results = run_OneClassSVM(X, y, args.kfold, scenario_fn)
        out_dir = f"exports/Scenario{args.scenario}/OCSVM"
    elif args.model == 'lof':
        results = run_LOF(X, y, args.kfold, scenario_fn)
        out_dir = f"exports/Scenario{args.scenario}/LOF"
    elif args.model == 'ee':
        results = run_EllipticEnvelope(X, y, args.kfold, scenario_fn)
        out_dir = f"exports/Scenario{args.scenario}/EllipticEnvelope"
    elif args.model == 'svm':
        results = run_binary_svm(X, y, args.kfold, scenario_fn)
        out_dir = f"exports/Scenario{args.scenario}/SVM"
    elif args.model == 'knn':
        results = run_knn(X, y, args.kfold, scenario_fn)
        out_dir = f"exports/Scenario{args.scenario}/kNN"
    elif args.model == 'rf':
        results = run_random_forest(X, y, args.kfold, scenario_fn)
        out_dir = f"exports/Scenario{args.scenario}/RandomForest"
    else:
        print(f"Unknown model: {args.model}")
        return

    os.makedirs(out_dir, exist_ok=True)

    if args.scenario == '2' or args.scenario == '3':
        print(f"Running {args.model.upper()} on Scenario {args.scenario} ...")
        for fold_idx, attack_id, test_idx, y_pred, y_test in results:
            out_file = f"{out_dir}/Predictions_Fold{fold_idx+1}.csv"
            export_model_output(haiEnd_df, test_idx, y_pred, out_file)
    else:       
        print(f"Running {args.model.upper()} on Scenario {args.scenario} ...")
        for fold_idx, test_idx, y_pred, y_test in results:
            out_file = f"{out_dir}/Predictions_Fold{fold_idx+1}.csv"
            export_model_output(haiEnd_df, test_idx, y_pred, out_file)

def run_from_toolbox(model, scenario, k=5, export="1"):
    """
    Allows toolbox.py (Task 3) to run this module.
    """
    class Args:
        pass

    Args.model = model
    Args.scenario = scenario
    Args.kfold = k
    Args.export = export
    Args.output_dir = "exports"

    _internal_main(Args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=['ocsvm', 'lof', 'ee', 'knn', 'svm', 'rf'])
    parser.add_argument('-sc', '--scenario', choices=['1', '2', '3'])
    parser.add_argument('-k','--kfold', type=int, default=5)
    parser.add_argument('-e', '--export', choices=['1', '2', '3'])
    args = parser.parse_args()

    # Require either -e or -m
    if not args.export and not args.model:
        parser.error("Either -e/--export or -m/--model is required")

    _internal_main(args)


if __name__ == "__main__":
    main()

