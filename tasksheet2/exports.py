import os
import pandas as pd

def export_scenario_1(merged_df, X, y, scenario_fn, out_dir="exports/Scenario1"):
    """
    Exports train/test CSV files for Scenario 1 (one-class models).
    k=5 folds → generate 10 files.
    """
    os.makedirs(out_dir, exist_ok=True)

    print("\n=== Exporting Scenario 1 Splits ===")

    for fold_idx, train_idx, test_idx in scenario_fn(X, y):

        train_set = merged_df.iloc[train_idx]
        test_set  = merged_df.iloc[test_idx]

        train_file = f"{out_dir}/Train_Fold{fold_idx+1}.csv"
        test_file  = f"{out_dir}/Test_Fold{fold_idx+1}.csv"

        train_set.to_csv(train_file, index=False)
        test_set.to_csv(test_file, index=False)

        print(f"Saved: {train_file}  |  {test_file}")


def export_scenario_2(merged_df, X, y, scenario_fn,
                      out_dir="exports/Scenario2"):
    """
    Scenario 2:
        - Hold ONE attack type out of training
        - Train on normal + (n-1) attack types
        - Test on normal fold + ALL attack types
    
    Note: scenario_fn selects one random attack_id to hold out and uses it for all folds.
    """

    print("\n=== Exporting Scenario 2 Splits ===")
    os.makedirs(out_dir, exist_ok=True)

    for fold_idx, held_out, train_idx, test_idx in scenario_fn(X, y):

        train_set = merged_df.iloc[train_idx]
        test_set  = merged_df.iloc[test_idx]

        train_file = f"{out_dir}/Train_Fold{fold_idx+1}.csv"
        test_file  = f"{out_dir}/Test_Fold{fold_idx+1}.csv"

        train_set.to_csv(train_file, index=False)
        test_set.to_csv(test_file, index=False)

        print(f"Fold {fold_idx+1} [HeldOut={held_out}] Saved: {train_file} | {test_file}")


def export_scenario_3(merged_df, X, y, scenario_fn,
                      out_dir="exports/Scenario3"):
    """
    Scenario 3:
        - Train on normal + EXACTLY ONE attack type
        - Test on normal fold + ALL attack types
    
    Note: scenario_fn selects one random attack_id to train on and uses it for all folds.
    """

    print("\n=== Exporting Scenario 3 Splits ===")
    os.makedirs(out_dir, exist_ok=True)

    for fold_idx, selected_type, train_idx, test_idx in scenario_fn(X, y):

        train_set = merged_df.iloc[train_idx]
        test_set  = merged_df.iloc[test_idx]

        train_file = f"{out_dir}/Train_Fold{fold_idx+1}.csv"
        test_file  = f"{out_dir}/Test_Fold{fold_idx+1}.csv"

        train_set.to_csv(train_file, index=False)
        test_set.to_csv(test_file, index=False)

        print(f"Fold {fold_idx+1} [TrainOn={selected_type}] Saved: {train_file} | {test_file}")


def export_model_output(merged_df, test_idx, y_pred, out_file):
    """
    Exports model predictions to CSV.
    Marks each row in test_idx with predicted label and attack status.
    Also prints a summary of predictions.
    """
    output_df = merged_df.iloc[test_idx].copy()
    output_df['predicted_label'] = y_pred
    output_df['attack_status'] = output_df['predicted_label'].apply(lambda x: 'ATTACK' if x == 1 else 'NORMAL')
    output_df.to_csv(out_file, index=False)
    print(f"Model output exported to: {out_file}")
    
    # Print summary statistics
    print("\n=== Prediction Summary ===")
    print(f"Total test samples: {len(y_pred)}")
    print(f"Detected attacks: {(y_pred == 1).sum()}")
    print(f"Normal samples: {(y_pred == 0).sum()}\n")