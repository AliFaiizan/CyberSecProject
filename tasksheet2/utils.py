from typing import List, Tuple
import pandas as pd

def load_and_clean_data(train_files: List[str], test_files: List[str], attack_cols: List[str]=None, label_files: List[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load & Clean Data
    - Read all train CSVs for HAIend 23.05
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
        
        # # Remove attack rows (keep only normal data for training)
        # if attack_cols and all(col in df.columns for col in attack_cols):
        #     normal_mask = df[attack_cols] == 0
        #     df = df[normal_mask]
        #     print(f"  After removing attacks from normal: {df.shape}")
        
        train_dfs.append(df)
    
    # Load test data
    test_dfs = []
    for file in test_files:
        print(f"Loading {file}...")
        df = pd.read_csv(file)
        print(f"  Original shape: {df.shape}")
        
        # # Check if attack columns exist in dataframe
        # if attack_cols:  # This checks if attack_cols is not None and not empty
        #     if all(col in df.columns for col in attack_cols):
        #         # Count rows with Attack == 1
        #         attack_count = (df[attack_cols] == 1).any(axis=1).sum()
        #         print(f"  Number of attack rows: {attack_count}")
        #         # Remove attack samples (keep only rows where all attack columns are 0)
        #         df = df[(df[attack_cols] == 0).all(axis=1)]
        #         print(f"  After removing attacks: {df.shape}")
        # else:
        #     print("  No attack columns provided, skipping attack removal")
        
        test_dfs.append(df)
    

    # Combine all data
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Load and merge labels with test data BEFORE dropping timestamp
    if label_files is not None:
        label_dfs = []
        for label_file in label_files:
            print(f"Loading labels from {label_file}...")
            label_df = pd.read_csv(label_file)
            label_dfs.append(label_df)
        
        labels = pd.concat(label_dfs, ignore_index=True)
        
        
        # Merge on 'timestamp'
        if 'Timestamp' in test_df.columns:
            test_df = test_df.rename(columns={'Timestamp': 'timestamp'})
        test_df = pd.merge(test_df, labels, on='timestamp', how='left')
        print(f"  Labels merged. Test shape: {test_df.shape}")
        
                # Count attack rows in labels
        if 'label' in labels.columns:
            attack_count = (test_df['label'] == 1).sum()
            print(f"  Attack rows in labels: {attack_count}")
        # # Remove attack rows using label column
        # if 'label' in test_df.columns:
        #     test_df = test_df[test_df['label'] == 0]
        #     print(f"  After removing attack rows: {test_df.shape}")
    # Drop timestamp and attack columns
    cols_to_drop = ['Timestamp', 'timestamp'] # normal df has Timestamp , attack has timestamp

    # if attack_cols:
    #     # Only add attack columns that exist in the DataFrame
    #     cols_to_drop += [col for col in attack_cols if col in train_df.columns]
        
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"Final training data shape: {train_df.shape}")
    print(f"Final test data shape: {test_df.shape}") 
    
    # # Handle NaN values
    # train_df = train_df.fillna(method='ffill').fillna(0)
    # test_df = test_df.fillna(method='ffill').fillna(0)
    merged_dataset = pd.concat([train_df, test_df], ignore_index=True)
    
    merged_dataset['label'] = merged_dataset['label'].fillna(0) # Fill missing labels with 0 (normal)
    attack_count = (merged_dataset['label'] == 1).sum()
    print(f"Total attack rows in merged dataset: {attack_count}")
    return merged_dataset