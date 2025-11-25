#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from utils import load_and_clean_data                          # SAME as Task 1 ✔
from task1 import (                                            # SAME splitters ✔
    scenario_1_split,
    scenario_2_split,
    scenario_3_split
)

# ------------------------------------------------------------
# Sliding window generator (unchanged)
# ------------------------------------------------------------
def create_windows(X, y, M):
    Xw, yw = [], []
    for i in range(len(X) - M):
        Xw.append(X[i:i+M])
        yw.append(y[i+M])
    return np.array(Xw), np.array(yw)

# ------------------------------------------------------------
# CNN architecture (6 blocks required by assignment)
# ------------------------------------------------------------
def build_cnn(input_shape, dropout_rate=0.25, lr=0.001):
    model = Sequential()

    # 4 convolutional blocks, each having 2×Conv + BN + ReLU + Dropout
    for _ in range(4):
        model.add(Conv1D(64, 3, padding="same", input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(dropout_rate))

        model.add(Conv1D(64, 3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(dropout_rate))

    # Dense layers
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(dropout_rate))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(dropout_rate))

    # Output layer: 2 classes (normal=0, attack=1)
    model.add(Dense(2, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CNN classifier")
    parser.add_argument("-sc", "--scenario", type=int, required=True, choices=[1,2,3])
    parser.add_argument("-M", type=int, required=True, help="Window size (rows per sample)")
    parser.add_argument("-e", "--epochs", type=int, default=5)
    args = parser.parse_args()

    sc = args.scenario
    M = args.M
    epochs = args.epochs

    print("\n=== Loading & Cleaning Data (SAME as Task 1) ===")
    merged = load_and_clean_data(
        ["../datasets/hai-22.04/train1.csv"],
        ["../datasets/hai-22.04/test1.csv"]
    )  # EXACTLY same as Task 1 ✔

    # Build X, y EXACTLY like Task 1
    X = merged.drop(columns=["timestamp", "Attack"], errors="ignore").values
    y = merged["Attack"].astype(int)

    # Choose the same scenario splitter used in Task 1
    if sc == 1:
        scenario_fn = scenario_1_split
    elif sc == 2:
        scenario_fn = scenario_2_split
    else:
        scenario_fn = scenario_3_split

    out_dir = f"exports/Scenario{sc}/CNN"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Running CNN for Scenario {sc} ===")

    for result in scenario_fn(X, y):
        if sc == 1:
            fold_idx, train_idx, test_idx = result
        else:
            fold_idx, attack_id, train_idx, test_idx = result

        print(f"\n--- Fold {fold_idx+1} ---")

        # Extract splits EXACTLY like Task 1
        X_train = X[train_idx]
        y_train = y.iloc[train_idx].reset_index(drop=True)

        X_test = X[test_idx]
        y_test = y.iloc[test_idx].reset_index(drop=True)

        # Create sliding windows
        X_train_w, y_train_w = create_windows(X_train, y_train, M)
        X_test_w, y_test_w = create_windows(X_test, y_test, M)

        # Convert labels to one-hot
        y_train_cat = to_categorical(y_train_w, num_classes=2)
        y_test_cat = to_categorical(y_test_w, num_classes=2)

        # Build and train CNN
        model = build_cnn((M, X_train_w.shape[2]))
        model.fit(
            X_train_w, y_train_cat,
            validation_data=(X_test_w, y_test_cat),
            epochs=epochs,
            batch_size=64,
            verbose=1
        )

        # Predict
        preds_prob = model.predict(X_test_w)
        preds = np.argmax(preds_prob, axis=1)

        # Build Task 1 compatible output
        df_out = pd.DataFrame({
            "predicted_label": preds,
            "Attack": y_test_w
        })

        out_file = f"{out_dir}/Predictions_Fold{fold_idx+1}.csv"
        df_out.to_csv(out_file, index=False)

        print(f"\nSaved: {out_file}")
        print("Detected attacks:", (preds == 1).sum())
        print("Normal samples:", (preds == 0).sum())

if __name__ == "__main__":
    main()
