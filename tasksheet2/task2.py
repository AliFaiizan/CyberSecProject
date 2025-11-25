#!/usr/bin/env python3
import argparse
import os
from pyexpat import model
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from utils import load_and_clean_data
from task1 import (
    scenario_1_split,
    scenario_2_split,
    scenario_3_split,
)

# --------------------------------------------------------------------
# Create sliding windows (window label = 1 if ANY attack in window)
# --------------------------------------------------------------------
def create_windows(X, y, M, stride=1):
    Xw, yw = [], []
    N = len(X)
    if N < M:
        return np.empty((0, M, X.shape[1])), np.empty((0,), dtype=int)

    for start in range(0, N - M + 1, stride):
        end = start + M
        Xw.append(X[start:end])
        window_labels = y[start:end]
        label = 1 if np.any(window_labels == 1) else 0
        yw.append(label)

    return np.array(Xw), np.array(yw, dtype=int)


# --------------------------------------------------------------------
# Balance windows (oversample minority class)
# --------------------------------------------------------------------
def balance_windows(Xw, yw):
    """
    Oversample the minority class to get roughly balanced data.
    Xw: (Nw, M, n_features)
    yw: (Nw,)
    """
    if len(yw) == 0:
        return Xw, yw

    idx_pos = np.where(yw == 1)[0]
    idx_neg = np.where(yw == 0)[0]

    if len(idx_pos) == 0 or len(idx_neg) == 0:
        # Only one class present → nothing to balance
        return Xw, yw

    # Oversample minority to match majority
    if len(idx_pos) < len(idx_neg):
        idx_pos_over = np.random.choice(idx_pos, size=len(idx_neg), replace=True)
        idx_bal = np.concatenate([idx_neg, idx_pos_over])
    else:
        idx_neg_over = np.random.choice(idx_neg, size=len(idx_pos), replace=True)
        idx_bal = np.concatenate([idx_pos, idx_neg_over])

    np.random.shuffle(idx_bal)
    return Xw[idx_bal], yw[idx_bal]


# --------------------------------------------------------------------
# Build the required 6-block CNN
#   4 conv blocks (each: 2×[Conv+BN+ReLU+Dropout])
#   then 2 fully-connected layers → softmax(2)
# --------------------------------------------------------------------
def build_cnn(input_shape, dropout_rate=0.4, lr=1e-4):
    model = Sequential()

    for block_idx in range(4):
        # First conv in block
        if block_idx == 0:
            model.add(Conv1D(64, 3, padding="same", input_shape=input_shape))
        else:
            model.add(Conv1D(64, 3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(dropout_rate))

        # Second conv in block
        model.add(Conv1D(64, 3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(dropout_rate))

    # Dense blocks
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(dropout_rate))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(dropout_rate))

    # Output (2 classes: normal / attack)
    model.add(Dense(2, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def _internal_main(args):
    """
    Internal main function that handles the CNN training logic.
    Expects args object with: scenario, M, epochs, stride
    """
    sc = args.scenario
    M = args.M
    epochs = args.epochs
    stride = args.stride

    np.random.seed(42)

    print("\n=== Loading & Cleaning Data ===")
    merged = load_and_clean_data(
        ["../datasets/hai-22.04/train1.csv"],
        ["../datasets/hai-22.04/test1.csv"],
        attack_cols=None,
    )

    # Features: drop Attack + timestamp
    feature_cols = [c for c in merged.columns if c not in ("Attack", "timestamp")]
    X = merged[feature_cols].values
    y = merged["Attack"].astype(int).values  # numpy array

    # Choose scenario split function
    if sc == 1:
        scenario_fn = scenario_1_split
    elif sc == 2:
        scenario_fn = scenario_2_split
    else:
        scenario_fn = scenario_3_split

    out_dir = f"exports/Scenario{sc}/CNN"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Running CNN for Scenario {sc} ===")

    # k-fold loops (pass y as pandas.Series to avoid numpy .shift() error)
    for result in scenario_fn(X, pd.Series(y)):
        if sc == 1:
            fold_idx, train_idx, test_idx = result
            print(f"\n--- Fold {fold_idx + 1} ---")
        else:
            fold_idx, attack_id, train_idx, test_idx = result
            print(f"\n--- Fold {fold_idx + 1} (AttackID={attack_id}) ---")

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # Create windows (train: configurable stride, test: stride=1 for full resolution)
        X_train_w, y_train_w = create_windows(X_train, y_train, M, stride=stride)
        X_test_w, y_test_w = create_windows(X_test, y_test, M, stride=1)

        if X_train_w.shape[0] == 0 or X_test_w.shape[0] == 0:
            print("Not enough samples to create windows. Skipping fold.")
            continue

        # Balance training windows
        X_train_w, y_train_w = balance_windows(X_train_w, y_train_w)

        # One-hot labels
        y_train_cat = to_categorical(y_train_w, num_classes=2)
        y_test_cat = to_categorical(y_test_w, num_classes=2)

        # Build CNN
        n_features = X_train_w.shape[2]
        model = build_cnn((M, n_features))

        # Early stopping to avoid heavy overfitting
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
                verbose=1,
            )
        ]

        print("Training CNN...")
        model.fit(
            X_train_w,
            y_train_cat,
            validation_data=(X_test_w, y_test_cat),
            epochs=epochs,
            batch_size=256,
            callbacks=callbacks,
            verbose=1,
        )

        # Predict on test windows
        preds_prob = model.predict(X_test_w, verbose=0)
        preds = np.argmax(preds_prob, axis=1)

        # Export in Task 1 style (window-level labels)
        df_out = pd.DataFrame(
            {
                "predicted_label": preds,
                "Attack": y_test_w,
            }
        )

        out_file = f"{out_dir}/Predictions_Fold{fold_idx + 1}.csv"
        df_out.to_csv(out_file, index=False)

        # Summary (window-level)
        detected_attacks = (preds == 1).sum()
        normal_windows = (preds == 0).sum()
        total = len(preds)
        acc = (preds == y_test_w).mean()

        print(f"\nSaved: {out_file}")
        print(f"Total windows: {total}")
        print(f"Detected attacks: {detected_attacks}")
        print(f"Normal windows: {normal_windows}")
        print(f"Window accuracy: {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="CNN Deep Learning Classifier (Task 2)")
    parser.add_argument(
        "-sc",
        "--scenario",
        required=True,
        type=int,
        choices=[1, 2, 3],
        help="Scenario number 1 | 2 | 3",
    )
    parser.add_argument(
        "-M",
        required=True,
        type=int,
        help="Window size M (rows per CNN input)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=5,
        type=int,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Stride for training windows (default=10, use 1 for max overlap)",
    )
    args = parser.parse_args()

    _internal_main(args)


def run_from_toolbox(scenario, M, epochs=5, stride=10):
    """
    Allows toolbox.py (Task 3) to run this module.
    
    Args:
        scenario: Scenario number (1, 2, or 3)
        M: Window size (rows per CNN input)
        epochs: Maximum training epochs (default: 5)
        stride: Stride for training windows (default: 10)
    """
    class Args:
        pass

    Args.scenario = scenario
    Args.M = M
    Args.epochs = epochs
    Args.stride = stride

    _internal_main(Args)

if __name__ == "__main__":
    main()
