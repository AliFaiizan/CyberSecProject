#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from utils import load_and_clean_data
from task1 import (
    scenario_1_split,
    scenario_2_split,
    scenario_3_split,
)

# --------------------------------------------------------------------
# Create sliding windows
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
# Normalize data
# --------------------------------------------------------------------
def normalize_features(X_train, X_test):
    """Normalize features across training and test sets"""
    original_train_shape = X_train.shape
    original_test_shape = X_test.shape

    # Reshape to 2D for scaling
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_test_scaled = scaler.transform(X_test_2d)

    # Reshape back to original shape
    X_train_scaled = X_train_scaled.reshape(original_train_shape)
    X_test_scaled = X_test_scaled.reshape(original_test_shape)

    return X_train_scaled, X_test_scaled


# --------------------------------------------------------------------
# Build 6-block CNN (4 conv blocks + 2 dense blocks)
# --------------------------------------------------------------------
def build_cnn_6blocks(input_shape, dropout_rate=0.3, lr=1e-3):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Block 1
    model.add(Conv1D(32, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))

    # Block 2
    model.add(Conv1D(32, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))

    # Block 3
    model.add(Conv1D(64, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))

    # Block 4
    model.add(Conv1D(64, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))

    # Block 5
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(dropout_rate))

    # Block 6
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(dropout_rate))

    # Output
    model.add(Dense(2, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", Precision(name='precision'), Recall(name='recall')]
    )
    return model


# --------------------------------------------------------------------
# Main CNN function
# --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CNN Deep Learning Classifier")
    parser.add_argument("-sc", "--scenario", required=True, type=int, choices=[1, 2, 3])
    parser.add_argument("-M", required=True, type=int, help="Window size")
    parser.add_argument("-e", "--epochs", default=10, type=int, help="Training epochs")
    parser.add_argument("--stride", type=int, default=10, help="Window stride")

    args = parser.parse_args()
    sc = args.scenario
    M = args.M
    epochs = args.epochs
    stride = args.stride

    np.random.seed(42)

    print("\n=== Loading Data for CNN ===")
    merged = load_and_clean_data(
        ["../datasets/hai-22.04/train1.csv"],
        ["../datasets/hai-22.04/test1.csv"],
        attack_cols=None,
    )

    # Features: drop Attack + timestamp
    feature_cols = [c for c in merged.columns if c not in ("Attack", "timestamp")]
    X = merged[feature_cols].values
    y = merged["Attack"].astype(int).values

    # Choose scenario split function
    if sc == 1:
        scenario_fn = scenario_1_split
    elif sc == 2:
        scenario_fn = scenario_2_split
    else:
        scenario_fn = scenario_3_split

    out_dir = f"exports/Scenario{sc}/CNN"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Running CNN for Scenario {sc} | M={M} | Epochs={epochs} ===")

    # K-fold training
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

        # Create windows
        X_train_w, y_train_w = create_windows(X_train, y_train, M, stride=stride)
        X_test_w, y_test_w = create_windows(X_test, y_test, M, stride=1)

        if X_train_w.shape[0] == 0 or X_test_w.shape[0] == 0:
            print("Not enough samples to create windows. Skipping fold.")
            continue

        # Normalize features
        X_train_w, X_test_w = normalize_features(X_train_w, X_test_w)

        # One-hot labels
        y_train_cat = to_categorical(y_train_w, num_classes=2)
        y_test_cat = to_categorical(y_test_w, num_classes=2)

        # Class weights
        try:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train_w),
                y=y_train_w
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        except:
            class_weight_dict = {0: 1.0, 1: 1.0}

        # Build CNN
        n_features = X_train_w.shape[2]
        model = build_cnn_6blocks((M, n_features))

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
                min_delta=0.001,
                verbose=1,
            )
        ]

        print(f"Training samples: {len(X_train_w)}, Attack ratio: {(y_train_w == 1).mean():.4f}")
        print("Training 6-block CNN...")

        model.fit(
            X_train_w,
            y_train_cat,
            validation_data=(X_test_w, y_test_cat),
            epochs=epochs,
            batch_size=128,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
        )

        # Predict
        preds_prob = model.predict(X_test_w, verbose=0)
        preds = np.argmax(preds_prob, axis=1)

        # Export predictions
        df_out = pd.DataFrame({
            "predicted_label": preds,
            "Attack": y_test_w,
        })

        out_file = f"{out_dir}/Predictions_Fold{fold_idx + 1}.csv"
        df_out.to_csv(out_file, index=False)

        # Metrics
        accuracy = (preds == y_test_w).mean()
        tp = ((preds == 1) & (y_test_w == 1)).sum()
        fp = ((preds == 1) & (y_test_w == 0)).sum()
        fn = ((preds == 0) & (y_test_w == 1)).sum()

        precision = tp / max((tp + fp), 1)
        recall = tp / max((tp + fn), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        print(f"\n=== Fold {fold_idx + 1} Results ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"Rows of held-out attack rows:", (y_test == 1).sum())
        print(f"Windows of held-out attack windows:", (y_test_w == 1).sum())
        print(f"Attack windows detected: {tp}/{y_test_w.sum()}")
        print(f"Saved: {out_file}")

# -------------------------------------------------------------
# run_from_toolbox() — used by toolbox.py
# -------------------------------------------------------------
def run_from_toolbox(scenario, M, epochs, stride):
    import sys
    sys.argv = [
        "task2.py",
        "-sc", str(scenario),
        "-M", str(M),
        "-e", str(epochs),
        "--stride", str(stride),
    ]
    main()

if __name__ == "__main__":
    main()

