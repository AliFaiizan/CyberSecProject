# task2_cnn_latent.py
import os
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, Activation, Dropout,
    Flatten, Dense, Input
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ------------------------------------------------------------
# Create sliding windows on latent features
# ------------------------------------------------------------
def create_windows(Z, y, M, stride=1):
    Xw, yw = [], []
    N = len(Z)

    for start in range(0, N - M + 1, stride):
        end = start + M
        Xw.append(Z[start:end])
        yw.append(1 if np.any(y[start:end] == 1) else 0)

    return np.array(Xw), np.array(yw)


# ------------------------------------------------------------
# CNN architecture (6 blocks)
# ------------------------------------------------------------
def build_cnn_6blocks(input_shape, lr=1e-3, dropout_rate=0.3):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # 4 Conv blocks, each with 2 Conv1D layers
    for filters in [32, 32, 64, 64]:
        for _ in range(2):
            model.add(Conv1D(filters, kernel_size=3, padding="same"))
            model.add(BatchNormalization())
            model.add(Activation("relu"))
            model.add(Dropout(dropout_rate))

    # Dense blocks
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(dropout_rate))

    # Output
    model.add(Dense(2, activation="softmax"))

    model.compile(
        optimizer=Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ------------------------------------------------------------
# Run CNN for each fold of scenario 2 or 3
# ------------------------------------------------------------
def run_cnn_latent(Z, y, scenario_fn, k, out_dir, M=20, epochs=10):
    os.makedirs(out_dir, exist_ok=True)

    print(f"[CNN] latent_dim={Z.shape[1]}, window_size M={M}")

    metrics = []

    for result in scenario_fn(Z, pd.Series(y), k):
        if len(result) == 3:
            fold_idx, train_idx, test_idx = result
            attack_id = None
        else:
            fold_idx, attack_id, train_idx, test_idx = result

        print(f"\n[CNN] === Fold {fold_idx+1} ===")

        Z_train, y_train = Z[train_idx], y[train_idx]
        Z_test, y_test = Z[test_idx], y[test_idx]

        X_train_w, y_train_w = create_windows(Z_train, y_train, M, stride=1)
        X_test_w, y_test_w = create_windows(Z_test, y_test, M, stride=1)

        if len(X_train_w) == 0 or len(X_test_w) == 0:
            print("Skipping fold — insufficient window data")
            continue

        # One-hot encode
        y_train_cat = to_categorical(y_train_w, 2)
        y_test_cat = to_categorical(y_test_w, 2)

        input_shape = (M, Z.shape[1])
        model = build_cnn_6blocks(input_shape)

        cb = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        model.fit(
            X_train_w, y_train_cat,
            validation_data=(X_test_w, y_test_cat),
            epochs=epochs,
            batch_size=128,
            callbacks=[cb],
            verbose=1
        )

        preds = np.argmax(model.predict(X_test_w, verbose=0), axis=1)

        # Save predictions
        df = pd.DataFrame({"predicted_label": preds, "Attack": y_test_w})
        df.to_csv(f"{out_dir}/Predictions_Fold{fold_idx+1}.csv", index=False)

        # Metrics
        tp = ((preds == 1) & (y_test_w == 1)).sum()
        fp = ((preds == 1) & (y_test_w == 0)).sum()
        fn = ((preds == 0) & (y_test_w == 1)).sum()

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        print(f"[CNN Fold {fold_idx+1}] Precision={precision:.4f}, Recall={recall:.4f}")

        metrics.append({
            "fold": fold_idx + 1,
            "precision": precision,
            "recall": recall,
            "attack_id": attack_id
        })

    pd.DataFrame(metrics).to_csv(f"{out_dir}/metrics_summary.csv", index=False)
    print("[CNN] Completed all folds")

