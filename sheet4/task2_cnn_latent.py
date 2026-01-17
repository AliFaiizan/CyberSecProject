#!/usr/bin/env python3
import os
import time
import psutil
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler

process = psutil.Process()

# ============================================================
# Sliding windows on LATENT FEATURES
# ============================================================
def create_windows(Z, y, M, stride=1):
    Xw, yw = [], []
    N = len(Z)

    for start in range(0, N - M + 1, stride):
        end = start + M
        Xw.append(Z[start:end])
        label = 1 if np.any(y[start:end] == 1) else 0
        yw.append(label)

    return np.array(Xw), np.array(yw, dtype=int)

# ============================================================
# Normalize windows
# ============================================================
def normalize_windows(X_train, X_test):
    scaler = StandardScaler()

    Xtr = X_train.reshape(-1, X_train.shape[-1])
    Xte = X_test.reshape(-1,  X_test.shape[-1])

    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    return (
        Xtr.reshape(X_train.shape),
        Xte.reshape(X_test.shape),
    )

# ============================================================
# SAME CNN from TaskSheet 2
# ============================================================
def build_cnn(input_shape, dropout_rate=0.3, lr=1e-3):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv1D(32, 3, padding="same")); model.add(BatchNormalization())
    model.add(Activation("relu")); model.add(Dropout(dropout_rate))

    model.add(Conv1D(32, 3, padding="same")); model.add(BatchNormalization())
    model.add(Activation("relu")); model.add(Dropout(dropout_rate))

    model.add(Conv1D(64, 3, padding="same")); model.add(BatchNormalization())
    model.add(Activation("relu")); model.add(Dropout(dropout_rate))

    model.add(Conv1D(64, 3, padding="same")); model.add(BatchNormalization())
    model.add(Activation("relu")); model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(128, activation="relu")); model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation="relu")); model.add(Dropout(dropout_rate))

    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ============================================================
# MAIN ENTRY FOR TASK 2 (NOW WITH FULL RUNTIME TRACKING)
# ============================================================
def run_cnn_latent(Z, y, scenario_id, k, out_dir, M=20):

    from scenarios import scenario_2_split, scenario_3_split
    scenario_fn = scenario_2_split if scenario_id == 2 else scenario_3_split

    results = []

    for res in scenario_fn(pd.DataFrame(Z), pd.Series(y)):

        fold_idx, attack_id, train_idx, test_idx = res
        print(f"\n[CNN] Fold {fold_idx+1}   Attack={attack_id}")

        start_total = time.time()

        Z_train, Z_test = Z[train_idx], Z[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # -------------------------------
        # 1) Window creation
        # -------------------------------
        t0 = time.time()
        X_train_w, y_train_w = create_windows(Z_train, y_train, M)
        X_test_w, y_test_w = create_windows(Z_test, y_test, M)
        t_window = time.time() - t0
        mem_window = process.memory_info().rss

        if len(X_train_w) == 0 or len(X_test_w) == 0:
            print("[CNN] Not enough windows — skipping fold.")
            continue

        # -------------------------------
        # 2) Normalization
        # -------------------------------
        t0 = time.time()
        X_train_w, X_test_w = normalize_windows(X_train_w, X_test_w)
        t_norm = time.time() - t0
        mem_norm = process.memory_info().rss

        # -------------------------------
        # 3) One-hot labels
        # -------------------------------
        y_train_cat = to_categorical(y_train_w, 2)
        y_test_cat = to_categorical(y_test_w, 2)

        # -------------------------------
        # 4) Build + Train CNN
        # -------------------------------
        model = build_cnn((M, Z.shape[1]))

        t0 = time.time()
        cb = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
        model.fit(
            X_train_w, y_train_cat,
            validation_data=(X_test_w, y_test_cat),
            epochs=10,
            batch_size=128,
            verbose=1,
            callbacks=cb
        )
        t_train = time.time() - t0
        mem_train = process.memory_info().rss

        # -------------------------------
        # 5) Prediction runtime
        # -------------------------------
        t0 = time.time()
        preds = np.argmax(model.predict(X_test_w, verbose=0), axis=1)
        t_pred = time.time() - t0
        mem_pred = process.memory_info().rss

        # -------------------------------
        # 6) Total runtime
        # -------------------------------
        total_runtime = time.time() - start_total
        total_memory = process.memory_info().rss

        # Save window-based predictions
        pd.DataFrame({
            "predicted_label": preds,
            "Attack": y_test_w
        }).to_csv(f"{out_dir}/Predictions_Fold{fold_idx+1}.csv", index=False)

        # Return extended metrics to task2.py
        results.append(
            (
                fold_idx,
                model,
                X_test_w,
                y_test_w,
                total_runtime,
                total_memory,
                test_idx,
                {
                    "window_time": t_window,
                    "window_mem": mem_window,
                    "norm_time": t_norm,
                    "norm_mem": mem_norm,
                    "train_time": t_train,
                    "train_mem": mem_train,
                    "pred_time": t_pred,
                    "pred_mem": mem_pred,
                }
            )
        )

    return results
