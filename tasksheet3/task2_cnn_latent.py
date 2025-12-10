import os
import time
import psutil
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

from sklearn.preprocessing import StandardScaler
from scenarios import scenario_2_split, scenario_3_split

process = psutil.Process()


# =====================================================================
# Sliding window generation
# =====================================================================
def create_windows(Z, y, M, stride=1):
    Xw, yw = [], []
    N = len(Z)
    for start in range(0, N - M + 1, stride):
        end = start + M
        center = start + M // 2
        Xw.append(Z[start:end])
        yw.append(y[center])
    return np.array(Xw), np.array(yw)


# =====================================================================
# CNN architecture
# =====================================================================
def build_cnn(input_shape, lr=1e-3, dropout_rate=0.3):
    model = Sequential()
    model.add(Input(shape=input_shape))

    for filters in [32, 32, 64, 64]:
        for _ in range(2):
            model.add(Conv1D(filters, kernel_size=3, padding="same"))
            model.add(BatchNormalization())
            model.add(Activation("relu"))
            model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# =====================================================================
# Main CNN routine
# =====================================================================
def run_cnn_latent(Z, y, scenario_id, k, out_dir, M=20, epochs=5):
    os.makedirs(out_dir, exist_ok=True)

    if scenario_id == 2:
        scenario_fn = scenario_2_split
    elif scenario_id == 3:
        scenario_fn = scenario_3_split
    else:
        print("CNN disabled for Scenario 1.")
        return []

    results = []

    for fold_idx, attack_id, train_idx, test_idx in scenario_fn(Z, pd.Series(y), k):
        print(f"\n[CNN] Fold {fold_idx+1} (Attack {attack_id})")

        Z_train, y_train = Z[train_idx], y[train_idx]
        Z_test, y_test   = Z[test_idx], y[test_idx]

        # Standard scaling
        start = time.time()
        mem_before = process.memory_info().rss
        scaler = StandardScaler()
        Z_train = scaler.fit_transform(Z_train)
        Z_test  = scaler.transform(Z_test)
        fe1_time = time.time() - start
        fe1_mem  = process.memory_info().rss - mem_before

        # Window generation
        start = time.time()
        mem_before = process.memory_info().rss
        X_train_w, y_train_w = create_windows(Z_train, y_train, M)
        X_test_w,  y_test_w  = create_windows(Z_test,  y_test,  M)
        fe2_time = time.time() - start
        fe2_mem  = process.memory_info().rss - mem_before

        if len(X_train_w) == 0 or len(X_test_w) == 0:
            print("Skipping fold: empty window set.")
            continue

        y_train_cat = to_categorical(y_train_w, 2)
        y_test_cat  = to_categorical(y_test_w, 2)

        model = build_cnn((M, Z.shape[1]))
        early = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        # CNN training
        start = time.time()
        mem_before = process.memory_info().rss

        model.fit(
            X_train_w, y_train_cat,
            validation_data=(X_test_w, y_test_cat),
            epochs=epochs,
            batch_size=128,
            callbacks=[early],
            verbose=1
        )

        clf_time = time.time() - start
        clf_mem  = process.memory_info().rss - mem_before

        # Save model and windows
        model.save(f"{out_dir}/CNN_Fold{fold_idx+1}.h5")
        np.save(f"{out_dir}/X_test_windows_Fold{fold_idx+1}.npy", X_test_w)
        np.save(f"{out_dir}/y_test_windows_Fold{fold_idx+1}.npy", y_test_w)

        results.append(
            (fold_idx, model, X_test_w, y_test_w,
             fe1_time + fe2_time, fe1_mem + fe2_mem,
             clf_time, clf_mem)
        )

    return results