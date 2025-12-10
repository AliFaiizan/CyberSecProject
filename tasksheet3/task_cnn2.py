# ================================================================
# CNN CLASSIFIER UPDATED FOR LATENT FEATURES (Task Sheet 3)
# ================================================================

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, Activation, Dropout,
    Flatten, Dense, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# -------------------------------------------------------
# Create sliding windows on LATENT FEATURES
# -------------------------------------------------------
def create_windows_latent(Z, y, M, stride=1):
    """
    Z: latent vectors (N, latent_dim)
    y: attack labels (N,)
    M: window size
    stride: window step size
    
    Label rule:
      window_label = 1 if ANY attack is inside the window, else 0
    """
    Xw, yw = [], []
    N = len(Z)

    if N < M:
        return np.empty((0, M, Z.shape[1])), np.empty((0,), dtype=int)

    for start in range(0, N - M + 1, stride):
        end = start + M
        Xw.append(Z[start:end])
        window_labels = y[start:end]
        label = 1 if np.any(window_labels == 1) else 0
        yw.append(label)

    return np.array(Xw), np.array(yw, dtype=int)


# -------------------------------------------------------
# Normalize latent windows (same as before)
# -------------------------------------------------------
def normalize_latent_windows(X_train_w, X_test_w):
    shape_train = X_train_w.shape
    shape_test = X_test_w.shape

    X_train_flat = X_train_w.reshape(-1, shape_train[-1])
    X_test_flat  = X_test_w.reshape(-1, shape_test[-1])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled  = scaler.transform(X_test_flat)

    X_train_scaled = X_train_scaled.reshape(shape_train)
    X_test_scaled  = X_test_scaled.reshape(shape_test)

    return X_train_scaled, X_test_scaled


# -------------------------------------------------------
# Build 6-block CNN (NO CHANGES TO ARCHITECTURE)
# -------------------------------------------------------
def build_cnn_latent(input_shape, dropout_rate=0.3, lr=1e-3):
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

    # Dense blocks
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(dropout_rate))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(dropout_rate))

    model.add(Dense(2, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -------------------------------------------------------
# MASTER FUNCTION FOR CNN USING LATENT FEATURES
# -------------------------------------------------------
def run_cnn_latent(Z_train, y_train, Z_test, y_test, M, stride, epochs, out_file):
    """
    Z_train/Z_test: latent vectors from VAE (N_train, latent_dim)
    y_train/y_test: attack labels
    """

    latent_dim = Z_train.shape[1]

    # ---- Create windows ----
    X_train_w, y_train_w = create_windows_latent(Z_train, y_train, M, stride)
    X_test_w,  y_test_w  = create_windows_latent(Z_test,  y_test,  M, stride=1)

    if len(X_train_w) == 0 or len(X_test_w) == 0:
        print("[CNN] Not enough samples for windows — skipping.")
        return None

    # ---- Normalize ----
    X_train_w, X_test_w = normalize_latent_windows(X_train_w, X_test_w)

    # ---- One-hot labels ----
    y_train_cat = to_categorical(y_train_w, num_classes=2)
    y_test_cat = to_categorical(y_test_w, num_classes=2)

    # ---- Class weights ----
    try:
        cw = compute_class_weight("balanced", np.unique(y_train_w), y_train_w)
        class_weight = {i: cw[i] for i in range(2)}
    except:
        class_weight = {0: 1.0, 1: 1.0}

    # ---- Build CNN ----
    model = build_cnn_latent((M, latent_dim))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    # ---- Train CNN ----
    model.fit(
        X_train_w,
        y_train_cat,
        validation_data=(X_test_w, y_test_cat),
        epochs=epochs,
        batch_size=128,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # ---- Predict ----
    preds_prob = model.predict(X_test_w, verbose=0)
    preds = np.argmax(preds_prob, axis=1)

    # ---- Metrics ----
    accuracy = (preds == y_test_w).mean()
    tp = ((preds == 1) & (y_test_w == 1)).sum()
    fp = ((preds == 1) & (y_test_w == 0)).sum()
    fn = ((preds == 0) & (y_test_w == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print("\n=== CNN RESULTS ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # ---- Save predictions ----
    df = pd.DataFrame({
        "predicted_label": preds,
        "Attack": y_test_w,
    })
    df.to_csv(out_file, index=False)

    print(f"[CNN] Saved predictions → {out_file}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
