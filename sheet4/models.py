import numpy as np
from sklearn import svm
from sklearn.svm import OneClassSVM, SVC
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# cnn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

import time
import psutil
process = psutil.Process()
# =========================================================
# PER-FOLD CLASSIFIERS WITH NORMALIZATION
# =========================================================
def run_OneClassSVM_per_fold(Z_train, y_train, Z_test, y_test, params):
    """Train OCSVM with feature normalization"""
    
    # CRITICAL FIX: Normalize features to handle distribution mismatch
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    # Better hyperparameters
    model = OneClassSVM(**params)
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm)
    mem_after = process.memory_info().rss
    
    y_pred_raw = model.predict(Z_test_norm)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def run_LOF_per_fold(Z_train, y_train, Z_test, y_test,params):
    """Train LOF with feature normalization"""
    
    # Normalize features
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    # Better hyperparameters
    model = LocalOutlierFactor(**params, novelty=True)
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm)
    mem_after = process.memory_info().rss
    
    y_pred_raw = model.predict(Z_test_norm)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def run_EllipticEnvelope_per_fold(Z_train, y_train, Z_test, y_test,params):
    """Train EllipticEnvelope with feature normalization"""
    
    # Normalize features
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    # Better hyperparameters
    model = EllipticEnvelope(**params)
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm)
    mem_after = process.memory_info().rss
    
    y_pred_raw = model.predict(Z_test_norm)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def run_binary_svm_per_fold(Z_train, y_train, Z_test, y_test,params):
    """Train SVM with feature normalization"""
    
    # Normalize features
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    model = SVC(**params)
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm, y_train)
    mem_after = process.memory_info().rss
    
    y_pred = model.predict(Z_test_norm)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def run_knn_per_fold(Z_train, y_train, Z_test, y_test,params):
    """Train kNN with feature normalization"""
    
    # Normalize features
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    model = KNeighborsClassifier(**params)
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm, y_train)
    mem_after = process.memory_info().rss
    
    y_pred = model.predict(Z_test_norm)
    
    return y_pred, model, time.time() - start, mem_after - mem_before


def run_random_forest_per_fold(Z_train, y_train, Z_test, y_test, params):
    """Train RandomForest with feature normalization"""
    
    # Normalize features
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)
    
    model = RandomForestClassifier(**params)
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm, y_train)
    mem_after = process.memory_info().rss
    
    y_pred = model.predict(Z_test_norm)
    
    return y_pred, model, time.time() - start, mem_after - mem_before

def build_cnn_model(input_shape, params=None):
    """
    params can include:
      - dropout_rate
      - lr
      - conv_filters: tuple/list like (32, 64, 128, 256)
      - kernel_size
      - dense1_units
      - dense2_units
    """
    if params is None:
        params = {}

    dropout_rate = params.get("dropout_rate", 0.3)
    lr = params.get("lr", 1e-3)
    conv_filters = params.get("conv_filters", (32, 64, 128, 256))
    kernel_size = params.get("kernel_size", 3)
    dense1_units = params.get("dense1_units", 128)
    dense2_units = params.get("dense2_units", 64)

    model = Sequential()
    model.add(Input(shape=input_shape))

    # ===== 4 Conv blocks, each block has 2 conv layers =====
    # Block 1
    model.add(Conv1D(conv_filters[0], kernel_size, padding="same"))
    model.add(BatchNormalization()); model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(conv_filters[0], kernel_size, padding="same"))
    model.add(BatchNormalization()); model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))

    # Block 2
    model.add(Conv1D(conv_filters[1], kernel_size, padding="same"))
    model.add(BatchNormalization()); model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(conv_filters[1], kernel_size, padding="same"))
    model.add(BatchNormalization()); model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))

    # Block 3
    model.add(Conv1D(conv_filters[2], kernel_size, padding="same"))
    model.add(BatchNormalization()); model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(conv_filters[2], kernel_size, padding="same"))
    model.add(BatchNormalization()); model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))

    # Block 4
    model.add(Conv1D(conv_filters[3], kernel_size, padding="same"))
    model.add(BatchNormalization()); model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(conv_filters[3], kernel_size, padding="same"))
    model.add(BatchNormalization()); model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())

    # ===== 2 FC layers =====
    model.add(Dense(dense1_units, activation="relu"))
    model.add(Dropout(dropout_rate))

    model.add(Dense(dense2_units, activation="relu"))
    model.add(Dropout(dropout_rate))

    model.add(Dense(2, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def run_cnn_per_fold(Z_train, y_train, Z_test, y_test, params=None):
    """Train CNN on latent features with tunable hyperparameters."""
    if params is None:
        params = {}

    tf.keras.backend.clear_session()  # important when looping many trials

    start = time.time()
    mem_before = process.memory_info().rss

    # Normalize
    scaler = StandardScaler()
    Z_train_norm = scaler.fit_transform(Z_train)
    Z_test_norm = scaler.transform(Z_test)

    # Reshape for CNN: (samples, timesteps, channels)
    n_features = Z_train_norm.shape[1]
    Z_train_cnn = Z_train_norm.reshape(len(Z_train_norm), n_features, 1)
    Z_test_cnn = Z_test_norm.reshape(len(Z_test_norm), n_features, 1)

    # One-hot labels
    y_train_cat = to_categorical(y_train, 2)
    y_test_cat = to_categorical(y_test, 2)

    # Class weights (optional)
    use_class_weight = params.get("use_class_weight", True)
    if use_class_weight:
        n_normal = np.sum(y_train == 0)
        n_attack = np.sum(y_train == 1)
        total = len(y_train)
        if n_attack > 0 and n_normal > 0:
            class_weight = {
                0: total / (2 * n_normal),
                1: total / (2 * n_attack)
            }
        else:
            class_weight = None
    else:
        class_weight = None

    # Tunables
    batch_size = params.get("batch_size", 128)
    epochs = params.get("epochs", 10)
    patience = params.get("patience", 5)

    # Build model with params
    input_shape = (n_features, 1)
    model = build_cnn_model(input_shape, params=params)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        )
    ]

    model.fit(
        Z_train_cnn, y_train_cat,
        validation_data=(Z_test_cnn, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        verbose=0,
        callbacks=callbacks,
    )

    mem_after = process.memory_info().rss

    y_pred_proba = model.predict(Z_test_cnn, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    return y_pred, model, time.time() - start, mem_after - mem_before
