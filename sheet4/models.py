import numpy as np
from sklearn import svm
from sklearn.svm import OneClassSVM, SVC
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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
    model = OneClassSVM(kernel="rbf", nu=0.001, gamma='auto')
    
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
    model = LocalOutlierFactor(n_neighbors=20, metric='euclidean', novelty=True)
    
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
    model = EllipticEnvelope(contamination=0.01, random_state=42)
    
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
    
    model = SVC(kernel="rbf", C=10.0, gamma='scale')
    
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
    
    model = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
    
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
    
    model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=5, 
                                   random_state=42, n_jobs=-1)
    
    start = time.time()
    mem_before = process.memory_info().rss
    model.fit(Z_train_norm, y_train)
    mem_after = process.memory_info().rss
    
    y_pred = model.predict(Z_test_norm)
    
    return y_pred, model, time.time() - start, mem_after - mem_before
