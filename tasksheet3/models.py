import time
import psutil
import numpy as np
from sklearn.svm import OneClassSVM, SVC
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from utils2 import optimal_param_search

process = psutil.Process()


# =====================================================================
# Utility timing wrappers
# =====================================================================
def measure_feature_step(func):
    start = time.time()
    mem_before = process.memory_info().rss
    out = func()
    mem_after = process.memory_info().rss
    return out, (time.time() - start), (mem_after - mem_before)


def measure_classification_step(func):
    start = time.time()
    mem_before = process.memory_info().rss
    out = func()
    mem_after = process.memory_info().rss
    return out, (time.time() - start), (mem_after - mem_before)


# =====================================================================
# One-Class SVM (Scenario 1)
# =====================================================================
def run_OneClassSVM(X, y, k, scenario_fn):

    param_grid = {'nu': [0.001], 'gamma': ['scale']}
    def build_model(params): return OneClassSVM(kernel="rbf", **params)

    best_params, _ = optimal_param_search(X, y, scenario_fn, build_model, param_grid)
    print("Best OCSVM params:", best_params)

    pca = PCA(n_components=0.95)
    all_results = []

    for fold_idx, train_idx, test_idx in scenario_fn(X, y, k):

        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_test  = y.iloc[test_idx].values

        # PCA feature extraction
        (pca_out, fe_time, fe_mem) = measure_feature_step(
            lambda: (pca.fit_transform(X_train), pca.transform(X_test))
        )
        X_train_red, X_test_red = pca_out

        # Classification
        model = OneClassSVM(kernel="rbf", **best_params)
        (_, clf_time, clf_mem) = measure_classification_step(
            lambda: model.fit(X_train_red)
        )

        y_pred_raw = model.predict(X_test_red)
        y_pred = np.where(y_pred_raw == -1, 1, 0)

        all_results.append(
            (fold_idx, test_idx, y_pred, y_test, model,
             fe_time, fe_mem, clf_time, clf_mem)
        )

    return all_results


# =====================================================================
# Elliptic Envelope (Scenario 1)
# =====================================================================
def run_EllipticEnvelope(X, y, k, scenario_fn):

    param_grid = {'contamination': [0.001], 'support_fraction': [None]}
    def build_model(params): return EllipticEnvelope(**params, random_state=42)

    best_params, _ = optimal_param_search(X, y, scenario_fn, build_model, param_grid)
    print("Best EE params:", best_params)

    pca = PCA(n_components=0.95)
    all_results = []

    for fold_idx, train_idx, test_idx in scenario_fn(X, y, k):

        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_test  = y.iloc[test_idx].values

        (pca_out, fe_time, fe_mem) = measure_feature_step(
            lambda: (pca.fit_transform(X_train), pca.transform(X_test))
        )
        X_train_red, X_test_red = pca_out

        model = EllipticEnvelope(**best_params, random_state=42)
        (_, clf_time, clf_mem) = measure_classification_step(
            lambda: model.fit(X_train_red)
        )

        y_pred_raw = model.predict(X_test_red)
        y_pred = np.where(y_pred_raw == -1, 1, 0)

        all_results.append(
            (fold_idx, test_idx, y_pred, y_test, model,
             fe_time, fe_mem, clf_time, clf_mem)
        )

    return all_results


# =====================================================================
# LOF (Scenario 1)
# =====================================================================
def run_LOF(X, y, k, scenario_fn):

    param_grid = {'n_neighbors': [10], 'metric': ['euclidean']}
    def build_model(params): return LocalOutlierFactor(novelty=True, **params)

    best_params, _ = optimal_param_search(X, y, scenario_fn, build_model, param_grid)
    print("Best LOF params:", best_params)

    pca = PCA(n_components=0.95)
    all_results = []

    for fold_idx, train_idx, test_idx in scenario_fn(X, y, k):

        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_test  = y.iloc[test_idx].values

        (pca_out, fe_time, fe_mem) = measure_feature_step(
            lambda: (pca.fit_transform(X_train), pca.transform(X_test))
        )
        X_train_red, X_test_red = pca_out

        model = LocalOutlierFactor(**best_params, novelty=True)
        (_, clf_time, clf_mem) = measure_classification_step(
            lambda: model.fit(X_train_red)
        )

        y_pred_raw = model.predict(X_test_red)
        y_pred = np.where(y_pred_raw == -1, 1, 0)

        all_results.append(
            (fold_idx, test_idx, y_pred, y_test, model,
             fe_time, fe_mem, clf_time, clf_mem)
        )

    return all_results


# =====================================================================
# Binary SVM (Scenario 2 & 3)
# =====================================================================
def run_binary_svm(X, y, k, scenario_fn):

    best_params = {'C': 10.0, 'gamma': 'scale'}
    pca = PCA(n_components=0.95)
    all_results = []

    for fold_idx, attack_id, train_idx, test_idx in scenario_fn(X, y, k):

        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx].values

        (pca_out, fe_time, fe_mem) = measure_feature_step(
            lambda: (pca.fit_transform(X_train), pca.transform(X_test))
        )
        X_train_red, X_test_red = pca_out

        model = SVC(kernel="rbf", **best_params)
        (_, clf_time, clf_mem) = measure_classification_step(
            lambda: model.fit(X_train_red, y_train)
        )

        y_pred = model.predict(X_test_red)

        all_results.append(
            (fold_idx, attack_id, test_idx, y_pred, y_test, model,
             fe_time, fe_mem, clf_time, clf_mem)
        )

    return all_results


# =====================================================================
# kNN (Scenario 2 & 3)
# =====================================================================
def run_knn(X, y, k, scenario_fn):

    param_grid = {
        'n_neighbors': [3],
        'weights': ['uniform'],
        'metric': ['euclidean']
    }
    def build_model(params):
        return KNeighborsClassifier(
            n_neighbors=params['n_neighbors'],
            weights=params['weights'],
            metric=params['metric']
        )

    best_params, _ = optimal_param_search(X, y, lambda X, y: scenario_fn(X, y), build_model, param_grid)
    print("Best kNN params:", best_params)

    pca = PCA(n_components=0.95)
    all_results = []

    for fold_idx, attack_id, train_idx, test_idx in scenario_fn(X, y, k):

        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx].values

        (pca_out, fe_time, fe_mem) = measure_feature_step(
            lambda: (pca.fit_transform(X_train), pca.transform(X_test))
        )
        X_train_red, X_test_red = pca_out

        model = KNeighborsClassifier(**best_params)
        (_, clf_time, clf_mem) = measure_classification_step(
            lambda: model.fit(X_train_red, y_train)
        )

        y_pred = model.predict(X_test_red)

        all_results.append(
            (fold_idx, attack_id, test_idx, y_pred, y_test, model,
             fe_time, fe_mem, clf_time, clf_mem)
        )

    return all_results


# =====================================================================
# Random Forest (Scenario 2 & 3)
# =====================================================================
def run_random_forest(X, y, k, scenario_fn):

    param_grid = {
        'n_estimators': [50],
        'max_depth': [5],
        'min_samples_split': [5]
    }
    def build_model(params):
        return RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            random_state=42,
            n_jobs=-1
        )

    best_params, _ = optimal_param_search(X, y, lambda X, y: scenario_fn(X, y), build_model, param_grid)
    print("Best RF params:", best_params)

    pca = PCA(n_components=0.95)
    all_results = []

    for fold_idx, attack_id, train_idx, test_idx in scenario_fn(X, y, k):

        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx].values

        (pca_out, fe_time, fe_mem) = measure_feature_step(
            lambda: (pca.fit_transform(X_train), pca.transform(X_test))
        )
        X_train_red, X_test_red = pca_out

        model = RandomForestClassifier(**best_params, random_state=42)
        (_, clf_time, clf_mem) = measure_classification_step(
            lambda: model.fit(X_train_red, y_train)
        )

        y_pred = model.predict(X_test_red)

        all_results.append(
            (fold_idx, attack_id, test_idx, y_pred, y_test, model,
             fe_time, fe_mem, clf_time, clf_mem)
        )

    return all_results
