import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

def run_OneClassSVM(X, y, scenario_fn):

    all_fold_predictions = []   # store for later if needed

    for fold_idx, train_idx, test_idx in scenario_fn(X, y):

        X_train , X_test = X.iloc[train_idx] , X.iloc[test_idx]    #      # normal fold + all attacks
        y_test  = y.iloc[test_idx] # test set for current fold

        ocsvm = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
        ocsvm.fit(X_train)

        # OC-SVM outputs: +1 normal, -1 anomaly
        y_pred_raw = ocsvm.predict(X_test)

        # Map: -1 → attack(1), 1 → normal(0)
        y_pred = np.where(y_pred_raw == -1, 1, 0)

        acc = np.mean(y_pred == y_test)
        print(f"Fold {fold_idx+1}: Accuracy = {acc:.4f} | Train={len(train_idx)} | Test={len(test_idx)}")

        # Save fold results
        all_fold_predictions.append((fold_idx, test_idx, y_pred, y_test.values))

    return all_fold_predictions

def run_EllipticEnvelope(X, y, scenario_fn, contamination=0.01):
    
    all_fold_predictions = []

    for fold_idx, train_idx, test_idx in scenario_fn(X, y):

        X_train = X.iloc[train_idx]        # normal only
        X_test  = X.iloc[test_idx]         # normal fold + all attacks
        y_test  = y.iloc[test_idx]

        # Create model
        ee = EllipticEnvelope(contamination=contamination, random_state=42)

        # Fit ONLY normal
        ee.fit(X_train)

        # Predict on test
        y_pred_raw = ee.predict(X_test)   # +1 normal, -1 outlier
        y_pred = np.where(y_pred_raw == -1, 1, 0)   # convert to 0/1

        # Accuracy
        acc = np.mean(y_pred == y_test)
        print(f"Fold {fold_idx+1}: EllipticEnvelope Accuracy = {acc:.4f} | Train={len(train_idx)} | Test={len(test_idx)}")

        # Store for later
        all_fold_predictions.append((fold_idx, test_idx, y_pred, y_test.values))

    return all_fold_predictions

def run_LOF(X, y, scenario_fn, n_neighbors=20):
    
    all_fold_predictions = []

    for fold_idx, train_idx, test_idx in scenario_fn(X, y):

        X_train = X.iloc[train_idx]      # normal only
        X_test  = X.iloc[test_idx]       # normal + attack
        y_test  = y.iloc[test_idx]

        # LOF model (novelty=True allows .predict on X_test)
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True
        )

        # Fit ONLY normal samples
        lof.fit(X_train)

        # Predict on test
        y_pred_raw = lof.predict(X_test)     # +1 normal, -1 outlier
        y_pred = np.where(y_pred_raw == -1, 1, 0)

        # Accuracy
        acc = np.mean(y_pred == y_test)
        print(f"Fold {fold_idx+1}: LOF Accuracy = {acc:.4f} | Train={len(train_idx)} | Test={len(test_idx)}")

        # Store results
        all_fold_predictions.append((fold_idx, test_idx, y_pred, y_test.values))

    return all_fold_predictions
import numpy as np

from sklearn.svm import OneClassSVM, SVC
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from utils import optimal_param_search  # kept in case you use it later


def run_OneClassSVM(X, y, scenario_fn):
    """
    One-Class SVM for Scenario 1 (one-class setting).
    scenario_fn(X, y) must yield: (fold_idx, train_idx, test_idx)
    """
    best_params_ocsvm = {'nu': 0.001, 'gamma': 'scale'}
    all_fold_predictions = []

    pca = PCA(n_components=0.95)

    for fold_idx, train_idx, test_idx in scenario_fn(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        print(f"Training One-Class SVM with params: {best_params_ocsvm} on fold {fold_idx + 1}...")
        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)

        ocsvm = OneClassSVM(kernel='rbf', **best_params_ocsvm)
        ocsvm.fit(X_train_reduced)

        print(f"Predicting on test set for fold {fold_idx + 1}...")
        y_pred_raw = ocsvm.predict(X_test_reduced)          # +1 normal, -1 anomaly
        y_pred = np.where(y_pred_raw == -1, 1, 0)           # 1 = attack, 0 = normal

        acc = np.mean(y_pred == y_test)
        print(f"Fold {fold_idx + 1}: Accuracy = {acc:.4f} | Train={len(train_idx)} | Test={len(test_idx)}")

        all_fold_predictions.append((fold_idx, test_idx, y_pred, y_test.values))

    return all_fold_predictions


def run_EllipticEnvelope(X, y, scenario_fn):
    """
    EllipticEnvelope for Scenario 1.
    scenario_fn(X, y) must yield: (fold_idx, train_idx, test_idx)
    """
    best_params_ee = {'contamination': 0.001, 'support_fraction': None}
    all_fold_predictions = []

    pca = PCA(n_components=0.95)

    for fold_idx, train_idx, test_idx in scenario_fn(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)

        ee = EllipticEnvelope(**best_params_ee, random_state=42)
        print(f"Training EllipticEnvelope with params: {best_params_ee} on fold {fold_idx + 1}...")
        ee.fit(X_train_reduced)

        y_pred_raw = ee.predict(X_test_reduced)             # +1 normal, -1 outlier
        y_pred = np.where(y_pred_raw == -1, 1, 0)

        acc = np.mean(y_pred == y_test)
        print(f"Fold {fold_idx + 1}: EllipticEnvelope Accuracy = {acc:.4f} | "
              f"Train={len(train_idx)} | Test={len(test_idx)}")

        all_fold_predictions.append((fold_idx, test_idx, y_pred, y_test.values))

    return all_fold_predictions


def run_LOF(X, y, scenario_fn):
    """
    Local Outlier Factor for Scenario 1.
    scenario_fn(X, y) must yield: (fold_idx, train_idx, test_idx)
    """
    best_params_lof = {'n_neighbors': 20, 'metric': 'euclidean'}
    all_fold_predictions = []

    pca = PCA(n_components=0.95)

    for fold_idx, train_idx, test_idx in scenario_fn(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        lof = LocalOutlierFactor(
            **best_params_lof,
            novelty=True
        )

        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)

        print(f"Training LOF with params: {best_params_lof} on fold {fold_idx + 1}...")
        lof.fit(X_train_reduced)

        y_pred_raw = lof.predict(X_test_reduced)            # +1 normal, -1 outlier
        y_pred = np.where(y_pred_raw == -1, 1, 0)

        acc = np.mean(y_pred == y_test)
        print(f"Fold {fold_idx + 1}: LOF Accuracy = {acc:.4f} | "
              f"Train={len(train_idx)} | Test={len(test_idx)}")

        all_fold_predictions.append((fold_idx, test_idx, y_pred, y_test.values))

    return all_fold_predictions


def run_binary_svm(X, y, scenario_fn):
    """
    Binary SVM for Scenario 2 & 3.
    scenario_fn(X, y) must yield: (fold_idx, attack_id, train_idx, test_idx)
    """
    # More aggressive C and class_weight='balanced' to avoid predicting only normal
    best_params_svm = {
        'C': 10.0,
        'gamma': 'scale',
        'kernel': 'rbf',
        'class_weight': 'balanced'
    }

    svm_predictions = []
    pca = PCA(n_components=0.95)

    for fold_idx, attack_id, train_idx, test_idx in scenario_fn(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)

        model = SVC(**best_params_svm)
        model.fit(X_train_reduced, y_train)

        y_pred = model.predict(X_test_reduced)
        acc = (y_pred == y_test).mean()

        print(f"Fold {fold_idx + 1}, AttackID={attack_id}: Accuracy={acc:.4f} | "
              f"Train={len(train_idx)}, Test={len(test_idx)}")

        svm_predictions.append((fold_idx, attack_id, test_idx, y_pred, y_test.values))

    return svm_predictions


def run_knn(X, y, scenario_fn):
    """
    kNN for Scenario 2 & 3.
    scenario_fn(X, y) must yield: (fold_idx, attack_id, train_idx, test_idx)
    """
    best_params_knn = {'n_neighbors': 3, 'weights': 'uniform', 'metric': 'euclidean'}
    all_predictions = []

    pca = PCA(n_components=0.95)

    for fold_idx, attack_id, train_idx, test_idx in scenario_fn(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)

        model = KNeighborsClassifier(**best_params_knn)
        model.fit(X_train_reduced, y_train)

        y_pred = model.predict(X_test_reduced)
        acc = (y_pred == y_test).mean()

        print(f"Fold {fold_idx + 1}, AttackID={attack_id}: kNN Accuracy={acc:.4f}")

        all_predictions.append((fold_idx, attack_id, test_idx, y_pred, y_test.values))

    return all_predictions


def run_random_forest(X, y, scenario_fn):
    """
    Random Forest for Scenario 2 & 3.
    scenario_fn(X, y) must yield: (fold_idx, attack_id, train_idx, test_idx)
    """
    best_params_rf = {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 5}
    all_predictions = []

    pca = PCA(n_components=0.95)

    for fold_idx, attack_id, train_idx, test_idx in scenario_fn(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)

        model = RandomForestClassifier(**best_params_rf, random_state=42, n_jobs=-1)
        model.fit(X_train_reduced, y_train)

        y_pred = model.predict(X_test_reduced)
        acc = (y_pred == y_test).mean()

        print(f"Fold {fold_idx + 1}, AttackID={attack_id}: Random Forest Accuracy={acc:.4f}")

        all_predictions.append((fold_idx, attack_id, test_idx, y_pred, y_test.values))

    return all_predictions
