import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from optimization import optimal_param_search

def run_OneClassSVM(X, y, scenario_fn):

    param_grid_ocsvm = {
    'nu': [0.001, 0.01, 0.05],
    'gamma': ['scale', 0.1, 0.01, 0.001]
    }

    def build_ocsvm(params):
        return OneClassSVM(kernel='rbf', **params)

    best_params_ocsvm, results_ocsvm = optimal_param_search(X, y, scenario_fn, build_ocsvm, param_grid_ocsvm) 

    all_fold_predictions = []   # store for later if needed

    for fold_idx, train_idx, test_idx in scenario_fn(X, y):

        #xtest is feature data used to testing
        X_train , X_test = X.iloc[train_idx] , X.iloc[test_idx]    # labeled      # normal fold + all attacks
        y_test  = y.iloc[test_idx] # test set for current fold # binary label of testing attack or no 

        ocsvm = OneClassSVM(kernel='rbf', **best_params_ocsvm)
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
    param_grid_ee = {
    'contamination': [0.001, 0.01, 0.05],
    'support_fraction': [None, 0.7, 0.9]
    }

    def build_elliptic(params):
        return EllipticEnvelope(**params, random_state=42)
    
    best_params_ee, results_ee = optimal_param_search(X, y, scenario_fn, build_elliptic, param_grid_ee)

    all_fold_predictions = []

    for fold_idx, train_idx, test_idx in scenario_fn(X, y):

        X_train = X.iloc[train_idx]        # normal only
        X_test  = X.iloc[test_idx]         # normal fold + all attacks
        y_test  = y.iloc[test_idx]

        # Create model
        ee = EllipticEnvelope(**best_params_ee, random_state=42)

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
    param_grid_lof = {
    'n_neighbors': [10, 20, 30, 50],
    'metric': ['euclidean', 'manhattan']
    }

    def build_lof(params):
        return LocalOutlierFactor( novelty=True,**params)
    
    best_params_lof, results_lof = optimal_param_search(X, y, scenario_fn, build_lof, param_grid_lof)

    all_fold_predictions = []

    for fold_idx, train_idx, test_idx in scenario_fn(X, y):

        X_train = X.iloc[train_idx]      # normal only
        X_test  = X.iloc[test_idx]       # normal + attack
        y_test  = y.iloc[test_idx]

        # LOF model (novelty=True allows .predict on X_test)
        lof = LocalOutlierFactor(
            **best_params_lof,
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

def run_binary_svm(X, y, attack_type, attack_intervals, scenario_fn, C=1.0, gamma='scale'):
     
    param_grid_binary_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001]
    }
    def build_binary_svm(C=1.0, gamma='scale', kernel='rbf'):
        return SVC(C=C, gamma=gamma, kernel=kernel)
    best_params_svm, results_svm = optimal_param_search(X, y, lambda X,y: scenario_fn(X,y,attack_type,attack_intervals), build_binary_svm, param_grid_binary_svm)

    svm_predictions = []

    for fold_idx, attack_id, train_idx, test_idx in scenario_fn(X, y, attack_type, attack_intervals):

        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]

        model = SVC(C=C, gamma=gamma, kernel='rbf')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = (y_pred == y_test).mean()

        print(f"Fold {fold_idx+1}, AttackID={attack_id}: Accuracy={acc:.4f} | Train={len(train_idx)}, Test={len(test_idx)}")

        svm_predictions.append((fold_idx, attack_id, test_idx, y_pred, y_test.values))

    return svm_predictions

def run_knn(X, y, attack_type, attack_intervals, scenario_fn, params):

    def build_knn(params):
        return KNeighborsClassifier(
        n_neighbors=params['n_neighbors'],
        weights=params['weights'],
        metric=params['metric']
        )

    param_grid_knn = {
    'n_neighbors': [3, 5, 7, 11],       # odd numbers → avoid ties
    'weights': ['uniform', 'distance'], # common choices
    'metric': ['euclidean', 'manhattan'] # standard distances
    }
    best_params_knn, results_knn = optimal_param_search(X, y, lambda X,y: scenario_fn(X,y,attack_type,attack_intervals), build_knn, param_grid_knn)

    all_predictions = []

    for fold_idx, attack_id, train_idx, test_idx in scenario_fn(X, y, attack_type, attack_intervals):

        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]

        model = KNeighborsClassifier(**best_params_knn)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = (y_pred == y_test).mean()
        print(f"Fold {fold_idx+1}, AttackID={attack_id}: kNN Accuracy={acc:.4f}")

        all_predictions.append((fold_idx, attack_id, test_idx, y_pred, y_test.values))

    return all_predictions

def run_random_forest(X, y, attack_type, attack_intervals, scenario_fn, params):

    def build_rf(params):
        return RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        random_state=42,
        n_jobs=-1     # use all cores
        )

    param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
    }
    best_params_rf, results_rf = optimal_param_search(X, y, lambda X,y: scenario_fn(X,y,attack_type,attack_intervals), build_rf, param_grid_rf)

    all_predictions = []

    for fold_idx, attack_id, train_idx, test_idx in scenario_fn(X, y, attack_type, attack_intervals):

        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]

        model = RandomForestClassifier(**best_params_rf, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = (y_pred == y_test).mean()
        print(f"Fold {fold_idx+1}, AttackID={attack_id}: Random Forest Accuracy={acc:.4f}")

        all_predictions.append((fold_idx, attack_id, test_idx, y_pred, y_test.values))

    return all_predictions
