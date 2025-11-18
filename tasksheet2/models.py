import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC

def run_OneClassSVM(X, y, scenario_fn):

    all_fold_predictions = []   # store for later if needed

    for fold_idx, train_idx, test_idx in scenario_fn(X, y):

        #xtest is feature data used to testing
        X_train , X_test = X.iloc[train_idx] , X.iloc[test_idx]    # labeled      # normal fold + all attacks
        y_test  = y.iloc[test_idx] # test set for current fold # binary label of testing attack or no 

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

def run_binary_svm(X, y, attack_type, attack_intervals, scenario_fn, C=1.0, gamma='scale'):
    
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