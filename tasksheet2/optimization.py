import numpy as np

def optimal_param_search(X, y, scenario_fn, model_builder, param_grid):
    """
    model_builder(params) → returns a model instance
    param_grid = dict of lists
    """

    best_params = None
    best_score = -1
    results = []

    # Create all parameter combinations manually
    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))

        fold_scores = []

        for fold_idx, train_idx, test_idx in scenario_fn(X, y):
            X_train = X.iloc[train_idx]
            X_test  = X.iloc[test_idx]
            y_test  = y.iloc[test_idx]

            # Create model with current param combo
            model = model_builder(params)
            model.fit(X_train)

            # One-class models output +1 (normal) / -1 (attack)
            raw_pred = model.predict(X_test)
            y_pred = (raw_pred == -1).astype(int)  # attack=1, normal=0

            acc = np.mean(y_pred == y_test)
            fold_scores.append(acc)

        avg_score = np.mean(fold_scores)
        results.append((params, avg_score))

        print(f"Params {params} → Avg Acc = {avg_score:.4f}")

        # Keep best
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    print("\nBEST PARAMS:", best_params, "Score:", best_score)
    return best_params, results



