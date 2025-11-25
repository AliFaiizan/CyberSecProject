#!/usr/bin/env python3
import argparse, os
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Ensemble rules
# ---------------------------------------------------------
def ensemble(preds_list, method):
    P = np.vstack(preds_list)    # shape (3, N)

    if method == "random":
        idx = np.random.randint(0, 3, size=P.shape[1])
        return P[idx, np.arange(P.shape[1])]

    elif method == "majority":
        return (P.sum(axis=0) >= 2).astype(int)

    elif method == "all":
        return (P.sum(axis=0) == 3).astype(int)

    else:
        raise ValueError("Unknown ensemble method:", method)


# ---------------------------------------------------------
# Load a prediction CSV from Task 1
# ---------------------------------------------------------
def load_pred(path):
    df = pd.read_csv(path)

    # Your files use:
    #   Attack      -> true label
    #   predicted_  -> model prediction
    pred_col = "predicted_label"
    true_col = "Attack"

    if pred_col not in df.columns:
        raise KeyError(f"Prediction column '{pred_col}' not found in {path}")
    if true_col not in df.columns:
        raise KeyError(f"True label column '{true_col}' not found in {path}")

    return df[pred_col].values, df[true_col].values


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Task 2 Ensemble (Prediction Fusion)")
    parser.add_argument("-sc", "--scenario", required=True, type=int, choices=[1,2,3],
                        help="Scenario number: 1 (one-class), 2 or 3 (binary)")
    parser.add_argument("-m", "--method", required=True,
                        choices=["random", "majority", "all"],
                        help="Ensemble method")
    parser.add_argument("-f", "--fold", required=True, type=int,
                        help="Fold number")
    args = parser.parse_args()

    sc      = args.scenario
    fold    = args.fold
    method  = args.method

    print(f"\n=== Ensemble | Scenario {sc} | Fold {fold} | Method={method} ===")

    # -----------------------------------------------------
    # Set model names per scenario
    # -----------------------------------------------------
    if sc == 1:
        models = ["OCSVM", "LOF", "EE"]
        base   = "exports/Scenario1"
    else:
        models = ["RF", "KNN", "SVM"]
        base   = f"exports/Scenario{sc}"

    # -----------------------------------------------------
    # Load predictions for all models for this fold
    # -----------------------------------------------------
    preds = []
    true_y = None

    for model in models:
        path = f"{base}/{model}/Predictions_Fold{fold}.csv"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing prediction file: {path}")

        p, y = load_pred(path)
        preds.append(p)
        true_y = y   # same for all models

    # -----------------------------------------------------
    # Apply ensemble rule
    # -----------------------------------------------------
    final = ensemble(preds, method)

    # -----------------------------------------------------
    # Save ensemble output
    # -----------------------------------------------------
    out_dir = f"{base}/Ensemble"
    os.makedirs(out_dir, exist_ok=True)

    out_path = f"{out_dir}/Ensemble_{method}_Fold{fold}.csv"
    pd.DataFrame({"predicted_": final, "Attack": true_y}).to_csv(out_path, index=False)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    print("\n=== Summary ===")
    print("Total samples:", len(final))
    print("Detected attacks:", final.sum())
    print("Normal samples:", len(final) - final.sum())
    print("Accuracy:", (final == true_y).mean())
    print("\nSaved to:", out_path)


if __name__ == "__main__":
    main()
