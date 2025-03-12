# src/model_testing.py

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

def load_test_data(test_csv, target_col="survival_status"):
    """
    Loads the original test set (with real distribution),
    splits into X_test, y_test.
    """
    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    print(f"[INFO] Loaded test set from {test_csv}, shape={df_test.shape}")
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Uses a trained model to predict on X_test, then prints metrics.
    Returns a dictionary of the main metrics.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary')
    rec = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    # Attempt AUC if the model supports predict_proba
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except AttributeError:
        auc = None

    # Print results
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    if auc is not None:
        print(f"AUC:       {auc:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc
    }

def main():
    # 1. Path to your test set
    TEST_PATH = "data/processed/bmt_test.csv"
    X_test, y_test = load_test_data(TEST_PATH)

    # 2. Models to load and test
    # These should match the filenames you used in model_training.py
    model_files = {
        "RandomForest": "models/randomforest.pkl",
        "XGBoost":      "models/xgboost.pkl",
        "SVM":          "models/svm.pkl",
        "LightGBM":     "models/lightgbm.pkl"
    }

    # 3. Evaluate each model
    results = {}
    for model_name, model_path in model_files.items():
        if not os.path.exists(model_path):
            print(f"[WARNING] {model_path} not found. Skipping {model_name}.")
            continue

        print(f"\n=== Loading {model_name} from {model_path} ===")
        model = joblib.load(model_path)

        print(f"Evaluating {model_name} on test set...")
        metrics = evaluate_model(model, X_test, y_test)
        results[model_name] = metrics

    # 4. Print summary of all results
    print("\n=== Summary of All Models ===")
    for model_name, metrics in results.items():
        auc_str = f"{metrics['auc']:.3f}" if metrics['auc'] is not None else "N/A"
        print(f"{model_name}: Acc={metrics['accuracy']:.3f}, "
              f"F1={metrics['f1']:.3f}, AUC={auc_str}")

if __name__ == "__main__":
    main()
