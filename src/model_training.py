# src/model_training.py

import os
import pandas as pd
import numpy as np
import joblib  # for saving/loading models

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def load_train_test(train_csv, test_csv, target="survival_status"):
    """
    Loads the balanced train set and the original test set,
    separates them into (X_train, y_train) and (X_test, y_test).

    This function also prints shapes and column names for debugging.
    """
    print(f"[INFO] Reading training data from: {train_csv}")
    train_df = pd.read_csv(train_csv)
    print("[DEBUG] train_df shape:", train_df.shape)
    print("[DEBUG] train_df columns:", train_df.columns.tolist())

    print(f"[INFO] Reading test data from: {test_csv}")
    test_df = pd.read_csv(test_csv)
    print("[DEBUG] test_df shape:", test_df.shape)
    print("[DEBUG] test_df columns:", test_df.columns.tolist())

    # Check that the target column is indeed in the dataset
    if target not in train_df.columns:
        raise ValueError(f"[ERROR] Target column '{target}' not found in train CSV!")
    if target not in test_df.columns:
        raise ValueError(f"[ERROR] Target column '{target}' not found in test CSV!")

    # Separate features and target
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    print("[DEBUG] X_train shape:", X_train.shape)
    print("[DEBUG] X_train columns:", X_train.columns.tolist())
    print("[DEBUG] y_train shape:", y_train.shape)

    print("[DEBUG] X_test shape:", X_test.shape)
    print("[DEBUG] X_test columns:", X_test.columns.tolist())
    print("[DEBUG] y_test shape:", y_test.shape)

    return X_train, y_train, X_test, y_test


def main():
    # 1. Specify paths
    TRAIN_PATH = "data/processed/bmt_train_balanced.csv"
    TEST_PATH  = "data/processed/bmt_test.csv"

    # 2. Load and inspect data
    X_train, y_train, X_test, y_test = load_train_test(TRAIN_PATH, TEST_PATH)

    # 3. Define the models you want to train
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "SVM": SVC(probability=True, kernel='rbf', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42)
    }

    # Ensure the 'models/' directory exists for saving
    os.makedirs("models", exist_ok=True)

    # 4. Train each model and save (no evaluation here)
    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")
        model.fit(X_train, y_train)

        # Optionally, check feature importances for tree-based models (for debugging)
        # if hasattr(model, "feature_importances_"):
        #     importances = model.feature_importances_
        #     print(f"[DEBUG] {model_name} feature importances:", importances)

        # Save the trained model to 'models/' folder
        model_filename = f"models/{model_name.lower()}.pkl"
        joblib.dump(model, model_filename)
        print(f"[INFO] {model_name} model saved to {model_filename}")


if __name__ == "__main__":
    main()
