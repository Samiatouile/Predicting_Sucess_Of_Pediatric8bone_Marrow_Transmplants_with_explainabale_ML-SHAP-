# class_imbalance.py (or model_training.py)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def fix_class_imbalance(input_csv, output_train_csv=None, output_test_csv=None):
    """
    Loads the cleaned dataset, splits into train/test, applies SMOTE to the train set
    (to fix class imbalance), and optionally saves them to CSV for future training.

    Parameters
    ----------
    input_csv : str
        Path to the final cleaned dataset (e.g. 'data/processed/bmt_dataset_cleaned.csv').

    output_train_csv : str or None
        If provided, the resampled train set is saved to this file.

    output_test_csv : str or None
        If provided, the untouched test set is saved to this file.

    Returns
    -------
    (X_train_res, y_train_res, X_test, y_test)
        Numpy arrays or DataFrames containing resampled train set + original test set.
    """
    # 1. Load the final cleaned dataset
    df = pd.read_csv(input_csv)
    print(f"Loaded dataset from {input_csv}, shape={df.shape}")

    # 2. Separate features (X) and target (y)
    #    Make sure your target column is named 'survival_status' (adjust if different)
    X = df.drop("survival_status", axis=1)
    y = df["survival_status"]
    y = df["survival_status"].astype(int)

    # 3. Train/Test split (stratify keeps the original class ratio in each split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Before SMOTE on training set:")
    print("  y_train distribution:", np.bincount(y_train))
    print("Test set distribution:", np.bincount(y_test))

    # 4. Apply SMOTE on the train set only
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    print("\nAfter SMOTE on training set:")
    print("  y_train_res distribution:", np.bincount(y_train_res))

    # 5. Optionally save the resampled train set and original test set
    if output_train_csv:
        # Combine X_train_res + y_train_res into one DataFrame
        train_res_df = X_train_res.copy()
        train_res_df["survival_status"] = y_train_res
        train_res_df.to_csv(output_train_csv, index=False)
        print(f"[INFO] Resampled train set saved to {output_train_csv}")

    if output_test_csv:
        # Combine X_test + y_test into one DataFrame
        test_df = X_test.copy()
        test_df["survival_status"] = y_test
        test_df.to_csv(output_test_csv, index=False)
        print(f"[INFO] Test set saved to {output_test_csv}")

    # Return them in case you want to use them in code directly (no saving)
    return X_train_res, y_train_res, X_test, y_test


 
    


if __name__ == "__main__":
    # Example usage:
    # We load 'bmt_dataset_cleaned.csv' from data/processed,
    # do a train/test split, fix the imbalance on the train set via SMOTE,
    # and write them to two new CSVs for future model training.
    input_path = "data/processed/bmt_dataset_cleaned.csv"
    train_path = "data/processed/bmt_train_balanced.csv"
    test_path  = "data/processed/bmt_test.csv"

    X_train_res, y_train_res, X_test, y_test = fix_class_imbalance(
        input_csv=input_path,
        output_train_csv=train_path,
        output_test_csv=test_path
    )

    print("\nDone. The balanced train set and test set are now saved to disk.")
