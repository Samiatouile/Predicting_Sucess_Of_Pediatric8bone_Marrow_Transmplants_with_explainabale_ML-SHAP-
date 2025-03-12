# src/data_processing.py

import os
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas.api.types import CategoricalDtype  # for checking categorical dtypes

###############################################################################
#                               GLOBAL SETTINGS                               #
###############################################################################
# Columns known to be categorical, based on ARFF/EDA:
CATEGORICAL_COLS = [
    "Recipientgender", "Stemcellsource", "Donorage35", "IIIV", "Gendermatch",
    "DonorABO", "RecipientABO", "RecipientRh", "ABOmatch", "CMVstatus",
    "DonorCMV", "RecipientCMV",
    "Disease",  # We'll still treat Disease as categorical for missing-value imputation.
    "Riskgroup", "Txpostrelapse", "Diseasegroup", "HLAmatch", "HLAmismatch",
    "Antigen", "Alel", "HLAgrI", "Recipientage10", "Recipientageint", "Relapse",
    "aGvHDIIIIV", "extcGvHD"
]

# Columns truly numeric, including 'survival_status' if you consider it numeric (0/1).
NUMERIC_COLS = [
    "Donorage", "Recipientage", "CD34kgx10d6", "CD3dCD34", "CD3dkgx10d8",
    "Rbodymass", "ANCrecovery", "PLTrecovery", "time_to_aGvHD_III_IV",
    "survival_time", "survival_status"
]

###############################################################################
#                               HELPER FUNCTIONS                              #
###############################################################################
def load_data(csv_path: str) -> pd.DataFrame:
    """
    Loads dataset from a CSV file path and returns a pandas DataFrame.
    """
    return pd.read_csv(csv_path)


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert known categorical columns to 'category' dtype.
    """
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
      - Numeric columns: impute with mean
      - Categorical columns: impute with most frequent (mode)
    """
    # Separate numeric from categorical data
    numeric_data = df[NUMERIC_COLS].copy()
    categorical_data = df[CATEGORICAL_COLS].copy()

    # Impute numeric columns with mean
    numeric_imputer = SimpleImputer(strategy='mean')
    numeric_data_imputed = pd.DataFrame(
        numeric_imputer.fit_transform(numeric_data),
        columns=numeric_data.columns
    )

    # Impute categorical columns with the most frequent value
    freq_imputer = SimpleImputer(strategy='most_frequent')
    categorical_data_imputed = pd.DataFrame(
        freq_imputer.fit_transform(categorical_data),
        columns=categorical_data.columns
    ).astype("category")

    # Update original df with imputed data
    for col in numeric_data_imputed.columns:
        df[col] = numeric_data_imputed[col]
    for col in categorical_data_imputed.columns:
        df[col] = categorical_data_imputed[col]

    return df


def encode_disease(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the 'Disease' attribute into numeric codes.
    """
    if "Disease" in df.columns:
        # Ensure 'Disease' is a categorical dtype
        if not isinstance(df["Disease"].dtype, CategoricalDtype):
            df["Disease"] = df["Disease"].astype("category")
        df["Disease"] = df["Disease"].cat.codes
    return df


def handle_outliers(df: pd.DataFrame, method: str = "none") -> pd.DataFrame:
    """
    Optionally handle outliers in numeric columns.
      - method='none': do nothing
      - method='clip': clip outliers at [1st percentile, 99th percentile]
    """
    if method == "clip":
        for col in NUMERIC_COLS:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)
    return df


def scale_numeric_data(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    """
    Scales numeric columns using either 'standard' (StandardScaler) or 'minmax' (MinMaxScaler).
    Excludes 'survival_status' from scaling.
    """
    # Only include columns that still exist in df
    numeric_cols_to_scale = [c for c in NUMERIC_COLS if c != "survival_status" and c in df.columns]

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("scale method must be 'standard' or 'minmax'")

    df_scaled = df.copy()
    df_scaled[numeric_cols_to_scale] = scaler.fit_transform(df_scaled[numeric_cols_to_scale])
    return df_scaled


def drop_correlated_features(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    Drops one of each pair of highly correlated numeric features.
    The target variable 'survival_status' is preserved.
    """
    corr_matrix = df[NUMERIC_COLS].corr().abs()
    # Use the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Identify columns to drop; ensure 'survival_status' is not dropped
    to_drop = [col for col in upper.columns if col != "survival_status" and any(upper[col] > threshold)]
    print(f"[INFO] Dropping correlated features: {to_drop}")
    df.drop(columns=to_drop, inplace=True)
    return df


###############################################################################
#                          MAIN PIPELINE ORCHESTRATION                        #
###############################################################################
def process_data(
    input_csv: str,
    output_csv: str,
    scale_method: str = None,
    outlier_method: str = "none",
    corr_threshold: float = None
) -> pd.DataFrame:
    """
    Orchestrates the data processing pipeline:
      1) Load data
      2) Convert columns to appropriate dtypes
      3) Impute missing values
      4) Encode 'Disease' to numeric codes
      5) Optionally handle outliers
      6) Optionally drop highly correlated features
      7) Optionally scale numeric data
      8) Save the cleaned CSV file
      9) Return the processed DataFrame
    """
    # 1. Load data
    df = load_data(input_csv)
    print(f"[INFO] Loaded dataset from {input_csv}, shape={df.shape}")

    # 2. Convert dtypes (categorical/numeric)
    df = convert_dtypes(df)

    # 3. Handle missing values
    df = handle_missing_values(df)

    # 4. Encode 'Disease' as numeric codes
    df = encode_disease(df)

    # 5. Outlier handling (optional)
    df = handle_outliers(df, method=outlier_method)

    # 6. Drop correlated features (if threshold provided)
    if corr_threshold is not None:
        df = drop_correlated_features(df, threshold=corr_threshold)

    # 7. Scale numeric data (optional)
    if scale_method is not None:
        df = scale_numeric_data(df, method=scale_method)

    # 8. Save the processed dataset
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Processed dataset saved to {output_csv}. Shape={df.shape}")

    # 9. Return the processed DataFrame so you can inspect it
    return df


###############################################################################
#                          STANDALONE EXECUTION EXAMPLE                       #
###############################################################################
if __name__ == "__main__":
    INPUT_CSV = "data/processed/bmt_dataset.csv"           # Original file from EDA
    OUTPUT_CSV = "data/processed/bmt_dataset_cleaned.csv"  # Processed file output

    final_df = process_data(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        scale_method="standard",    # Options: "standard", "minmax", or None
        outlier_method="none",      # Options: "none", "clip", etc.
        corr_threshold=0.8          # Drop features with correlation > 0.8
    )

    # Show the updated DataFrame in the console
    print("\n=== HEAD OF PROCESSED DATA ===")
    print(final_df.head())

    print("\n=== DATAFRAME INFO ===")
    final_df.info()
