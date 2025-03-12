# eda.py

# 1. Imports & Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# If running in a Jupyter environment and want to use display():
# from IPython.display import display

# 2. Load Data
df = pd.read_csv("data/processed/bmt_dataset.csv")  # Adjust path if needed

# 3. Initial Exploration
print("DataFrame Shape:", df.shape)
# display(df.head())  # works in Jupyter, otherwise use print(df.head())
print(df.head())
df.info()

# 4. Statistical Summariess
print("\n--- Numeric Summary ---")
print(df.describe())

# Check for duplicate rows
duplicate_count = df.duplicated().sum()
print("\nNumber of duplicate rows:", duplicate_count)

# 5. Missing Values
missing_counts = df.isna().sum()
print("\nMissing Values:\n", missing_counts)

# or visually with a heatmap:
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

# percentage of missing values to see if it's severe:
total_rows = len(df)
missing_percent = (df.isna().sum() / total_rows) * 100
print("\nPercentage of Missing Values:\n", missing_percent)

# 6. Identify Categorical vs. Numeric Columns
# Based on your ARFF, many columns that look numeric are actually coded categories.
# We'll define which ones are truly categorical (including binary 0/1 columns),
# and convert them to 'category' dtype.

categorical_cols = [
    "Recipientgender", "Stemcellsource", "Donorage35", "IIIV", "Gendermatch",
    "DonorABO", "RecipientABO", "RecipientRh", "ABOmatch", "CMVstatus",
    "DonorCMV", "RecipientCMV", "Disease", "Riskgroup", "Txpostrelapse",
    "Diseasegroup", "HLAmatch", "HLAmismatch", "Antigen", "Alel", "HLAgrI",
    "Recipientage10", "Recipientageint", "Relapse", "aGvHDIIIIV", "extcGvHD"
    # We deliberately exclude 'survival_status' here if you want it treated as numeric for correlation.
]

# Convert them to category if they exist in df
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

# 7. Quick check on some example categorical column (e.g. Disease)
if "Disease" in df.columns:
    print("\n--- Disease Value Counts ---")
    print(df["Disease"].value_counts())

# 8. Outliers Example for a Truly Numeric Column
# e.g. 'Donorage' (assuming it's real numeric)
if 'Donorage' in df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df['Donorage'])
    plt.title("Boxplot for Donorage")
    plt.show()

# 9. Distribution of Target Variable
# Suppose 'survival_status' is 0=did not survive, 1=survived => It's numeric/binary
if 'survival_status' in df.columns:
    survival_counts = df['survival_status'].value_counts()
    plt.figure(figsize=(5,4))
    sns.barplot(x=survival_counts.index, y=survival_counts.values)
    plt.title("Distribution of survival_status")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.show()
    print("Class Distribution:\n", survival_counts)

# 10. Categorical Columns: Plot distributions
print("\n--- Categorical Distributions ---")
for col in categorical_cols:
    if col in df.columns:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts())
        # Optional bar plot for each categorical col:
        plt.figure(figsize=(5,4))
        sns.countplot(x=col, data=df)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.show()

# 11. Numeric Feature Distributions & Correlations
# Now that we've converted known categorical columns to category dtype, let's find numeric ones:
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Histograms for numeric features
print("\n--- Histograms for Numeric Features ---")
df[numeric_cols].hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Correlation matrix
print("\n--- Correlation Matrix for Numeric Features ---")
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=False, cmap="YlGnBu")
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

# Example: Display correlation with 'survival_status' if it's in numeric_cols
if 'survival_status' in numeric_cols:
    target_corr = corr_matrix['survival_status'].sort_values(ascending=False)
    print("\nCorrelation with survival_status:\n", target_corr)


print("\n--- EDA Complete ---\n")

