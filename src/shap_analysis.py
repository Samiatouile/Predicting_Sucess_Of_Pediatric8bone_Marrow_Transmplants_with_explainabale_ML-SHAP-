# src/shap_explain.py

import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_test_data(script_dir, target_col="survival_status"):
    """
    Loads the original test set (with real distribution), 
    from data/processed/bmt_test.csv relative to this script's location.
    Returns X_test, y_test.
    """
    # Build a path to bmt_test.csv, which we assume is in ../data/processed/
    test_csv = os.path.join(script_dir, "..", "data", "processed", "bmt_test.csv")
    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    print(f"[INFO] Loaded test set from {test_csv}, shape={df_test.shape}")
    return X_test, y_test

def shap_for_tree_model(model, X_test, model_name):
    """
    Uses shap.TreeExplainer for tree-based models (RandomForest, XGBoost, LightGBM).
    Generates and saves a summary plot.
    """
    explainer = shap.TreeExplainer(model)
    # Compute SHAP values for the entire test set
    shap_values = explainer.shap_values(X_test)

    # Global summary plot (beeswarm)
    plt.title(f"SHAP Summary - {model_name}")
    shap.summary_plot(shap_values, X_test, show=False)
    out_file = f"shap_summary_{model_name.lower()}.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.clf()  # clear figure
    print(f"[INFO] Saved SHAP summary plot for {model_name} to {out_file}")

def shap_for_svm(model, X_test, model_name, sample_size=200):
    """
    Uses shap.KernelExplainer for SVM (model-agnostic).
    We sample X_test to avoid extreme compute times.
    Generates and saves a summary plot.
    """
    # Sample if large
    if len(X_test) > sample_size:
        X_sample = X_test.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_test

    # Small background set
    background = X_sample.sample(n=min(50, len(X_sample)), random_state=42)

    # Define a predict function returning probability of class 1
    def predict_fn(data):
        return model.predict_proba(data)[:, 1]

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_sample, nsamples=100)

    plt.title(f"SHAP Summary - {model_name}")
    shap.summary_plot(shap_values, X_sample, show=False)
    out_file = f"shap_summary_{model_name.lower()}.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.clf()
    print(f"[INFO] Saved SHAP summary plot for {model_name} to {out_file}")

def main():
    # 1. Determine the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Load test data
    X_test, y_test = load_test_data(script_dir)

    # 3. Create a subfolder to store the SHAP plots
    #    We'll store them in shap_plots under the same folder as this script
    shap_plots_dir = os.path.join(script_dir, "shap_plots")
    os.makedirs(shap_plots_dir, exist_ok=True)
    os.chdir(shap_plots_dir)  # Switch to shap_plots as the working directory

    # 4. Dictionary of model files (relative to ../models)
    model_dir = os.path.join(script_dir, "..", "models")
    model_files = {
        "RandomForest": os.path.join(model_dir, "randomforest.pkl"),
        "XGBoost":      os.path.join(model_dir, "xgboost.pkl"),
        "SVM":          os.path.join(model_dir, "svm.pkl"),
        "LightGBM":     os.path.join(model_dir, "lightgbm.pkl")
    }

    # 5. For each model, load & create SHAP summary
    for model_name, model_path in model_files.items():
        if not os.path.exists(model_path):
            print(f"[WARNING] {model_path} not found. Skipping {model_name}.")
            continue

        print(f"\n=== Loading {model_name} from {model_path} ===")
        model = joblib.load(model_path)

        # If it’s tree-based -> TreeExplainer, else SVM -> KernelExplainer
        if model_name in ["RandomForest", "XGBoost", "LightGBM"]:
            shap_for_tree_model(model, X_test, model_name)
        else:
            shap_for_svm(model, X_test, model_name)

    print(f"\n[INFO] All SHAP summary plots saved in: {shap_plots_dir}")

if __name__ == "__main__":
    main()
