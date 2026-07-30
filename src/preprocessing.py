"""
preprocessing.py
-----------------
Data loading, cleaning, imputation, normalization, and EDA for the PIMA
Indians Diabetes Dataset.

IMPORTANT: If data/raw/pima.csv is not found, this module generates a
SYNTHETIC placeholder dataset with the same columns and roughly PIMA-like
statistical properties, purely so the rest of the codebase (ML pipeline,
individual ODE model) can be tested end-to-end before the real dataset is
added. Replace data/raw/pima.csv with the real PIMA CSV before producing
any results you intend to put in the report.

Real dataset sources (download and place at data/raw/pima.csv):
- Kaggle: "Pima Indians Diabetes Database"
- UCI ML Repository: "Pima Indians Diabetes"
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "pima.csv")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "pima_clean.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")

FEATURE_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET_COL = "Outcome"

# Columns where a value of 0 is biologically impossible and should be
# treated as missing (this is the standard, well-documented PIMA cleaning
# step -- see Mujumdar & Vaidehi 2019 and general PIMA preprocessing practice).
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def _generate_synthetic_pima(n=768, seed=42):
    """
    Generates a placeholder dataset with the same schema and roughly
    PIMA-like distributions/class balance (~35% positive), so the pipeline
    can be developed and tested without the real file present.
    THIS IS NOT REAL DATA -- do not use synthetic results in the report.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(n * 0.35)
    n_neg = n - n_pos

    def make_group(n_group, glucose_mu, insulin_mu, bmi_mu, age_mu, label):
        return pd.DataFrame({
            "Pregnancies": rng.poisson(3.5, n_group),
            "Glucose": rng.normal(glucose_mu, 25, n_group).clip(40, 250),
            "BloodPressure": rng.normal(72, 12, n_group).clip(0, 122),
            "SkinThickness": rng.normal(23, 10, n_group).clip(0, 60),
            "Insulin": rng.normal(insulin_mu, 90, n_group).clip(0, 600),
            "BMI": rng.normal(bmi_mu, 6, n_group).clip(15, 60),
            "DiabetesPedigreeFunction": rng.gamma(2, 0.2, n_group).clip(0.08, 2.5),
            "Age": rng.normal(age_mu, 10, n_group).clip(21, 81).astype(int),
            "Outcome": label
        })

    neg = make_group(n_neg, 110, 80, 30, 28, 0)
    pos = make_group(n_pos, 145, 130, 34.5, 38, 1)
    df = pd.concat([neg, pos], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    # inject some zeros to mimic real PIMA's missing-as-zero quirk
    zero_frac = 0.05
    for col in ["SkinThickness", "Insulin"]:
        idx = rng.choice(df.index, size=int(len(df) * 0.3), replace=False)
        df.loc[idx, col] = 0
    for col in ["Glucose", "BloodPressure", "BMI"]:
        idx = rng.choice(df.index, size=int(len(df) * zero_frac), replace=False)
        df.loc[idx, col] = 0

    return df


def load_raw_data():
    """Loads the real PIMA CSV if present, else falls back to a synthetic
    placeholder and prints a clear warning."""
    if os.path.exists(RAW_PATH):
        df = pd.read_csv(RAW_PATH)
        print(f"[preprocessing] Loaded REAL dataset from {RAW_PATH} ({len(df)} rows).")
    else:
        df = _generate_synthetic_pima()
        print("=" * 70)
        print("[preprocessing] WARNING: data/raw/pima.csv not found.")
        print("[preprocessing] Using SYNTHETIC placeholder data for testing only.")
        print("[preprocessing] Download the real PIMA dataset and place it at:")
        print(f"                {os.path.abspath(RAW_PATH)}")
        print("=" * 70)
    return df


def clean_data(df):
    """Replaces biologically-impossible zeros with NaN, then median-imputes."""
    df = df.copy()
    for col in ZERO_AS_MISSING_COLS:
        df[col] = df[col].replace(0, np.nan)
    missing_before = df[ZERO_AS_MISSING_COLS].isna().sum()

    for col in ZERO_AS_MISSING_COLS:
        df[col] = df[col].fillna(df[col].median())

    print("[preprocessing] Missing values imputed (median) per column:")
    print(missing_before.to_string())
    return df


def run_eda(df, save=True):
    """Generates and saves standard EDA figures: class balance, correlation
    heatmap, and feature distributions split by outcome."""
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Class balance
    plt.figure(figsize=(5, 4))
    counts = df[TARGET_COL].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values,
                hue=counts.index.astype(str), palette="viridis", legend=False)
    plt.xlabel("Outcome (0 = Non-diabetic, 1 = Diabetic)")
    plt.ylabel("Count")
    plt.title("Class Balance")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 5, str(v), ha="center")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIG_DIR, "01_class_balance.png"), dpi=130)
    plt.close()

    # 2. Correlation heatmap
    plt.figure(figsize=(9, 7))
    corr = df[FEATURE_COLS + [TARGET_COL]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIG_DIR, "02_correlation_heatmap.png"), dpi=130)
    plt.close()

    # 3. Feature distributions split by outcome
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.flatten(), FEATURE_COLS):
        sns.kdeplot(data=df, x=col, hue=TARGET_COL, fill=True, alpha=0.4, ax=ax)
        ax.set_title(col)
    plt.suptitle("Feature Distributions by Outcome", fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIG_DIR, "03_feature_distributions.png"), dpi=130)
    plt.close()

    # 4. Pairplot on a reduced feature subset (full 8-feature pairplot is unreadable)
    subset = ["Glucose", "BMI", "Age", "Insulin", TARGET_COL]
    g = sns.pairplot(df[subset], hue=TARGET_COL, diag_kind="kde", palette="husl")
    g.fig.suptitle("Pairplot: Glucose, BMI, Age, Insulin", y=1.02)
    if save:
        g.savefig(os.path.join(FIG_DIR, "04_pairplot.png"), dpi=130)
    plt.close("all")

    print(f"[preprocessing] EDA figures saved to {os.path.abspath(FIG_DIR)}")


def preprocess_pipeline(save_processed=True):
    """Full pipeline: load -> clean -> EDA -> return cleaned df."""
    df = load_raw_data()
    df_clean = clean_data(df)
    run_eda(df_clean)
    if save_processed:
        os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
        df_clean.to_csv(PROCESSED_PATH, index=False)
        print(f"[preprocessing] Cleaned data saved to {os.path.abspath(PROCESSED_PATH)}")
    return df_clean


if __name__ == "__main__":
    preprocess_pipeline()
