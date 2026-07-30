"""
ml_pipeline.py
---------------
Machine Learning classification pipeline for diabetes prediction on the
PIMA dataset. Extends Mujumdar & Vaidehi (2019)'s classifier comparison
with proper k-fold cross-validation, confusion matrix heatmaps, ROC-AUC
curves, and feature importance analysis.

Note on tooling: this environment has no internet access, so XGBoost/SHAP/
Keras are substituted with solid scikit-learn equivalents:
  - GradientBoostingClassifier instead of XGBoost
  - MLPClassifier as the simple ANN
  - permutation_importance instead of SHAP
If you have internet on your own machine, feel free to add
`from xgboost import XGBClassifier` to the MODELS dict below and re-run --
the rest of the pipeline (CV, plots, tables) will work unchanged.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier, ExtraTreesClassifier,
                               BaggingClassifier)
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              confusion_matrix, classification_report)
from sklearn.inspection import permutation_importance

from preprocessing import preprocess_pipeline, FEATURE_COLS, TARGET_COL, FIG_DIR

TABLE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "tables")

RANDOM_STATE = 42

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=9),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
    "LDA": LinearDiscriminantAnalysis(),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Bagging": BaggingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "MLP (ANN)": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=RANDOM_STATE),
}


def get_train_test(df, test_size=0.2):
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler


def cross_validate_models(X_train, y_train, k=5):
    """Runs stratified k-fold CV for every model and returns a summary
    DataFrame with mean +/- std for each metric (not just a single point
    estimate accuracy)."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    rows = []
    for name, model in MODELS.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        row = {"Model": name}
        for m in scoring:
            vals = scores[f"test_{m}"]
            row[f"{m}_mean"] = vals.mean()
            row[f"{m}_std"] = vals.std()
        rows.append(row)
        print(f"[ml_pipeline] CV done: {name:22s} "
              f"acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}")

    return pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False).reset_index(drop=True)


def fit_and_evaluate_on_holdout(X_train, y_train, X_test, y_test):
    """Fits every model on the full training set and evaluates on the held-out
    test set -- produces confusion matrices, classification reports, and ROC
    curves per model."""
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    results = []
    reports = {}
    roc_data = {}

    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

        results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                         "Recall": rec, "F1": f1, "AUC": auc})
        reports[name] = classification_report(y_test, y_pred, target_names=["Non-diabetic", "Diabetic"])

        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_data[name] = (fpr, tpr, auc)

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return results_df, reports, roc_data


def plot_confusion_matrices(models_fitted, X_test, y_test, top_n=6):
    """Confusion matrix heatmaps for the top-N models by accuracy."""
    accs = {name: accuracy_score(y_test, m.predict(X_test)) for name, m in models_fitted.items()}
    top_models = sorted(accs, key=accs.get, reverse=True)[:top_n]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, name in zip(axes.flatten(), top_models):
        model = models_fitted[name]
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        ax.set_title(f"{name}\nAcc={accs[name]:.3f}")
    plt.suptitle("Confusion Matrices — Top Models", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_confusion_matrices.png"), dpi=130)
    plt.close()


def plot_roc_curves(roc_data):
    plt.figure(figsize=(8, 7))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=1.8)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "06_roc_curves.png"), dpi=130)
    plt.close()


def plot_accuracy_comparison(results_df):
    plt.figure(figsize=(11, 6))
    order = results_df.sort_values("Accuracy", ascending=True)
    bars = plt.barh(order["Model"], order["Accuracy"], color=sns.color_palette("viridis", len(order)))
    plt.xlabel("Accuracy")
    plt.title("Model Comparison — Held-out Test Accuracy")
    plt.xlim(0, 1)
    for bar, val in zip(bars, order["Accuracy"]):
        plt.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "07_accuracy_comparison.png"), dpi=130)
    plt.close()


def plot_feature_importance(best_model_name, models_fitted, X_test, y_test, feature_names):
    """Permutation importance (model-agnostic, stands in for SHAP)."""
    model = models_fitted[best_model_name]
    result = permutation_importance(model, X_test, y_test, n_repeats=20,
                                      random_state=RANDOM_STATE, scoring="accuracy")
    order = np.argsort(result.importances_mean)

    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[order], result.importances_mean[order],
              xerr=result.importances_std[order], color="teal")
    plt.xlabel("Permutation Importance (accuracy drop)")
    plt.title(f"Feature Importance — {best_model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "08_feature_importance.png"), dpi=130)
    plt.close()


def run_ml_pipeline():
    df = preprocess_pipeline()

    X_train, X_test, y_train, y_test, scaler = get_train_test(df)

    print("\n[ml_pipeline] === Cross-validation (5-fold) ===")
    cv_results = cross_validate_models(X_train, y_train, k=5)
    cv_results.to_csv(os.path.join(TABLE_DIR, "cv_results.csv"), index=False)

    print("\n[ml_pipeline] === Holdout evaluation ===")
    results_df, reports, roc_data = fit_and_evaluate_on_holdout(X_train, y_train, X_test, y_test)
    results_df.to_csv(os.path.join(TABLE_DIR, "holdout_results.csv"), index=False)

    # refit all models (fit_and_evaluate already did, but we need the objects for plotting)
    models_fitted = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        models_fitted[name] = model

    plot_confusion_matrices(models_fitted, X_test, y_test)
    plot_roc_curves(roc_data)
    plot_accuracy_comparison(results_df)

    best_model_name = results_df.iloc[0]["Model"]
    plot_feature_importance(best_model_name, models_fitted, X_test, y_test, FEATURE_COLS)

    # Save classification reports as text
    with open(os.path.join(TABLE_DIR, "classification_reports.txt"), "w") as f:
        for name, report in reports.items():
            f.write(f"{'='*60}\n{name}\n{'='*60}\n{report}\n\n")

    # Save best model's key metrics -- this is the handoff used by
    # ode_population.py's ML-informed screening scenario (recall/sensitivity
    # motivates a *range* of treatment-rate increases, not a precise formula --
    # see docs/module-C-status.md for the reasoning).
    best_row = results_df.iloc[0].to_dict()
    handoff = {
        "best_model": best_model_name,
        "accuracy": best_row["Accuracy"],
        "recall_sensitivity": best_row["Recall"],
        "precision": best_row["Precision"],
        "f1": best_row["F1"],
        "auc": best_row["AUC"],
    }
    with open(os.path.join(TABLE_DIR, "ml_handoff_metrics.json"), "w") as f:
        json.dump(handoff, f, indent=2)

    print("\n[ml_pipeline] Best model:", best_model_name)
    print("[ml_pipeline] Results table:\n", results_df.round(3).to_string(index=False))
    print(f"\n[ml_pipeline] All figures saved to {os.path.abspath(FIG_DIR)}")
    print(f"[ml_pipeline] All tables saved to {os.path.abspath(TABLE_DIR)}")
    print(f"[ml_pipeline] Handoff metrics for ODE module saved to ml_handoff_metrics.json")

    return results_df, cv_results, handoff


if __name__ == "__main__":
    run_ml_pipeline()
