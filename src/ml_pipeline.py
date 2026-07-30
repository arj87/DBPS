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
import joblib

from preprocessing import preprocess_pipeline, FEATURE_COLS, TARGET_COL, FIG_DIR

TABLE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

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
    

def _safe_filename(name):
    """Turns a model name like 'Logistic Regression' or 'SVM (RBF)' into a
    filesystem-safe filename fragment: 'logistic_regression', 'svm_rbf'."""
    import re
    name = name.lower().replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


def save_models(models_fitted, scaler, best_model_name, results_df, selection_metric="F1"):
    """
    Persists trained models to disk as .pkl files (via joblib), with the
    model's name baked into the filename so it's immediately obvious which
    model is saved without opening the file (e.g. best_model_adaboost.pkl,
    not a generic best_model.pkl that silently changes meaning between runs).

    Saves TWO models deliberately:
      1. The actual best-performing model on this run, selected by
         `selection_metric` (default F1 -- more clinically meaningful than
         raw Accuracy for a diagnosis task, since it balances precision and
         recall rather than being skewed by class imbalance).
      2. AdaBoost specifically, because Mujumdar & Vaidehi (2019) -- our
         primary ML source paper -- found AdaBoost was their best performer
         after pipelining (98.8% accuracy). Saving it separately lets the
         report make a direct comparison against the literature, regardless
         of whether AdaBoost happens to also be the best model this run.

    Any stale best_model_*.pkl files from a previous run (with a different
    winning model name) are removed first, so the models/ folder never ends
    up with confusing leftover files from an old run.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Clear out any previous best_model_*.pkl so stale files don't linger
    for f in os.listdir(MODEL_DIR):
        if f.startswith("best_model_") and f.endswith(".pkl"):
            os.remove(os.path.join(MODEL_DIR, f))

    best_model = models_fitted[best_model_name]
    best_metric_value = results_df.loc[results_df["Model"] == best_model_name, selection_metric].values[0]
    best_bundle = {"model": best_model, "scaler": scaler,
                   "feature_cols": FEATURE_COLS, "model_name": best_model_name,
                   "selection_metric": selection_metric,
                   "selection_metric_value": float(best_metric_value)}
    best_filename = f"best_model_{_safe_filename(best_model_name)}.pkl"
    best_path = os.path.join(MODEL_DIR, best_filename)
    joblib.dump(best_bundle, best_path)

    adaboost_model = models_fitted["AdaBoost"]
    adaboost_acc = results_df.loc[results_df["Model"] == "AdaBoost", "Accuracy"].values[0]
    adaboost_f1 = results_df.loc[results_df["Model"] == "AdaBoost", "F1"].values[0]
    adaboost_bundle = {"model": adaboost_model, "scaler": scaler,
                       "feature_cols": FEATURE_COLS, "model_name": "AdaBoost",
                       "accuracy_this_run": float(adaboost_acc),
                       "f1_this_run": float(adaboost_f1)}
    adaboost_path = os.path.join(MODEL_DIR, "adaboost_model.pkl")
    joblib.dump(adaboost_bundle, adaboost_path)

    print(f"[ml_pipeline] Saved best model by {selection_metric} "
          f"({best_model_name}, {selection_metric}={best_metric_value:.3f}) -> {best_path}")
    print(f"[ml_pipeline] Saved AdaBoost (Acc={adaboost_acc:.3f}, F1={adaboost_f1:.3f}) -> {adaboost_path}")

    if best_model_name != "AdaBoost":
        print(f"[ml_pipeline] NOTE: best model here ({best_model_name}) differs from "
              f"Mujumdar & Vaidehi (2019)'s best performer (AdaBoost, 98.8% accuracy "
              f"in their pipelined result). Worth discussing this discrepancy in the "
              f"report -- likely causes: different preprocessing choices, dataset "
              f"variant, selection metric (F1 vs Accuracy), or hyperparameter differences.")
    else:
        print(f"[ml_pipeline] AdaBoost is the best model here too -- consistent with "
              f"Mujumdar & Vaidehi (2019)'s finding. Good literature-agreement point "
              f"for the report.")


def load_saved_model(name="best"):
    """
    Loads a previously saved model bundle.
    name: 'best' (auto-finds whichever best_model_*.pkl is present,
    regardless of which model won) or 'adaboost'.
    """
    if name == "adaboost":
        path = os.path.join(MODEL_DIR, "adaboost_model.pkl")
    else:
        candidates = [f for f in os.listdir(MODEL_DIR) if f.startswith("best_model_") and f.endswith(".pkl")] \
            if os.path.isdir(MODEL_DIR) else []
        if not candidates:
            raise FileNotFoundError(f"No best_model_*.pkl found in {MODEL_DIR} -- run ml_pipeline.py first.")
        path = os.path.join(MODEL_DIR, candidates[0])

    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at {path} -- run ml_pipeline.py first.")
    bundle = joblib.load(path)
    print(f"[ml_pipeline] Loaded model: {bundle['model_name']} (from {os.path.basename(path)})")
    return bundle

def run_ml_pipeline():
    df = preprocess_pipeline()

    X_train, X_test, y_train, y_test, scaler = get_train_test(df)

    print("\n[ml_pipeline] === Cross-validation (5-fold) ===")
    cv_results = cross_validate_models(X_train, y_train, k=5)
    cv_results.to_csv(os.path.join(TABLE_DIR, "cv_results.csv"), index=False)

    print("\n[ml_pipeline] === Holdout evaluation ===")
    results_df, reports, roc_data = fit_and_evaluate_on_holdout(X_train, y_train, X_test, y_test)
    

    # refit all models (fit_and_evaluate already did, but we need the objects for plotting)
    models_fitted = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        models_fitted[name] = model

    plot_confusion_matrices(models_fitted, X_test, y_test)
    plot_roc_curves(roc_data)
    plot_accuracy_comparison(results_df)

    SELECTION_METRIC = "F1"
    results_df = results_df.sort_values(SELECTION_METRIC, ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]
    results_df.to_csv(os.path.join(TABLE_DIR, "holdout_results.csv"), index=False)
    print(f"[ml_pipeline] Selecting best model by {SELECTION_METRIC}: {best_model_name}")

    plot_feature_importance(best_model_name, models_fitted, X_test, y_test, FEATURE_COLS)

    save_models(models_fitted, scaler, best_model_name, results_df, selection_metric=SELECTION_METRIC)

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
