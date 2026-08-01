"""
ode_individual.py
------------------
Individual-level glucose-insulin-food dynamics, adapted from Kumnungkit et
al. (2022), "Universal Minimal Model for Glucose-Insulin Relationship with
the Influence of Food Dynamic."

CONCEPTUAL NOTE (state this in the report methodology section):
PIMA is cross-sectional (one snapshot per patient, no time-series data), so
it cannot be used to fit this ODE model directly. Instead, the ML
classifier's predicted class (0/1) selects which literature-derived
parameter regime (Normal vs Diabetic, from Kumnungkit et al. Table 1) is
simulated, and the patient's real fasting glucose value from PIMA seeds the
initial condition G(0). This is an illustrative, risk-stratified simulation
-- a standard technique in clinical decision-support modelling -- not a
per-patient fitted model.

Note on notation: Kumnungkit's Hill-function constant is renamed
`alpha_hill` here to avoid clashing with the population model's `alpha`
(healthy-lifestyle adoption rate) used elsewhere in the project.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
TABLE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "tables")

# Literature-derived parameter sets (Kumnungkit et al. 2022, Table 1,
# Minimal Model columns) -- NOT fitted from PIMA.
PARAMS = {
    "normal": dict(beta=6.050, gamma=12.86, eta=4.086, R0=2.1, EG0=1e-3,
                   SI=3.06e-3, alpha_hill=1e4, Imax=0.28, kI=0.01, kQ=0.098),
    "diabetic": dict(beta=1.778, gamma=7.089, eta=4.641, R0=2.5, EG0=2.5e-3,
                      SI=1.14e-3, alpha_hill=1e4, Imax=0.93, kI=0.06, kQ=0.026),
}


def food_intake(t, meals=((0, 15, 50000), (240, 255, 40000), (480, 495, 45000))):
    """Step-function meal input D(t). `meals` is a tuple of
    (start_min, end_min, magnitude) windows -- default simulates
    breakfast/lunch/dinner-style spaced meals over an 8-hour window."""
    for start, end, magnitude in meals:
        if start <= t <= end:
            return float(magnitude)
    return 0.0


def glucose_insulin_food_model(t, y, p):
    """
    NOTE ON A CORRECTED SIGN (documented here deliberately -- flag this in
    your report's methodology/limitations, it's good practice to show you
    caught and fixed this):
    The source PDF's equation for dG/dt was garbled by OCR extraction. The
    correct minimal-model form (standard Bergman-style glucose-insulin
    dynamics, consistent with Kumnungkit et al.'s parameter definitions --
    EG0 = "glucose effectiveness", SI = "insulin SENSITIVITY", both of which
    are physiologically glucose-CLEARANCE terms) is:
        dG/dt = R0 - [EG0 + SI*I]*G + kQ*Q
    i.e. the insulin term must be NEGATIVE (higher insulin -> lower glucose).
    Using +SI*I*G instead created an unphysical positive feedback loop
    (glucose and insulin amplifying each other without bound) -- caught via
    the sanity check that glucose diverged to non-physiological values.
    """
    Q, G, I = y
    D = food_intake(t)
    dQ = -p["beta"] * Q + (p["eta"] * D) / (p["gamma"] ** 2 + Q ** 2)
    dG = p["R0"] - (p["EG0"] + p["SI"] * I) * G + p["kQ"] * Q
    dI = p["Imax"] * G ** 2 / (p["alpha_hill"] + G ** 2) - p["kI"] * I
    return [dQ, dG, dI]


def simulate_regime(regime, glucose0, insulin0=10.0, t_span=(0, 600), n_points=600):
    """Simulates one parameter regime ('normal' or 'diabetic') starting from
    a given real glucose value."""
    p = PARAMS[regime]
    y0 = [0.0, glucose0, insulin0]
    t_eval = np.linspace(*t_span, n_points)
    sol = solve_ivp(glucose_insulin_food_model, t_span, y0, args=(p,),
                     t_eval=t_eval, method="RK45", max_step=1.0)
    return sol


def simulate_for_patient(glucose0, predicted_class, **kwargs):
    """Convenience wrapper: predicted_class (0/1) from the ML classifier
    selects the regime, real glucose0 seeds the initial condition."""
    regime = "diabetic" if int(predicted_class) == 1 else "normal"
    sol = simulate_regime(regime, glucose0, **kwargs)
    return sol, regime


def plot_regime_comparison(glucose0_normal=100, glucose0_diabetic=160):
    """Side-by-side comparison of Normal vs Diabetic glucose-insulin
    response to the same meal schedule, starting from representative
    real-world fasting glucose values."""
    os.makedirs(FIG_DIR, exist_ok=True)

    sol_normal = simulate_regime("normal", glucose0_normal)
    sol_diabetic = simulate_regime("diabetic", glucose0_diabetic)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].plot(sol_normal.t, sol_normal.y[1], label="Normal regime", linewidth=2, color="tab:green")
    axes[0].plot(sol_diabetic.t, sol_diabetic.y[1], label="Diabetic regime", linewidth=2, color="tab:red")
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_ylabel("Glucose G(t) (mg/dl)")
    axes[0].set_title("Simulated Blood Glucose Response")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(sol_normal.t, sol_normal.y[2], label="Normal regime", linewidth=2, color="tab:green")
    axes[1].plot(sol_diabetic.t, sol_diabetic.y[2], label="Diabetic regime", linewidth=2, color="tab:red")
    axes[1].set_xlabel("Time (minutes)")
    axes[1].set_ylabel("Insulin I(t) (\u03bcU/ml)")
    axes[1].set_title("Simulated Insulin Response")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    plt.suptitle("Individual-Level Glucose-Insulin-Food Dynamics: Normal vs Diabetic Regime", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "09_individual_regime_comparison.png"), dpi=130)
    plt.close()
    print(f"[ode_individual] Regime comparison plot saved to {FIG_DIR}")


def plot_patient_demo(glucose0, predicted_class, patient_label="Test Patient"):
    """Full single-patient demo: real glucose value + ML prediction ->
    simulate -> plot, with the real value marked for reference."""
    os.makedirs(FIG_DIR, exist_ok=True)
    sol, regime = simulate_for_patient(glucose0, predicted_class)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(sol.t, sol.y[1], linewidth=2, color="tab:blue")
    axes[0].axhline(glucose0, color="gray", linestyle="--", label=f"PIMA fasting glucose = {glucose0}")
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_ylabel("Glucose (mg/dl)")
    axes[0].set_title("Glucose")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(sol.t, sol.y[2], linewidth=2, color="tab:orange")
    axes[1].set_xlabel("Time (minutes)")
    axes[1].set_ylabel("Insulin (\u03bcU/ml)")
    axes[1].set_title("Insulin")
    axes[1].grid(True, alpha=0.4)

    plt.suptitle(f"{patient_label} — Predicted class: {regime} (ML-selected regime)", fontsize=13)
    plt.tight_layout()
    fname = os.path.join(FIG_DIR, "10_single_patient_demo.png")
    plt.savefig(fname, dpi=130)
    plt.close()
    print(f"[ode_individual] Single-patient demo saved to {fname}")
    return sol, regime


def summary_table():
    """Saves a small table comparing peak glucose / time-to-peak / AUC-like
    summary statistic between regimes -- useful, quotable numbers for the
    report rather than only a plot."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    rows = []
    for regime, g0 in [("normal", 100), ("diabetic", 160)]:
        sol = simulate_regime(regime, g0)
        G = sol.y[1]
        peak_val = G.max()
        peak_time = sol.t[np.argmax(G)]
        auc_like = np.trapezoid(G, sol.t)
        rows.append({"Regime": regime, "Initial Glucose": g0,
                      "Peak Glucose": round(peak_val, 1),
                      "Time to Peak (min)": round(peak_time, 1),
                      "Glucose AUC (approx.)": round(auc_like, 1)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLE_DIR, "individual_model_summary.csv"), index=False)
    print("[ode_individual] Summary table:\n", df.to_string(index=False))
    return df

def batch_patient_demo(n_patients=8, save=True):
    """
    Runs the individual ODE simulation across several REAL patients (not
    just one), using the ML pipeline's saved best model to predict each
    patient's class, then selecting the corresponding ODE regime.

    This gives the individual-level section comparable depth to the
    population-level section instead of resting on a single-patient demo,
    and is a genuine code-level link between Module B (ML) and Module C
    (individual ODE) -- the saved model is actually loaded and used here,
    not just referenced in the report narrative.

    Prerequisites: run ml_pipeline.py first (so a saved best_model_*.pkl
    exists).
    """
    import sys
    sys.path.append(os.path.dirname(__file__))
    from ml_pipeline import load_saved_model
    from preprocessing import preprocess_pipeline, TARGET_COL

    bundle = load_saved_model("best")
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    df = preprocess_pipeline(save_processed=False)

    # Class-balanced sample: half actually-diabetic, half actually-not,
    # so the grid shows both cases (and any misclassifications are visible)
    n_each = n_patients // 2
    sample_pos = df[df[TARGET_COL] == 1].sample(
        n=min(n_each, (df[TARGET_COL] == 1).sum()), random_state=1)
    sample_neg = df[df[TARGET_COL] == 0].sample(
        n=min(n_each, (df[TARGET_COL] == 0).sum()), random_state=1)
    sample = pd.concat([sample_pos, sample_neg]).reset_index(drop=True)

    X_sample_scaled = scaler.transform(sample[feature_cols].values)
    predicted = model.predict(X_sample_scaled)

    n = len(sample)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    rows = []
    for i, (idx, patient) in enumerate(sample.iterrows()):
        glucose0 = patient["Glucose"]
        actual = int(patient[TARGET_COL])
        pred = int(predicted[i])
        sol, regime = simulate_for_patient(glucose0, pred)

        ax = axes[i]
        ax.plot(sol.t, sol.y[1], linewidth=2,
                color="tab:red" if pred == 1 else "tab:green")
        ax.axhline(glucose0, color="gray", linestyle="--", linewidth=1)
        match = "correct" if pred == actual else "MISCLASSIFIED"
        ax.set_title(f"Patient {idx} | Actual={actual} Pred={pred} ({match})", fontsize=10)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Glucose (mg/dl)")
        ax.grid(True, alpha=0.4)

        rows.append({
            "PatientIndex": idx, "RealGlucose": glucose0,
            "ActualOutcome": actual, "PredictedOutcome": pred,
            "RegimeUsed": regime, "Correct": pred == actual,
            "PeakGlucose": round(sol.y[1].max(), 1),
        })

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Individual-Level Glucose Simulation Across Real Patients (ML-Predicted Regime)", fontsize=14)
    plt.tight_layout()
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIG_DIR, "18_batch_patient_grid.png"), dpi=130)
    plt.close()

    summary = pd.DataFrame(rows)
    if save:
        os.makedirs(TABLE_DIR, exist_ok=True)
        summary.to_csv(os.path.join(TABLE_DIR, "batch_patient_summary.csv"), index=False)

    accuracy_on_sample = summary["Correct"].mean()
    print(f"[ode_individual] Batch demo: {n} real patients, "
          f"{accuracy_on_sample:.0%} correctly classified in this sample")
    print(summary.to_string(index=False))

    return summary

if __name__ == "__main__":
    plot_regime_comparison()
    plot_patient_demo(glucose0=148, predicted_class=1, patient_label="Example Patient A")
    summary_table()
    batch_patient_demo(n_patients=24)
