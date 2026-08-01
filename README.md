<div align="center">

# 🩺 DBPS — Diabetes Prediction Using Mathematical Modelling & Machine Learning

**A hybrid framework linking differential-equation disease dynamics with supervised classification for diabetes risk prediction.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-ODE%20Solvers-8CAAE6?style=flat&logo=scipy&logoColor=white)
![SymPy](https://img.shields.io/badge/SymPy-Symbolic%20Math-3B5526?style=flat)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)

</div>

---

## What this project actually does

Most diabetes-prediction projects pick a lane: either a classifier trained on tabular patient data, *or* a differential-equation model of disease progression. This project builds **both**, at **two different scales**, and — critically — makes them talk to each other through real code rather than parallel chapters that never meet:

| Layer | Scale | Method | What it answers |
|---|---|---|---|
| 🧬 **Individual** | One patient | 3-equation nonlinear ODE system | *"What does this patient's glucose-insulin response look like over a day?"* |
| 🌍 **Population** | A whole cohort | 5-compartment ODE system | *"How does the diabetic population evolve over 20 years under different interventions?"* |
| 🤖 **Machine Learning** | Cross-sectional data | 12-classifier comparison | *"Given clinical features, is this patient diabetic?"* |

The ML classifier's real, measured performance (its recall/sensitivity on held-out data) is used to select which physiological regime the individual ODE model simulates for a given patient, **and** to bound a realistic *range* of outcomes in the population model's early-screening scenario — a genuine, quantitative coupling between the classification layer and both mathematical models, not just a shared report.

---

## The Mathematics

### 1. Individual-level: glucose-insulin-food dynamics

Adapted from Kumnungkit et al. (2022), a minimal nonlinear ODE system tracking intestinal glucose $Q(t)$, blood glucose $G(t)$, and insulin $I(t)$:

$$
\frac{dQ}{dt} = -\beta Q + \frac{\eta \, D(t)}{\gamma^2 + Q^2}
$$

$$
\frac{dG}{dt} = R_0 - \big(E_{G0} + S_I I\big) G + k_Q Q
$$

$$
\frac{dI}{dt} = \frac{I_{max} G^2}{\alpha_{hill} + G^2} - k_I I
$$

where $D(t)$ is a meal-intake function, $S_I$ is insulin sensitivity, and $E_{G0}$ is glucose effectiveness. Two literature-validated parameter regimes (**normal** / **diabetic**) are simulated, with the ML classifier's prediction selecting which regime applies to a given patient.

> **A real bug caught here:** an early version had the insulin term as $+S_I I G$ (positive), causing an unphysical runaway feedback loop where glucose diverged to non-physiological values. The correct sign is negative — insulin *clears* glucose — fixed and documented in `src/ode_individual.py`.

### 2. Population-level: compartmental disease progression

Extending Ferdous (2023), a 5-compartment system (Susceptible, Affected, Treated, healthy-Lifestyle, Prevented):

$$
\frac{dS}{dt} = b - (\mu + \varepsilon + \alpha)S
$$

$$
\frac{dA}{dt} = \varepsilon S - (\tau + \mu + \delta_1)A + \gamma L
$$

$$
\frac{dT}{dt} = \tau A - (\mu + \delta_2)T
$$

$$
\frac{dL}{dt} = \alpha S - (\gamma + \mu + \beta)L
$$

$$
\frac{dP}{dt} = \beta L - \mu P
$$

For this system, `src/ode_population.py` derives (symbolically, via `sympy`):
- The **equilibrium point** $\varphi(S^*, A^*, T^*, L^*, P^*)$
- The **Jacobian matrix** and its **eigenvalues**, to classify local stability
- The **normalized forward sensitivity index**:

$$
\Gamma^v_\rho = \frac{\partial v}{\partial \rho} \cdot \frac{\rho}{v}
$$

  for every state variable $v$ against every parameter $\rho$

> **Validated, not just computed:** the equilibrium point and eigenvalues derived here match Ferdous (2023)'s published values *exactly*, and the sensitivity indices match her Table 4 to three decimal places — computed completely independently, not copied. Strong evidence the implementation is mathematically correct.

### 3. The ML ↔ ODE coupling

The classifier's recall/sensitivity motivates — but does **not** precisely determine — a bounded *range* of screening-driven treatment-rate increases in the population model (τ scenarios at +10%/+30%/+50%), rather than an invented exact conversion. This framing is deliberate: see [Key Results](#key-results-so-far) below for why.

---

## Key Results (so far)

- **Best model: AdaBoost**, selected by F1 (not raw accuracy) — `Accuracy = 0.760, Precision = 0.681, Recall = 0.593, F1 = 0.634, AUC = 0.818`
- This **matches Mujumdar & Vaidehi (2019)**'s own finding that AdaBoost was their best-performing classifier — independent literature agreement, not a coincidence to gloss over.
- **Why F1 over accuracy matters here**: the model's recall (59.3%) is meaningfully weaker than its accuracy suggests — roughly 4 in 10 actual diabetic patients would be missed by this classifier. Confirmed at the individual level too: a 24-real-patient batch simulation showed 12/12 correct on non-diabetics but only 6/12 correct on diabetics, closely matching the model's true test-set recall.
- This recall gap is **exactly why** the ML-informed population scenario is framed as a bounded range rather than a guaranteed effect — the model's imperfect sensitivity is the concrete justification, not just caution for its own sake.

---

## Project Structure

```
DBPS/
├── data/
│   ├── raw/pima.csv          # real PIMA dataset (768 rows)
│   └── processed/
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_ml_pipeline.ipynb
│   ├── 03_ode_individual_model.ipynb
│   └── 04_ode_population_model.ipynb
├── src/
│   ├── preprocessing.py       # cleaning, imputation, EDA
│   ├── ml_pipeline.py         # 12 classifiers, CV, ROC, feature importance, model persistence
│   ├── ode_individual.py      # glucose-insulin-food ODE + batch real-patient simulation
│   ├── ode_population.py      # compartmental ODE, equilibrium/stability/sensitivity, ML-informed scenario
│   └── utils.py
├── models/                    # saved trained models (joblib .pkl), auto-named by winning model
├── results/
│   ├── figures/                # 18 generated plots
│   └── tables/                 # CV results, holdout results, equilibrium point, eigenvalues,
│                                # sensitivity indices, batch-patient summary, ML handoff metrics
├── docs/                      # (currently empty — for project notes/logs if you add any)
├── report/                    # Word/LaTeX report source
├── requirements.txt
└── README.md
```

---

## Setup (macOS)

```bash
cd ~/projects/DBPS

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# optional, needs internet: richer alternatives to the built-in substitutes
pip install xgboost shap
```

### Getting the dataset

Download the **PIMA Indians Diabetes Dataset** (Kaggle or UCI ML Repository) and save as `data/raw/pima.csv`. Expected columns:
`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

If this file is missing, the code still runs end-to-end on a synthetic placeholder (with a clear warning printed) — useful for testing, **never for reportable results**.

---

## Running the project

Run in this order — later modules depend on earlier outputs (the ML pipeline must run before the individual/population ODE modules, since both load its saved model / handoff metrics):

```bash
cd src
python3 preprocessing.py       # cleans data, generates EDA figures
python3 ml_pipeline.py         # full ML pipeline, saves best model + handoff metrics
python3 ode_individual.py      # individual glucose-insulin simulation + real-patient batch demo
python3 ode_population.py      # extended population model + ML-informed scenario
```

Or, for report-ready screenshots, open the matching notebooks in order (`01 → 02 → 03 → 04`) in Jupyter or VS Code.

All figures land in `results/figures/`, all tables in `results/tables/`, trained models in `models/`.

---

## Future Work

- **Fuzzy logic risk scoring** — a Mamdani fuzzy inference system producing a Fuzzy Diabetes Risk Score (FDRS) as an additional engineered ML feature. Deliberately descoped from the current build to keep ODE + ML as a clean, defensible centerpiece within the project timeline; a natural next extension.
- **Addressing the recall gap** — SMOTE or class-weighting to counter PIMA's class imbalance (~500 non-diabetic vs. ~268 diabetic), which likely contributes to the model's weaker sensitivity.
- **A second dataset** (e.g. Sylhet Diabetes Dataset) for a generalization check beyond PIMA.

---

<div align="center">

*Built on Ferdous (2023), Akinsola & Oluyo (2014), Kumnungkit et al. (2022), and Mujumdar & Vaidehi (2019) — extended, validated, and connected.*

</div>