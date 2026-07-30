# DBPS — Diabetes Prediction Using Mathematical Modelling and Machine Learning

A hybrid framework combining ODE-based mathematical modelling (individual-level glucose-insulin dynamics and population-level disease progression) with a machine learning classification pipeline for diabetes risk prediction.

---

## Project Structure

```
DBPS/
├── data/
│   ├── raw/              # place the real pima.csv here (see "Getting the dataset" below)
│   └── processed/        # cleaned data is written here automatically
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_ml_pipeline.ipynb
│   ├── 03_ode_individual_model.ipynb
│   └── 04_ode_population_model.ipynb
├── src/
│   ├── preprocessing.py       # data cleaning, imputation, EDA
│   ├── ml_pipeline.py         # ~12 classifiers, CV, confusion matrices, ROC, feature importance
│   ├── ode_individual.py      # glucose-insulin-food ODE model (Kumnungkit et al. 2022)
│   ├── ode_population.py      # compartmental ODE model, equilibrium/stability/sensitivity (Ferdous 2023)
│   └── utils.py
├── results/
│   ├── figures/           # all generated plots (17 figures across the full pipeline)
│   └── tables/            # all generated result tables (CSV/JSON/TXT)
├── docs/
│   ├── module-B-status.md     # weekly ML progress notes (for report-writer handoff)
│   └── module-C-status.md     # weekly ODE progress notes (for report-writer handoff)
├── report/                # put your Word/LaTeX report source here
├── requirements.txt
└── README.md
```

---

## Setup (macOS)

```bash
cd ~/projects/DBPS          # or wherever you cloned the repo

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional, needs internet) install extras for richer results:
pip install xgboost shap
```

## Getting the dataset

Download the **PIMA Indians Diabetes Dataset** (768 rows, 8 features + Outcome) from either:
- Kaggle: search "Pima Indians Diabetes Database"
- UCI ML Repository: "Pima Indians Diabetes"

Save the CSV as:
```
data/raw/pima.csv
```

**Without this file, the code still runs** — `preprocessing.py` automatically falls back to a synthetic placeholder dataset so you can test the pipeline end-to-end. You will see a clear `WARNING` printed when this happens. **Do not use synthetic-data results in the report** — only results generated after adding the real CSV are valid for submission.

Expected columns: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

---

## Running the project

Run modules directly:
```bash
cd src
python3 preprocessing.py       # cleans data, generates EDA figures
python3 ml_pipeline.py         # runs the full ML pipeline, saves handoff metrics
python3 ode_individual.py      # individual glucose-insulin simulation
python3 ode_population.py      # extended population model (run this AFTER ml_pipeline.py
                                # so it can pick up ml_handoff_metrics.json)
```

Or open the notebooks in Jupyter (recommended for the report — screenshot-friendly):
```bash
pip install jupyter
jupyter notebook notebooks/
```
Run them in order: `01 → 02 → 03 → 04` (04 depends on 02's output).

All figures land in `results/figures/`, all tables in `results/tables/`.

---

## What each module does

| Module | What it is | Key output |
|---|---|---|
| `preprocessing.py` | Cleans PIMA data (handles impossible zero values), runs EDA | Correlation heatmap, distributions, pairplot |
| `ml_pipeline.py` | Trains/evaluates ~12 classifiers with 5-fold CV | Accuracy/Precision/Recall/F1/AUC comparison, confusion matrices, ROC curves, feature importance |
| `ode_individual.py` | Simulates one patient's glucose-insulin response to meals | ML classifier's prediction selects which literature-validated parameter regime (normal/diabetic) to simulate |
| `ode_population.py` | Simulates disease progression across a population over 20 years | Equilibrium point, stability (eigenvalues), sensitivity analysis, and an ML-informed screening scenario |

**The key coupling point (this is the project's central contribution):** the ML classifier's recall/sensitivity on held-out data motivates — but does not precisely determine — a plausible *range* of increased treatment-detection rates, which is then simulated in the population model as a scenario comparison. This is deliberately framed as a range, not an invented exact conversion — see the docstring in `ode_population.py` for the full reasoning, and make sure this framing carries through into the report chapter.

---

## Validation note (worth stating explicitly in the report)

The extended population model's equilibrium point and eigenvalues were derived independently (symbolically, via sympy) and **exactly match the published values in Ferdous (2023)**, and the sensitivity indices match her Table 4 to three decimal places. This is strong evidence the model implementation is mathematically correct, not just "code that runs."

---

## Known limitations to state in the report

1. PIMA is cross-sectional — it cannot be used to *fit* the individual-level ODE model, only to select which literature-derived regime to simulate (see `ode_individual.py` docstring).
2. This environment substitutes XGBoost→GradientBoosting, SHAP→permutation importance, and Keras ANN→sklearn MLPClassifier due to no-internet constraints during initial development. If you have internet access, swap these in for potentially stronger results.
3. The ML-informed population scenario uses a *range* of plausible treatment-rate increases, not a mathematically derived exact value — state this explicitly to avoid overclaiming.
