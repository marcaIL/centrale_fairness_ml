# COMPAS Fairness Analysis Project

A machine learning project analyzing fairness in predictive models using the COMPAS recidivism dataset. This project trains naive models (Logistic Regression and XGBoost), applies bias mitigation via sample reweighting, and compares all models on both performance and fairness metrics.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Project Details](#project-details)

## Prerequisites

- **macOS or Linux**
- **Python 3.9+**
- **uv** (fast Python package installer and resolver)

Install uv if you don't have it:
```bash
# macOS (using Homebrew)
brew install uv

# Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or visit: https://github.com/astral-sh/uv
```

## Installation

1. **Clone or navigate to the project folder:**
```bash
cd centrale_fairness_ml
```

2. **Install dependencies using uv:**
```bash
uv sync
```

This command will:
- Create a virtual environment in `.venv/`
- Install all dependencies listed in `pyproject.toml`
- Resolve dependency versions using `uv.lock`

3. **Verify installation:**
```bash
uv run python --version
```

## Project Structure

```
centrale_fairness_ml/
├── README.md                               # This file
├── pyproject.toml                          # Project configuration and dependencies
├── uv.lock                                 # Locked dependency versions
├── run.sh                                  # Main entry point (runs all scripts)
├── __main__.py                             # Python entry point (calls run.sh)
├── explo.ipynb                             # Exploratory data analysis notebook
├── data/
│   └── compas-scores-two-years.csv        # COMPAS recidivism dataset
├── src/
│   ├── data_processing.py                 # Data preprocessing pipeline (DataPreprocessor)
│   ├── utils.py                           # Constants, metrics and plotting utilities
│   ├── models_naive_training.py           # Naive model training (with race)
│   ├── models_naive_no_race_training.py   # Naive model training (without race)
│   ├── bias_mitigation.py                 # Bias mitigation via sample reweighting
│   └── models_comparison.py              # Cross-model ROC curves comparison
```
## Usage

### Run all scripts

To run the full pipeline (training, bias mitigation, comparison):

```bash
./run.sh
```

The scripts are executed in the following order:
1. `models_naive_training.py` — trains models with race features
2. `models_naive_no_race_training.py` — trains models without race features
3. `bias_mitigation.py` — trains models with sample reweighting to mitigate racial bias
4. `models_comparison.py` — compares all 6 models with ROC curves

> **Note:** `models_comparison.py` requires the outputs of the three previous scripts (saved model weights and scalers).

### Explore the data

```bash
uv run jupyter notebook explo.ipynb
```

## Project Details

### Models

Three training strategies are applied, each with two classifiers:

| Strategy | Description | Output folder |
|---|---|---|
| Naive with race | Full feature set | `training_output/` |
| Naive without race | Race columns removed | `training_no_race_output/` |
| Bias mitigated | Sample reweighting based on discrimination bias | `training_mitigated_output/` |

**Classifiers:**
1. **Logistic Regression** — hyperparameter tuning on `C` via GridSearchCV (5-fold, F1 scoring)
2. **XGBoost** — hyperparameter tuning on `max_depth`, `learning_rate`, `n_estimators` via GridSearchCV (5-fold, F1 scoring)

### Bias Mitigation

The mitigation strategy is based on **discrimination bias** (difference in positive outcome rates between privileged and deprived groups):
- **Privileged group:** Caucasian, Asian, Hispanic, Other
- **Deprived group:** African-American, Native American
- Samples from the deprived group (positive label) are upweighted by `1 + bias_score`
- Samples from the privileged group (positive label) are downweighted by `1 - bias_score`

### Fairness Metrics

For each model, the following fairness metrics are computed across race, age category, and sex:
- Recidivism rates per group
- Model prediction rates per group
- **Equal opportunity bias** (TPR difference between privileged and deprived groups)

### Data

The COMPAS recidivism dataset includes:
- Demographic features: `age`, `sex`, `race`
- Criminal history: `juv_fel_count`, `juv_misd_count`, `juv_other_count`, `priors_count`
- Engineered features: `jail_duration`, `time_btw_offense_and_jail`
- Charge info: `c_charge_degree`
- Target: `is_recid` (binary recidivism label)

Train/test split: **50/50**, `random_state=1234`

### Outputs

Each training run produces:
- Saved model weights (`.save` via joblib)
- ROC curves (`.png`)
- Fairness metric charts by race, age, sex
- SHAP feature importance bar charts

The comparison script produces:
- `comparison_output/roc_curves_xgboost.png`
- `comparison_output/roc_curves_logistic_regression.png`
- `comparison_output/roc_curves_all_models.png`

## Dependencies

- `pandas` — data manipulation
- `scikit-learn` — models, preprocessing, GridSearchCV
- `xgboost` — gradient boosting classifier
- `matplotlib` — visualizations
- `shap` — feature importance and model interpretability
- `joblib` — model serialization
