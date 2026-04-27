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
- **uv** 

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
│   ├── models_comparison.py              # Cross-model ROC curves comparison
│   ├── privacy_attacks/
│       ├── aia_utils.py                          # Shared helpers for the AIA
│       ├── attribute_inference_race.py           # AIA on race (3 training variants)
│       └── defense_output_perturbation.py        # Output perturbation defense (naive only)
│   └── adversarial_attacks/
│       ├── evasion_utils.py                      # PGD attack + SmoothedModel wrapper
│       ├── evasion_attack.py                     # Evasion attack on naive models (race stratified)
│       └── evasion_defenses.py                   # Adv. training vs randomized smoothing
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
5. `privacy_attacks/attribute_inference_race.py` — black-box AIA on `race` against the 6 models
6. `privacy_attacks/defense_output_perturbation.py` — output perturbation defense applied to the naive models, with privacy/utility trade-off
7. `adversarial_attacks/evasion_attack.py` — PGD evasion attack on naive models, stratified by race
8. `adversarial_attacks/evasion_defenses.py` — adversarial training vs randomized smoothing defense comparison

> **Note:** `models_comparison.py` and the privacy scripts require the outputs of the three training scripts (saved model weights and scalers).

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
- `adversarial-robustness-toolbox` — privacy attacks toolkit (AIA reference)

## Privacy Attacks

### Attribute Inference Attack (AIA) on `race`

A black-box attacker observes the non-race features of a record plus the model's
positive-class probability, and tries to predict whether the record belongs to
the **privileged** race group or the **deprived** group.

The attack is run on the 6 models (LogReg + XGBoost × naive / no_race / mitigated).

Outputs:
- `privacy_output/reports/aia_race_results.csv`
- `privacy_output/images/aia_race_comparison.png`

### Defense — Output Perturbation (naive models only)

Four perturbation strategies are applied to the naive models' `predict_proba`:
`none`, `label_only`, `rounding` (k=2,1,0), `laplace` noise (b=0.05–0.30).

Outputs:
- `privacy_output/reports/defense_perturbation_results.csv`
- `privacy_output/images/privacy_utility_tradeoff.png`

## Adversarial Attacks

### Evasion Attack (PGD)

A PGD evasion attack perturbs numerical features within an L∞ epsilon-ball to
flip the model prediction from `recid=1` to `recid=0`. Gradients are estimated
via finite differences (works for both LogReg and XGBoost).

The attack is stratified by race group to measure **differential vulnerability**.

Outputs:
- `adversarial_output/reports/evasion_attack_results.csv`
- `adversarial_output/images/evasion_attack_results.png`

### Defenses: Adversarial Training vs Randomized Smoothing

Two defenses are compared on the naive models:

- **Adversarial Training**: retrain on original + adversarial examples (ε=0.3)
- **Randomized Smoothing**: average predictions over 30 Gaussian-noised copies (σ=0.25)

For each defense, the PGD attack is re-run at multiple epsilon values.

Outputs:
- `adversarial_output/reports/defense_comparison_results.csv`
- `adversarial_output/images/defense_comparison.png`

