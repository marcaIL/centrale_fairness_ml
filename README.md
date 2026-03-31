# COMPAS Fairness Analysis Project

A machine learning project analyzing fairness in predictive models using the COMPAS recidivism dataset. This project compares naive models (Logistic Regression and XGBoost) and evaluates their performance and fairness characteristics.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Project Details](#project-details)

## Prerequisites

- **macOS, Linux, or Windows**
- **Python 3.9+**
- **uv** (fast Python package installer and resolver)

Install uv if you don't have it:
```bash
# macOS (using Homebrew)
brew install uv

# Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (using PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or visit: https://github.com/astral-sh/uv
```

## Installation

1. **Clone or navigate to the project folder:**
```bash
cd project-compas-based
```

2. **Install dependencies using uv:**
```bash
uv sync
```

This command will:
- Create a virtual environment in `.venv/`
- Install all dependencies listed in `pyproject.toml`
- Resolve dependency versions automatically

3. **Verify installation:**
```bash
uv run python --version
```

## Project Structure

```
project-compas-based/
├── README.md                          # This file
├── pyproject.toml                     # Project configuration and dependencies
├── main.py                            # Main entry point
├── explo.ipynb                        # Exploratory data analysis notebook
├── data/
│   └── compas-scores-two-years.csv   # COMPAS recidivism dataset
├── src/
│   ├── models_naive_training.py      # Naive model training (LogReg + XGBoost)
│   ├── data_processing.py            # Data preprocessing pipeline
│   └── utils.py                       # Utility functions and constants
├── training_output/
│   ├── models_weights/               # Saved model weights
│   └── images/                       # Generated visualizations
```

## Usage

### Run the naive model training

Train baseline models (Logistic Regression and XGBoost) and generate ROC curves:

```bash
uv run src/models_naive_training.py
```

**Output:**
- Trained models saved to `training_output/models_weights/`
  - `logreg_naive.pkl` (Logistic Regression)
  - `xgb_naive.ubj` (XGBoost)
- ROC curves visualization: `training_output/images/roc_curves.png`
- Classification reports printed to console

### Explore the data

Open the Jupyter notebook for exploratory data analysis:

```bash
uv run jupyter notebook explo.ipynb
```

## Project Details

### Models

1. **Logistic Regression**
   - Hyperparameter tuning with GridSearchCV (C parameter)
   - Cross-validation: 5-fold with F1 scoring

2. **XGBoost**
   - Hyperparameter tuning (max_depth, learning_rate, n_estimators)
   - Cross-validation: 5-fold with F1 scoring
   - Regularization: subsample=0.8, colsample_bytree=0.8

### Data

The COMPAS recidivism dataset contains:
- Defendant demographic information
- Criminal history features
- Recidivism outcome (binary target)
- Subject to fairness analysis across demographic groups

### Evaluation

Models are evaluated using:
- **Classification Report**: precision, recall, F1-score
- **ROC Curves**: AUC and model discrimination

## Dependencies

- `pandas` - Data manipulation and analysis
- `scikit-learn` - Machine learning models and preprocessing
- `xgboost` - Gradient boosting classifier
- `matplotlib` - Data visualization
- `shap` - Model interpretability (for fairness analysis)

## Notes

- All data preprocessing is handled by the `DataPreprocessor` class
- Numerical features are standardized before model training
- Models are trained on 50% of the data, tested on 50%
- Training output directories are created automatically if they don't exist
