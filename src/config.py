"""
Centralized configuration for the COMPAS Fairness ML project.
All paths, hyperparameters, random seeds, group definitions, and epsilon values
are defined here to avoid magic numbers and hardcoded strings across scripts.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


# ── Data paths ────────────────────────────────────────────────────────────────

DATA_PATH = "data/compas-scores-two-years.csv"


# ── Feature definitions ───────────────────────────────────────────────────────

NUMERICAL_FEATURES: List[str] = [
    "age",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "jail_duration",
    "time_btw_offense_and_jail",
]

CATEGORICAL_FEATURES: List[str] = [
    "sex_Female",
    "sex_Male",
    "c_charge_degree_F",
    "c_charge_degree_M",
    "race_African-American",
    "race_Asian",
    "race_Caucasian",
    "race_Hispanic",
    "race_Native American",
    "race_Other",
]

RACE_COLUMNS: List[str] = [c for c in CATEGORICAL_FEATURES if c.startswith("race_")]

TARGET = "is_recid"


# ── Group definitions (fairness) ─────────────────────────────────────────────

PRIVILEGED_RACES: List[str] = ["Caucasian", "Asian", "Hispanic", "Other"]
DEPRIVED_RACES: List[str] = ["African-American", "Native American"]


# ── Random seeds ──────────────────────────────────────────────────────────────

SPLIT_RANDOM_STATE = 1234
SHAP_SAMPLE_RANDOM_STATE = 42


# ── Train/test split ─────────────────────────────────────────────────────────

TEST_SIZE = 0.5


# ── Hyperparameter grids ─────────────────────────────────────────────────────

PARAM_GRID_LOGREG: Dict[str, Any] = {
    "C": [0.01, 0.1, 0.25, 0.5, 1, 2, 4, 10],
}

PARAM_GRID_XGB: Dict[str, Any] = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.1, 0.01, 0.005],
    "n_estimators": [50, 80, 100, 500],
}

XGB_FIXED_PARAMS: Dict[str, Any] = {
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
}

LOGREG_MAX_ITER = 1000
GRIDSEARCH_CV = 5
GRIDSEARCH_SCORING = "f1"

SHAP_SAMPLE_SIZE = 500


# ── Output directory configs ─────────────────────────────────────────────────

@dataclass
class OutputPaths:
    """Output directory structure for a training run."""
    base_dir: str
    images_dir: str = ""
    models_dir: str = ""
    reports_dir: str = ""
    scalers_dir: str = ""

    def __post_init__(self):
        self.images_dir = f"{self.base_dir}/images"
        self.models_dir = f"{self.base_dir}/models_weights"
        self.reports_dir = f"{self.base_dir}/reports"
        self.scalers_dir = f"{self.base_dir}/scalers"


TRAINING_OUTPUT = OutputPaths(base_dir="training_output")
TRAINING_NO_RACE_OUTPUT = OutputPaths(base_dir="training_no_race_output")
TRAINING_MITIGATED_OUTPUT = OutputPaths(base_dir="training_mitigated_output")
COMPARISON_OUTPUT_DIR = "comparison_output"
PRIVACY_OUTPUT_DIR = "privacy_output"
ADVERSARIAL_OUTPUT_DIR = "adversarial_output"


# ── Privacy / adversarial epsilons ────────────────────────────────────────────

EVASION_EPSILONS: List[float] = [0.01, 0.05, 0.1, 0.2, 0.5]
PERTURBATION_EPSILONS: List[float] = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

