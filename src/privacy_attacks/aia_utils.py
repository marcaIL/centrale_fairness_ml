import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from data_processing import DataPreprocessor
from utils import DATA_PATH, NUMERICAL_FEATURES, TARGET
import joblib

# Same groups as in bias_mitigation.py
PRIVILEGED_RACES = ['Caucasian', 'Asian', 'Hispanic', 'Other']
DEPRIVED_RACES = ['African-American', 'Native American']

RACE_COLUMNS = [
    'race_African-American', 'race_Asian', 'race_Caucasian',
    'race_Hispanic', 'race_Native American', 'race_Other'
]
PRIVILEGED_RACE_COLS = [f'race_{r}' for r in PRIVILEGED_RACES]


def load_test_data(scaler_path='training_output/scalers/scaler_original_dataset.save'):
    """Load COMPAS and return the standardized test set + binary race target."""
    raw_df = pd.read_csv(DATA_PATH)
    mlready_df, _ = DataPreprocessor(raw_df=raw_df).preprocess()
    _, mlready_test_df = train_test_split(mlready_df, test_size=0.5, random_state=1234)

    scaler = joblib.load(scaler_path)
    mlready_test_df = mlready_test_df.copy()
    mlready_test_df[NUMERICAL_FEATURES] = scaler.transform(mlready_test_df[NUMERICAL_FEATURES])

    y_test = mlready_test_df[TARGET]
    X_test_with_race = mlready_test_df.drop(columns=[TARGET])
    X_test_no_race = X_test_with_race.drop(columns=RACE_COLUMNS)

    # 1 = privileged, 0 = deprived
    race_priv = X_test_with_race[PRIVILEGED_RACE_COLS].sum(axis=1).astype(int)

    return X_test_with_race, X_test_no_race, y_test, race_priv


def run_aia(proba_fn, X_for_model, X_attacker, race_priv, random_state=42):
    """
    Black-box AIA: attacker sees non-race features + model proba, tries to predict race_priv.
    Also computes a baseline (features only, no model output).
    """
    proba_target = proba_fn(X_for_model)

    # Attacker features = non-race features + model probability
    attacker_features = X_attacker.copy()
    attacker_features['model_proba'] = proba_target

    Xa_train, Xa_eval, ya_train, ya_eval = train_test_split(
        attacker_features, race_priv,
        test_size=0.5, stratify=race_priv, random_state=random_state
    )

    # Full attack (features + model proba)
    attacker = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    attacker.fit(Xa_train, ya_train)
    proba_attack = attacker.predict_proba(Xa_eval)[:, 1]
    pred_attack = (proba_attack >= 0.5).astype(int)

    # Baseline: features only, no model output
    Xb_train = Xa_train.drop(columns=['model_proba'])
    Xb_eval = Xa_eval.drop(columns=['model_proba'])
    baseline = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    baseline.fit(Xb_train, ya_train)
    proba_baseline = baseline.predict_proba(Xb_eval)[:, 1]
    pred_baseline = (proba_baseline >= 0.5).astype(int)

    return {
        'aia_accuracy': accuracy_score(ya_eval, pred_attack),
        'aia_balanced_accuracy': balanced_accuracy_score(ya_eval, pred_attack),
        'aia_auc': roc_auc_score(ya_eval, proba_attack),
        'baseline_accuracy': accuracy_score(ya_eval, pred_baseline),
        'baseline_balanced_accuracy': balanced_accuracy_score(ya_eval, pred_baseline),
        'baseline_auc': roc_auc_score(ya_eval, proba_baseline),
        'n_eval': int(len(ya_eval)),
        'pct_privileged_eval': float(ya_eval.mean()),
    }

