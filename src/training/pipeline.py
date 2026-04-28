"""
Shared training pipeline for COMPAS models.
Encapsulates: data loading, scaling, GridSearchCV, evaluation, SHAP,
fairness metrics, and report saving.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import (
    GRIDSEARCH_CV,
    GRIDSEARCH_SCORING,
    LOGREG_MAX_ITER,
    NUMERICAL_FEATURES,
    PARAM_GRID_LOGREG,
    PARAM_GRID_XGB,
    PRIVILEGED_RACES,
    DEPRIVED_RACES,
    SHAP_SAMPLE_RANDOM_STATE,
    SHAP_SAMPLE_SIZE,
    SPLIT_RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    XGB_FIXED_PARAMS,
    OutputPaths,
)
from utils import (
    compute_metrics,
    equal_opportunity_bias,
    ml2gold,
    save_model_comparison,
    save_shap_bar,
)


# ── Configuration dataclass for a training run ───────────────────────────────

@dataclass
class TrainingConfig:
    """All knobs for a single training run."""

    output: OutputPaths
    drop_race_features: bool = False
    sample_weights: Optional[pd.Series] = field(default=None, repr=False)

    # Override-able but sane defaults from config.py
    param_grid_logreg: Dict = field(default_factory=lambda: dict(PARAM_GRID_LOGREG))
    param_grid_xgb: Dict = field(default_factory=lambda: dict(PARAM_GRID_XGB))
    test_size: float = TEST_SIZE
    split_random_state: int = SPLIT_RANDOM_STATE


# ── Pipeline class ───────────────────────────────────────────────────────────

class TrainingPipeline:
    """Runs a full train + evaluate + report cycle."""

    def __init__(self, mlready_df: pd.DataFrame, config: TrainingConfig):
        self.config = config
        self._ensure_dirs()

        # Train/test split (shared random state → same split across variants)
        self.mlready_train_df, self.mlready_test_df = train_test_split(
            mlready_df, test_size=config.test_size, random_state=config.split_random_state
        )

        # Scale numerical features
        self.scaler = StandardScaler()
        self.mlready_train_df[NUMERICAL_FEATURES] = self.scaler.fit_transform(
            self.mlready_train_df[NUMERICAL_FEATURES]
        )
        self.mlready_test_df[NUMERICAL_FEATURES] = self.scaler.transform(
            self.mlready_test_df[NUMERICAL_FEATURES]
        )
        joblib.dump(self.scaler, os.path.join(config.output.scalers_dir, "scaler_original_dataset.save"))

        # Feature matrices
        train_df = self.mlready_train_df
        test_df = self.mlready_test_df

        if config.drop_race_features:
            race_cols = [c for c in train_df.columns if c.startswith("race_")]
            train_df = train_df.drop(columns=race_cols)
            test_df = test_df.drop(columns=race_cols)

        extra_drop = ["weight"] if "weight" in train_df.columns else []
        self.X_train = train_df.drop(columns=[TARGET] + extra_drop)
        self.y_train = train_df[TARGET]
        self.X_test = test_df.drop(columns=[TARGET] + extra_drop)
        self.y_test = self.mlready_test_df[TARGET]

        self.sample_weights = config.sample_weights

        # Populated after run()
        self.logistic_model = None
        self.xgb_model = None
        self.y_hat_logreg = None
        self.y_hat_xgb = None
        self.proba_logreg = None
        self.proba_xgb = None

    # ── Public API ────────────────────────────────────────────────────────

    def run(self) -> "TrainingPipeline":
        """Execute the full pipeline and return *self*."""
        self._train_logistic_regression()
        self._train_xgboost()
        self._save_classification_reports()
        self._save_hyperparams()
        self._save_roc_curves()
        self._evaluate_fairness()
        self._evaluate_equal_opportunity_bias()
        self._compute_and_save_shap()
        return self

    # ── Training ──────────────────────────────────────────────────────────

    def _train_logistic_regression(self):
        fit_params = {}
        if self.sample_weights is not None:
            fit_params["sample_weight"] = self.sample_weights

        grid = GridSearchCV(
            LogisticRegression(),
            self.config.param_grid_logreg,
            cv=GRIDSEARCH_CV,
            scoring=GRIDSEARCH_SCORING,
        )
        grid.fit(self.X_train, self.y_train, **fit_params)
        print(f"best parameter Logistic Regression : {grid.best_params_}")

        self.logistic_model = LogisticRegression(C=grid.best_params_["C"], max_iter=LOGREG_MAX_ITER)
        self.logistic_model.fit(self.X_train, self.y_train, **fit_params)
        self.y_hat_logreg = self.logistic_model.predict(self.X_test)
        self.proba_logreg = self.logistic_model.predict_proba(self.X_test)[:, 1]
        print(classification_report(self.y_test, self.y_hat_logreg))
        joblib.dump(self.logistic_model, os.path.join(self.config.output.models_dir, "logreg_naive.save"))

        self._best_params_logreg = grid.best_params_

    def _train_xgboost(self):
        fit_params = {}
        if self.sample_weights is not None:
            fit_params["sample_weight"] = self.sample_weights

        grid = GridSearchCV(
            XGBClassifier(),
            self.config.param_grid_xgb,
            cv=GRIDSEARCH_CV,
            scoring=GRIDSEARCH_SCORING,
        )
        grid.fit(self.X_train, self.y_train, **fit_params)
        print(f"best parameters XGB : {grid.best_params_}")

        self.xgb_model = XGBClassifier(
            n_estimators=grid.best_params_["n_estimators"],
            learning_rate=grid.best_params_["learning_rate"],
            max_depth=grid.best_params_["max_depth"],
            **XGB_FIXED_PARAMS,
        )
        self.xgb_model.fit(self.X_train, self.y_train, **fit_params)
        self.y_hat_xgb = self.xgb_model.predict(self.X_test)
        self.proba_xgb = self.xgb_model.predict_proba(self.X_test)[:, 1]
        print(classification_report(self.y_test, self.y_hat_xgb))
        joblib.dump(self.xgb_model, os.path.join(self.config.output.models_dir, "xgb_naive.save"))

        self._best_params_xgb = grid.best_params_

    # ── Reports ───────────────────────────────────────────────────────────

    def _save_classification_reports(self):
        reports_dir = self.config.output.reports_dir
        report_lr = classification_report(self.y_test, self.y_hat_logreg, output_dict=True)
        report_xgb = classification_report(self.y_test, self.y_hat_xgb, output_dict=True)
        pd.DataFrame(report_lr).transpose().to_csv(os.path.join(reports_dir, "classification_report_logreg.csv"))
        pd.DataFrame(report_xgb).transpose().to_csv(os.path.join(reports_dir, "classification_report_xgb.csv"))

    def _save_hyperparams(self):
        auc_lr = roc_auc_score(self.y_test, self.proba_logreg)
        auc_xgb = roc_auc_score(self.y_test, self.proba_xgb)
        rows = [
            {"model": "logreg", "C": self._best_params_logreg["C"], "auc": auc_lr},
            {
                "model": "xgb",
                "max_depth": self._best_params_xgb["max_depth"],
                "learning_rate": self._best_params_xgb["learning_rate"],
                "n_estimators": self._best_params_xgb["n_estimators"],
                "auc": auc_xgb,
            },
        ]
        pd.DataFrame(rows).to_csv(os.path.join(self.config.output.reports_dir, "hyperparameters.csv"), index=False)

    def _save_roc_curves(self):
        fpr_lr, tpr_lr, _ = roc_curve(self.y_test, self.proba_logreg)
        fpr_xgb, tpr_xgb, _ = roc_curve(self.y_test, self.proba_xgb)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_lr, tpr_lr, label="Logistic Regression")
        plt.plot(fpr_xgb, tpr_xgb, label="XGBoost")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(self.config.output.images_dir, "roc_curves.png"))
        plt.close()

    # ── Fairness ──────────────────────────────────────────────────────────

    def _evaluate_fairness(self):
        gold_test_df = ml2gold(self.mlready_test_df, self.scaler)

        gold_test_df["logistic_pred"] = self.y_hat_logreg
        metrics_lr = compute_metrics(gold_test_df, model_prediction="logistic_pred")

        gold_test_df["xgb_pred"] = self.y_hat_xgb
        metrics_xgb = compute_metrics(gold_test_df, model_prediction="xgb_pred")

        img_dir = self.config.output.images_dir
        for idx, suffix in enumerate(["race", "age", "sex"]):
            save_model_comparison(metrics_lr[idx], metrics_xgb[idx], title_suffix=suffix, path=img_dir)

        # Store for later use
        self._gold_test_df = gold_test_df

    def _evaluate_equal_opportunity_bias(self):
        gold = self._gold_test_df
        common_kwargs = dict(
            sensitive_col="race",
            target_col="is_recid",
            privileged_values=PRIVILEGED_RACES,
            deprived_values=DEPRIVED_RACES,
            positive_label=0,
        )

        results_lr = equal_opportunity_bias(gold, model_pred="logistic_pred", **common_kwargs)
        results_xgb = equal_opportunity_bias(gold, model_pred="xgb_pred", **common_kwargs)

        for tag, res in [("logistic regression", results_lr), ("XGBoost", results_xgb)]:
            print(f"Privileged group probability {tag}: {res['tpr_privileged']:.2f}")
            print(f"Deprived group probability {tag}: {res['tpr_deprived']:.2f}")
            print(f"Discrimination bias {tag}: {res['equal_opportunity_bias']:.2f}")

        bias_df = pd.DataFrame([{"model": "logreg", **results_lr}, {"model": "xgb", **results_xgb}])
        bias_df.to_csv(os.path.join(self.config.output.reports_dir, "equal_opportunity_bias.csv"), index=False)

        self._eob_results = {"logreg": results_lr, "xgb": results_xgb}

    # ── SHAP ──────────────────────────────────────────────────────────────

    def _compute_and_save_shap(self):
        sample = self.X_test.sample(min(SHAP_SAMPLE_SIZE, len(self.X_test)), random_state=SHAP_SAMPLE_RANDOM_STATE)
        img_dir = self.config.output.images_dir

        explainer_lr = shap.Explainer(self.logistic_model, self.X_train)
        sv_lr = explainer_lr(sample)
        save_shap_bar(sv_lr, "Global Feature Importance - Logistic Regression", "shap_logreg", img_dir)

        explainer_xgb = shap.Explainer(self.xgb_model)
        sv_xgb = explainer_xgb(sample)
        save_shap_bar(sv_xgb, "Global Feature Importance - XGBoost", "shap_xgb", img_dir)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _ensure_dirs(self):
        out = self.config.output
        for d in [out.images_dir, out.models_dir, out.reports_dir, out.scalers_dir]:
            os.makedirs(d, exist_ok=True)

