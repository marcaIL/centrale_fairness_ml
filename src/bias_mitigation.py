"""
Bias-mitigated training: reweights samples to reduce discrimination bias,
then trains Logistic Regression and XGBoost with sample weights.
Also compares equal opportunity bias before / after mitigation.
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    NUMERICAL_FEATURES,
    PRIVILEGED_RACES,
    DEPRIVED_RACES,
    SPLIT_RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    TRAINING_MITIGATED_OUTPUT,
)
from data_processing import DataPreprocessor
from training.pipeline import TrainingConfig, TrainingPipeline
from utils import (
    calculate_discrimination_bias,
    equal_opportunity_bias,
    ml2gold,
)


# ── Data loading ──────────────────────────────────────────────────────────────

raw_df = pd.read_csv(DATA_PATH)
data_preprocessor = DataPreprocessor(raw_df=raw_df)
mlready_df, gold_df = data_preprocessor.preprocess()

# Same split as naive training (same random state)
ml_ready_train_df, ml_ready_test_df = train_test_split(
    mlready_df, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE
)
gold_train_df = gold_df.loc[ml_ready_train_df.index]


# ── Discrimination bias on training data ──────────────────────────────────────

results = calculate_discrimination_bias(
    gold_train_df,
    sensitive_col="race",
    target_col="is_recid",
    privileged_values=PRIVILEGED_RACES,
    deprived_values=DEPRIVED_RACES,
    positive_label=0,
)
print(f"Privileged group probability (w): {results['prob_privileged']:.2f}")
print(f"Deprived group probability (b): {results['prob_deprived']:.2f}")
print(f"Discrimination bias: {results['discrimination_bias']:.2f}")


# ── Bias mitigation via reweighting ──────────────────────────────────────────

def bias_mitigation(gold_df, ml_ready_df, sensitive_col, target_col, privileged_values, deprived_values, positive_label):
    res = calculate_discrimination_bias(gold_df, sensitive_col, target_col, privileged_values, deprived_values, positive_label)
    bias_score = res["discrimination_bias"]

    ml_ready_df["weight"] = 1.0
    if bias_score > 0:
        privileged_mask = (
            (ml_ready_df["race_Caucasian"] == 1) | (ml_ready_df["race_Asian"] == 1)
            | (ml_ready_df["race_Hispanic"] == 1) | (ml_ready_df["race_Other"] == 1)
        ) & (ml_ready_df[target_col] == positive_label)
        deprived_mask = (
            (ml_ready_df["race_African-American"] == 1) | (ml_ready_df["race_Native American"] == 1)
        ) & (ml_ready_df[target_col] == positive_label)
        ml_ready_df.loc[privileged_mask, "weight"] = 1 - bias_score
        ml_ready_df.loc[deprived_mask, "weight"] = 1 + bias_score

    return ml_ready_df


ml_ready_train_df_mitigated = bias_mitigation(
    gold_train_df,
    ml_ready_train_df,
    sensitive_col="race",
    target_col="is_recid",
    privileged_values=PRIVILEGED_RACES,
    deprived_values=DEPRIVED_RACES,
    positive_label=0,
)

sample_weights = ml_ready_train_df_mitigated["weight"].values
# Drop weight column before passing to pipeline (pipeline handles the rest)
mlready_df_with_weight = mlready_df.copy()
# We need to inject weights into the pipeline via the config


# ── Run training pipeline with sample weights ────────────────────────────────
# The pipeline will re-split the data with the same random state, so we need
# to add the weight column to the full df *before* passing it in, then
# the pipeline drops it automatically.

mlready_df["weight"] = 1.0
mlready_df.loc[ml_ready_train_df_mitigated.index, "weight"] = ml_ready_train_df_mitigated["weight"]

config = TrainingConfig(
    output=TRAINING_MITIGATED_OUTPUT,
    drop_race_features=False,
    sample_weights=sample_weights,
)
pipeline = TrainingPipeline(mlready_df.drop(columns=["weight"]), config)
pipeline.run()


# ── Before / after mitigation comparison ──────────────────────────────────────

gold_test_df = pipeline._gold_test_df

# Load naive models (before mitigation) and compare
logreg_naive = joblib.load("training_output/models_weights/logreg_naive.save")
xgb_naive = joblib.load("training_output/models_weights/xgb_naive.save")

logreg_naive_pred = logreg_naive.predict(pipeline.X_test)
xgb_naive_pred = xgb_naive.predict(pipeline.X_test)

gold_test_df["logreg_naive_pred"] = logreg_naive_pred
gold_test_df["xgb_naive_pred"] = xgb_naive_pred

common_kwargs = dict(
    sensitive_col="race",
    target_col="is_recid",
    privileged_values=PRIVILEGED_RACES,
    deprived_values=DEPRIVED_RACES,
    positive_label=0,
)

results_logreg_naive = equal_opportunity_bias(gold_test_df, model_pred="logreg_naive_pred", **common_kwargs)
results_xgb_naive = equal_opportunity_bias(gold_test_df, model_pred="xgb_naive_pred", **common_kwargs)

print("\n--- BEFORE MITIGATION ---")
for tag, res in [("logistic regression (naive)", results_logreg_naive), ("XGBoost (naive)", results_xgb_naive)]:
    print(f"Privileged group probability {tag}: {res['tpr_privileged']:.2f}")
    print(f"Deprived group probability {tag}: {res['tpr_deprived']:.2f}")
    print(f"Discrimination bias {tag}: {res['equal_opportunity_bias']:.2f}")

# Save before/after comparison
comparison_df = pd.DataFrame([
    {"model": "logreg_naive", **results_logreg_naive},
    {"model": "xgb_naive", **results_xgb_naive},
    {"model": "logreg_mitigated", **pipeline._eob_results["logreg"]},
    {"model": "xgb_mitigated", **pipeline._eob_results["xgb"]},
])
os.makedirs(TRAINING_MITIGATED_OUTPUT.reports_dir, exist_ok=True)
comparison_df.to_csv(os.path.join(TRAINING_MITIGATED_OUTPUT.reports_dir, "bias_before_after_comparison.csv"), index=False)

# Save discrimination bias
discrim_df = pd.DataFrame([results])
discrim_df.to_csv(os.path.join(TRAINING_MITIGATED_OUTPUT.reports_dir, "discrimination_bias.csv"), index=False)
