import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import shap
import os

from config import (
    DATA_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    PRIVILEGED_RACES,
    DEPRIVED_RACES,
    SHAP_SAMPLE_SIZE,
    SHAP_SAMPLE_RANDOM_STATE,
)

# Re-export config constants for backward compatibility
__all__ = [
    "DATA_PATH",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "TARGET",
    "PRIVILEGED_RACES",
    "DEPRIVED_RACES",
    "compute_metrics",
    "reverse_dummify",
    "reverse_scaling",
    "ml2gold",
    "save_model_comparison",
    "equal_opportunity_bias",
    "calculate_discrimination_bias",
    "save_shap_bar",
]


# ── Fairness functions (deduplicated) ────────────────────────────────────────

def equal_opportunity_bias(gold_df, sensitive_col, target_col, privileged_values, deprived_values, positive_label, model_pred):
    """
    Compute Equal Opportunity bias between privileged and deprived groups.
    Returns TPR for each group and the bias score (TPR_privileged - TPR_deprived).
    """
    privileged_group = gold_df[gold_df[sensitive_col].isin(privileged_values)]
    deprived_group = gold_df[gold_df[sensitive_col].isin(deprived_values)]

    tpr_privileged = (
        ((privileged_group[target_col] == positive_label) & (privileged_group[model_pred] == positive_label)).sum()
        / (privileged_group[target_col] == positive_label).sum()
    )
    tpr_deprived = (
        ((deprived_group[target_col] == positive_label) & (deprived_group[model_pred] == positive_label)).sum()
        / (deprived_group[target_col] == positive_label).sum()
    )

    bias_score = tpr_privileged - tpr_deprived

    return {
        "tpr_privileged": tpr_privileged,
        "tpr_deprived": tpr_deprived,
        "equal_opportunity_bias": bias_score,
    }


def calculate_discrimination_bias(df, sensitive_col, target_col, privileged_values, deprived_values, positive_label):
    """
    Compute discrimination bias as the difference in positive-label rates
    between privileged and deprived groups.
    """
    privileged_group = df[df[sensitive_col].isin(privileged_values)]
    prob_privileged = (privileged_group[target_col] == positive_label).mean()

    deprived_group = df[df[sensitive_col].isin(deprived_values)]
    prob_deprived = (deprived_group[target_col] == positive_label).mean()

    bias_score = prob_privileged - prob_deprived

    return {
        "prob_privileged": prob_privileged,
        "prob_deprived": prob_deprived,
        "discrimination_bias": bias_score,
    }


def save_shap_bar(shap_values, title, filename, output_dir):
    """
    Save a SHAP bar plot to *output_dir*/<filename>.png.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close()


# ── Metric computation ───────────────────────────────────────────────────────

def compute_metrics(df, model_prediction=None):
    """
    Compute and print recidivism rates by race, age category, and sex,
    and optionally compare with model predictions.
    """
    metrics_ethnic = df.groupby('race').agg({'is_recid':['sum', 'count']})
    metrics_ethnic['Rate'] = round(100 * metrics_ethnic[('is_recid', 'sum')] / metrics_ethnic[('is_recid', 'count')], 2)
    metrics_ethnic['Recid'] = metrics_ethnic[('is_recid', 'sum')]
    metrics_ethnic['Total'] = metrics_ethnic[('is_recid', 'count')]
    metrics_ethnic = metrics_ethnic[['Recid', 'Total', 'Rate']]

    if model_prediction is not None:
        metrics_ethnic_pred = df.groupby('race').agg({model_prediction:['sum', 'count']})
        metrics_ethnic_pred['Model_Rate'] = round(100 * metrics_ethnic_pred[(model_prediction, 'sum')] / metrics_ethnic_pred[(model_prediction, 'count')], 2)
        metrics_ethnic_pred['Model_Recid'] = metrics_ethnic_pred[(model_prediction, 'sum')]
        metrics_ethnic = metrics_ethnic.join(metrics_ethnic_pred, how='inner', on = 'race')
        metrics_ethnic = metrics_ethnic[['Recid', 'Model_Recid', 'Total', 'Rate', 'Model_Rate']]

    metrics_ethnic = metrics_ethnic.sort_values(by='Total', ascending=False)
    print(metrics_ethnic, "\n")

    df['age_category'] = df['age'].apply(lambda x: '18-25' if 18 <= x <= 25 else ('26-45' if 26 <= x <= 45 else ('46-65' if 46 <= x <= 65 else '66+')))

    metrics_age = df.groupby('age_category').agg({'is_recid':['sum', 'count']})
    metrics_age['Rate'] = round(100 * metrics_age[('is_recid', 'sum')] / metrics_age[('is_recid', 'count')], 2)
    metrics_age['Recid'] = metrics_age[('is_recid', 'sum')]
    metrics_age['Total'] = metrics_age[('is_recid', 'count')]
    metrics_age = metrics_age[['Recid', 'Total', 'Rate']]

    if model_prediction is not None:
        metrics_age_pred = df.groupby('age_category').agg({model_prediction:['sum', 'count']})
        metrics_age_pred['Model_Rate'] = round(100 * metrics_age_pred[(model_prediction, 'sum')] / metrics_age_pred[(model_prediction, 'count')], 2)
        metrics_age_pred['Model_Recid'] = metrics_age_pred[(model_prediction, 'sum')]
        metrics_age = metrics_age.join(metrics_age_pred, how='inner', on = 'age_category')
        metrics_age = metrics_age[['Recid', 'Model_Recid', 'Total', 'Rate', 'Model_Rate']]

    metrics_age = metrics_age.sort_values(by='Total', ascending=False)
    print(metrics_age, "\n")

    metrics_sex = df.groupby('sex').agg({'is_recid':['sum', 'count']})
    metrics_sex['Rate'] = round(100 * metrics_sex[('is_recid', 'sum')] / metrics_sex[('is_recid', 'count')], 2)
    metrics_sex['Recid'] = metrics_sex[('is_recid', 'sum')]
    metrics_sex['Total'] = metrics_sex[('is_recid', 'count')]
    metrics_sex = metrics_sex[['Recid', 'Total', 'Rate']]

    if model_prediction is not None:
        metrics_sex_pred = df.groupby('sex').agg({model_prediction:['sum', 'count']})
        metrics_sex_pred['Model_Rate'] = round(100 * metrics_sex_pred[(model_prediction, 'sum')] / metrics_sex_pred[(model_prediction, 'count')], 2)
        metrics_sex_pred['Model_Recid'] = metrics_sex_pred[(model_prediction, 'sum')]
        metrics_sex = metrics_sex.join(metrics_sex_pred, how='inner', on = 'sex')
        metrics_sex = metrics_sex[['Recid', 'Model_Recid', 'Total', 'Rate', 'Model_Rate']]

    metrics_sex = metrics_sex.sort_values(by='Total', ascending=False)
    print(metrics_sex)

    return metrics_ethnic, metrics_age, metrics_sex


# ── Data transformation helpers ──────────────────────────────────────────────

def reverse_dummify(df):
    analysis_df = df.copy()
    
    groups = {
        'sex': [c for c in df.columns if c.startswith('sex_')],
        'race': [c for c in df.columns if c.startswith('race_')],
        'c_charge_degree': [c for c in df.columns if c.startswith('c_charge_degree_')]
    }

    for target_col, dummy_cols in groups.items():
        analysis_df[target_col] = (analysis_df[dummy_cols]
                                   .idxmax(axis=1)
                                   .str.replace(f"{target_col}_", ""))

        analysis_df = analysis_df.drop(columns=dummy_cols)
    return analysis_df


def reverse_scaling(df, scaler):
    df[NUMERICAL_FEATURES] = scaler.inverse_transform(df[NUMERICAL_FEATURES])
    return df

def ml2gold(df, scaler):
    df = reverse_dummify(df)
    df = reverse_scaling(df, scaler)
    return df


# ── Visualization ────────────────────────────────────────────────────────────

def save_model_comparison(metrics_logreg, metrics_xgb, title_suffix="", path="training_output/images/"):
    """
    Generate and save comparative bar plots from the metrics DataFrames.
    Bar heights represent recidivism counts, labels show recidivism rates.
    """
    os.makedirs(path, exist_ok=True)

    labels = metrics_logreg.index
    real_rate = metrics_logreg['Rate']
    logreg_rate = metrics_logreg['Model_Rate']
    xgb_rate = metrics_xgb['Model_Rate']
    real_recid = metrics_logreg['Recid']
    logreg_recid = metrics_logreg['Model_Recid']
    xgb_recid = metrics_xgb['Model_Recid']
    
    x = np.arange(len(labels))  
    width = 0.25               
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - width, real_recid, width, label='Real', color='#34495e')
    rects2 = ax.bar(x, logreg_recid, width, label='LogReg', color='#3498db')
    rects3 = ax.bar(x + width, xgb_recid, width, label='XGBoost', color='#e67e22')
    
    ax.set_ylabel('Number of Recidivisms')
    ax.set_title(f'Rate Comparison : Real vs Models ({title_suffix})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    
    def autolabel(rects, rates):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            rate = rates.iloc[i]
            label_text = f'{rate:.1f}%'
            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1, real_rate)
    autolabel(rects2, logreg_rate)
    autolabel(rects3, xgb_rate)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'{title_suffix}_comparison.png'))
    plt.close()