import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

# Vars 
DATA_PATH = "data/compas-scores-two-years.csv"

NUMERICAL_FEATURES = [
                      'age', 
                      'juv_fel_count', 
                      'juv_misd_count', 
                      'juv_other_count', 
                      'priors_count', 
                      'jail_duration', 
                      'time_btw_offense_and_jail'
                      ]

CATEGORICAL_FEATURES = [
                        'sex_Female', 
                        'sex_Male',
                        'c_charge_degree_F', 
                        'c_charge_degree_M', 
                        'race_African-American',
                        'race_Asian', 
                        'race_Caucasian', 
                        'race_Hispanic', 
                        'race_Native American',
                        'race_Other'
                        ]

TARGET = 'is_recid'

# Utils
def compute_metrics(df, model_prediction = None): 
    """
    Compute and print recidivism rates by race, age category, and sex, and optionally compare with model predictions.
    Args:       
        - df (pd.DataFrame): The input DataFrame containing the data.
        - model_prediction (str, optional): The column name of the model's predictions to compare against. Defaults to None.
    Returns:        
        None: Prints the computed metrics to the console.
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

def save_model_comparison(metrics_logreg, metrics_xgb, title_suffix="", path = "training_output/images/"):
    """
    Generate and save comparative bar plots from the metrics DataFrames.
    Bar heights represent recidivism counts, labels show recidivism rates.
    """
    # Data extraction
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
    
    # Barplots creation - heights represent recidivism counts
    rects1 = ax.bar(x - width, real_recid, width, label='Real', color='#34495e')
    rects2 = ax.bar(x, logreg_recid, width, label='LogReg', color='#3498db')
    rects3 = ax.bar(x + width, xgb_recid, width, label='XGBoost', color='#e67e22')
    
    # Titles and labels
    ax.set_ylabel('Number of Recidivisms')
    ax.set_title(f'Rate Comparison : Real vs Models ({title_suffix})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    
    # Adding rate labels on top of the bars
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
    plt.savefig(f'{path}{title_suffix}_comparison.png')
    plt.close()