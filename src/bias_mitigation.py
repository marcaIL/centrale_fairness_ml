import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from data_processing import DataPreprocessor
from utils import DATA_PATH, NUMERICAL_FEATURES, TARGET, ml2gold, compute_metrics, save_model_comparison
import shap

def calculate_discrimination_bias(df, sensitive_col, target_col, privileged_values, deprived_values, positive_label):
    
    privileged_group = df[df[sensitive_col].isin(privileged_values)]
    prob_privileged = (privileged_group[target_col] == positive_label).mean()
    
    deprived_group = df[df[sensitive_col].isin(deprived_values)]
    prob_deprived = (deprived_group[target_col] == positive_label).mean()
    
    bias_score = prob_privileged - prob_deprived
    
    return {
        "prob_privileged": prob_privileged,
        "prob_deprived": prob_deprived,
        "discrimination_bias": bias_score
    }


#Load data
raw_df = pd.read_csv(DATA_PATH)
data_preprocessor = DataPreprocessor(raw_df = raw_df)
mlready_df, gold_df = data_preprocessor.preprocess()

#to keep the same dataset used for training naives models
ml_ready_train_df, ml_ready_test_df = train_test_split(mlready_df, test_size=0.5, random_state=1234)
gold_train_df = gold_df.loc[ml_ready_train_df.index]

#vars
privileged_values_race = ['Caucasian', 'Asian', 'Hispanic', 'Other']
deprived_values_race = ['African-American', 'Native American']

#Race discrimination bias 
results = calculate_discrimination_bias(
    gold_train_df, 
    sensitive_col='race', 
    target_col='is_recid', 
    privileged_values=privileged_values_race, 
    deprived_values=deprived_values_race, 
    positive_label=0
)

print(f"Probabilité groupe privilégié (w): {results['prob_privileged']:.2f}")
print(f"Probabilité groupe défavorisé (b): {results['prob_deprived']:.2f}")
print(f"Biais de discrimination: {results['discrimination_bias']:.2f}")


def bias_mitigation(gold_df, ml_ready_df, sensitive_col, target_col, privileged_values, deprived_values, positive_label):
    # Calculate the bias score
    results = calculate_discrimination_bias(gold_df, sensitive_col, target_col, privileged_values, deprived_values, positive_label)
    bias_score = results['discrimination_bias']
    
    # If there is a bias, apply mitigation
    if bias_score > 0:
        # Mitigate bias by reweighting the samples
        ml_ready_df['weight'] = 1.0
        privileged_mask = ((ml_ready_df['race_Caucasian'] == 1) | (ml_ready_df['race_Asian'] == 1) | (ml_ready_df['race_Hispanic'] == 1) | (ml_ready_df['race_Other'] == 1)) & (ml_ready_df[target_col] == positive_label)
        deprived_mask = ((ml_ready_df['race_African-American'] == 1) | (ml_ready_df['race_Native American'] == 1)) & (ml_ready_df[target_col] == positive_label)
        ml_ready_df.loc[privileged_mask, 'weight'] = 1 - bias_score
        ml_ready_df.loc[deprived_mask, 'weight'] = 1 + bias_score
    else:
        ml_ready_df['weight'] = 1.0
    
    return ml_ready_df

# Apply bias mitigation
ml_ready_train_df_mitigated = bias_mitigation(
    gold_train_df,
    ml_ready_train_df,
    sensitive_col='race',
    target_col='is_recid',
    privileged_values=privileged_values_race, 
    deprived_values=deprived_values_race,
    positive_label=0)

#Normalize features (use mitigated training data)
scaler = StandardScaler()
ml_ready_train_df_mitigated[NUMERICAL_FEATURES] = scaler.fit_transform(ml_ready_train_df_mitigated[NUMERICAL_FEATURES])
ml_ready_test_df[NUMERICAL_FEATURES] = scaler.transform(ml_ready_test_df[NUMERICAL_FEATURES])

#X,y for train and test
X_train = ml_ready_train_df_mitigated.drop(columns=[TARGET, 'weight'])
y_train = ml_ready_train_df_mitigated[TARGET]
sample_weights_train = ml_ready_train_df_mitigated['weight'].values

X_test = ml_ready_test_df.drop(columns=[TARGET])
y_test = ml_ready_test_df[TARGET]

#Hyperparameter tuning for Logistic Regression
param_grid_logreg = {
    'C': [0.01, 0.1, 0.25, 0.5, 1, 2, 4, 10]
}

grid_logreg = GridSearchCV(LogisticRegression(), param_grid_logreg, cv=5, scoring='f1')
grid_logreg.fit(X_train, y_train, sample_weight=sample_weights_train)
print(f"best parameter Logistic Regression : {grid_logreg.best_params_}")

#Logistic regression training and evaluation
logistic_model = LogisticRegression(C = grid_logreg.best_params_['C'], max_iter=1000)
logistic_model.fit(X_train, y_train, sample_weight=sample_weights_train)
y_hat_logreg = logistic_model.predict(X_test)
proba_logreg = logistic_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_hat_logreg))
os.makedirs("training_mitigated_output/models_weights", exist_ok=True)
joblib.dump(logistic_model, 'training_mitigated_output/models_weights/logreg_naive.save')

#Hyperparameter tuning for XGBoost
param_grid_xgb = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.005],
    'n_estimators': [50, 80, 100, 500]
}

grid_xgb = GridSearchCV(XGBClassifier(), param_grid_xgb, cv=5, scoring='f1')
grid_xgb.fit(X_train, y_train, sample_weight=sample_weights_train)
print(f"best parameters XGB : {grid_xgb.best_params_}")

#XGBoost training and evaluation
xgb_classifier = XGBClassifier(
    n_estimators=grid_xgb.best_params_['n_estimators'],
    learning_rate=grid_xgb.best_params_['learning_rate'],
    max_depth=grid_xgb.best_params_['max_depth'],
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)

xgb_classifier.fit(X_train, y_train, sample_weight=sample_weights_train) 
y_hat_xgb = xgb_classifier.predict(X_test)
proba_xgb = xgb_classifier.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_hat_xgb))
os.makedirs("training_mitigated_output/models_weights", exist_ok=True)
joblib.dump(xgb_classifier, 'training_mitigated_output/models_weights/xgb_naive.save')

#ROC curves
logreg_roc = roc_curve(y_test, proba_logreg)
xgb_roc = roc_curve(y_test, proba_xgb)
plt.figure(figsize=(8, 6))
plt.plot(logreg_roc[0], logreg_roc[1], label='Logistic Regression')
plt.plot(xgb_roc[0], xgb_roc[1], label='XGBoost')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid()
plt.legend()
os.makedirs("training_mitigated_output/images", exist_ok=True)
plt.savefig('training_mitigated_output/images/roc_curves.png')

#Fairness metrics 
gold_test_df = ml2gold(ml_ready_test_df, scaler)

gold_test_df['logistic_pred'] = y_hat_logreg
metrics_logreg = compute_metrics(gold_test_df, model_prediction='logistic_pred')

gold_test_df['xgb_pred'] = y_hat_xgb
metrics_xgb = compute_metrics(gold_test_df, model_prediction='xgb_pred')

# Race
save_model_comparison(metrics_logreg[0], 
                      metrics_xgb[0], 
                      title_suffix="race",
                      path = "training_mitigated_output/images/")

# Age
save_model_comparison(metrics_logreg[1], 
                      metrics_xgb[1], 
                      title_suffix="age",
                      path = "training_mitigated_output/images/")

# Sex
save_model_comparison(metrics_logreg[2],
                      metrics_xgb[2],
                      title_suffix="sex",
                      path = "training_mitigated_output/images/")

#Equal opportunity bias
def equal_opportunity_bias(gold_df, sensitive_col, target_col, privileged_values, deprived_values, positive_label, model_pred):
    privileged_group = gold_df[gold_df[sensitive_col].isin(privileged_values)]
    deprived_group = gold_df[gold_df[sensitive_col].isin(deprived_values)]
    
    tpr_privileged = ((privileged_group[target_col] == positive_label) & (privileged_group[model_pred] == positive_label)).sum() / (privileged_group[target_col] == positive_label).sum()
    tpr_deprived = ((deprived_group[target_col] == positive_label) & (deprived_group[model_pred] == positive_label)).sum() / (deprived_group[target_col] == positive_label).sum()
    
    bias_score = tpr_privileged - tpr_deprived
    
    return {
        "tpr_privileged": tpr_privileged,
        "tpr_deprived": tpr_deprived,
        "equal_opportunity_bias": bias_score
    }

results_log_reg = equal_opportunity_bias(
    gold_test_df, 
    sensitive_col='race',   
    target_col='is_recid',
    privileged_values=privileged_values_race,
    deprived_values=deprived_values_race,
    positive_label=0,
    model_pred = "logistic_pred"
)

results_xgb = equal_opportunity_bias(
    gold_test_df, 
    sensitive_col='race',   
    target_col='is_recid',
    privileged_values=privileged_values_race,
    deprived_values=deprived_values_race,
    positive_label=0,
    model_pred = 'xgb_pred'
)

print(f"Probabilité groupe privilégié logistic regression: {results_log_reg['tpr_privileged']:.2f}")
print(f"Probabilité groupe défavorisé logistic regression: {results_log_reg['tpr_deprived']:.2f}")
print(f"Biais de discrimination logistic regression: {results_log_reg['equal_opportunity_bias']:.2f}")

print(f"Probabilité groupe privilégié XGBoost: {results_xgb['tpr_privileged']:.2f}")
print(f"Probabilité groupe défavorisé XGBoost: {results_xgb['tpr_deprived']:.2f}")
print(f"Biais de discrimination XGBoost: {results_xgb['equal_opportunity_bias']:.2f}")

#shap values 
X_test_sample = X_test.sample(min(500, len(X_test)), random_state=42)

explainer_logreg = shap.Explainer(logistic_model, X_train) 
shap_values_logreg = explainer_logreg(X_test_sample)

explainer_xgb = shap.Explainer(xgb_classifier)
shap_values_xgb = explainer_xgb(X_test_sample)


def save_shap_bar(shap_values, title, filename):
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'training_mitigated_output/images/{filename}.png')
    plt.close()

save_shap_bar(shap_values_logreg, "Global Feature Importance - Logistic Regression", "shap_logreg")
save_shap_bar(shap_values_xgb, "Global Feature Importance - XGBoost", "shap_xgb")