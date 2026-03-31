import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
import os
from utils import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET, DATA_PATH, compute_metrics, ml2gold, save_model_comparison 
from data_processing import DataPreprocessor
import shap     

# Load data and preprocess
raw_df = pd.read_csv(DATA_PATH)
data_preprocessor = DataPreprocessor(raw_df=raw_df)
mlready_df, _ = data_preprocessor.preprocess()

# Split into train and test (same as training)
mlready_train_df, mlready_test_df = train_test_split(mlready_df, test_size=0.5, random_state=1234)

# Load and apply scaler
scaler = joblib.load('training_output/scalers/scaler_original_dataset.save')
mlready_test_df[NUMERICAL_FEATURES] = scaler.transform(mlready_test_df[NUMERICAL_FEATURES])

# Prepare test data with race
X_test = mlready_test_df.drop(columns=[TARGET])
y_test = mlready_test_df[TARGET]

# Prepare test data without race
X_test_no_race = mlready_test_df.drop(columns=[TARGET, 'race_African-American', 'race_Asian', 'race_Caucasian', 'race_Hispanic', 'race_Native American', 'race_Other'])

# Load models
log_reg_model = joblib.load('training_output/models_weights/logreg_naive.save')
xgb_model = joblib.load('training_output/models_weights/xgb_naive.save')
log_reg_no_race_model = joblib.load('training_no_race_output/models_weights/logreg_naive.save')
xgb_no_race_model = joblib.load('training_no_race_output/models_weights/xgb_naive.save')
log_reg_mitigated_model = joblib.load('training_mitigated_output/models_weights/logreg_naive.save')
xgb_mitigated_model = joblib.load('training_mitigated_output/models_weights/xgb_naive.save')

# Generate probabilities
proba_logreg = log_reg_model.predict_proba(X_test)[:, 1]
proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
proba_logreg_no_race = log_reg_no_race_model.predict_proba(X_test_no_race)[:, 1]
proba_xgb_no_race = xgb_no_race_model.predict_proba(X_test_no_race)[:, 1]
proba_logreg_mitigated = log_reg_mitigated_model.predict_proba(X_test)[:, 1]
proba_xgb_mitigated = xgb_mitigated_model.predict_proba(X_test)[:, 1]

# Compute ROC curves
roc_logreg = roc_curve(y_test, proba_logreg)
roc_xgb = roc_curve(y_test, proba_xgb)
roc_logreg_no_race = roc_curve(y_test, proba_logreg_no_race)
roc_xgb_no_race = roc_curve(y_test, proba_xgb_no_race)
roc_logreg_mitigated = roc_curve(y_test, proba_logreg_mitigated)
roc_xgb_mitigated = roc_curve(y_test, proba_xgb_mitigated)

# Calculate AUC values
auc_logreg = auc(roc_logreg[0], roc_logreg[1])
auc_xgb = auc(roc_xgb[0], roc_xgb[1])
auc_logreg_no_race = auc(roc_logreg_no_race[0], roc_logreg_no_race[1])
auc_xgb_no_race = auc(roc_xgb_no_race[0], roc_xgb_no_race[1])
auc_logreg_mitigated = auc(roc_logreg_mitigated[0], roc_logreg_mitigated[1])
auc_xgb_mitigated = auc(roc_xgb_mitigated[0], roc_xgb_mitigated[1])

# Plot 1: XGBoost Models Comparison
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(roc_xgb[0], roc_xgb[1], label=f'XGBoost with Race (AUC: {auc_xgb:.4f})', linewidth=2.5, color='#1f77b4')
ax.plot(roc_xgb_no_race[0], roc_xgb_no_race[1], label=f'XGBoost without Race (AUC: {auc_xgb_no_race:.4f})', linewidth=2.5, color='#ff7f0e', linestyle='--')
ax.plot(roc_xgb_mitigated[0], roc_xgb_mitigated[1], label=f'XGBoost Mitigated (AUC: {auc_xgb_mitigated:.4f})', linewidth=2.5, color='#2ca02c', linestyle='-.')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('XGBoost Models - ROC Curves Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=11)
plt.tight_layout()
os.makedirs("comparison_output", exist_ok=True)
plt.savefig('comparison_output/roc_curves_xgboost.png', dpi=150)
print("XGBoost ROC curves saved to 'comparison_output/roc_curves_xgboost.png'")
plt.show()

# Plot 2: Logistic Regression Models Comparison
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(roc_logreg[0], roc_logreg[1], label=f'Logistic Regression with Race (AUC: {auc_logreg:.4f})', linewidth=2.5, color='#2ca02c')
ax.plot(roc_logreg_no_race[0], roc_logreg_no_race[1], label=f'Logistic Regression without Race (AUC: {auc_logreg_no_race:.4f})', linewidth=2.5, color='#d62728', linestyle='--')
ax.plot(roc_logreg_mitigated[0], roc_logreg_mitigated[1], label=f'Logistic Regression Mitigated (AUC: {auc_logreg_mitigated:.4f})', linewidth=2.5, color='#9467bd', linestyle='-.')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('Logistic Regression Models - ROC Curves Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig('comparison_output/roc_curves_logistic_regression.png', dpi=150)
print("Logistic Regression ROC curves saved to 'comparison_output/roc_curves_logistic_regression.png'")
plt.show()

# Plot 3: All Models Comparison
fig, ax = plt.subplots(figsize=(11, 8))
ax.plot(roc_logreg[0], roc_logreg[1], label=f'LogReg with Race (AUC: {auc_logreg:.4f})', linewidth=2.5)
ax.plot(roc_xgb[0], roc_xgb[1], label=f'XGBoost with Race (AUC: {auc_xgb:.4f})', linewidth=2.5)
ax.plot(roc_logreg_no_race[0], roc_logreg_no_race[1], label=f'LogReg without Race (AUC: {auc_logreg_no_race:.4f})', linewidth=2.5, linestyle='--')
ax.plot(roc_xgb_no_race[0], roc_xgb_no_race[1], label=f'XGBoost without Race (AUC: {auc_xgb_no_race:.4f})', linewidth=2.5, linestyle='--')
ax.plot(roc_logreg_mitigated[0], roc_logreg_mitigated[1], label=f'LogReg Mitigated (AUC: {auc_logreg_mitigated:.4f})', linewidth=2.5, linestyle='-.')
ax.plot(roc_xgb_mitigated[0], roc_xgb_mitigated[1], label=f'XGBoost Mitigated (AUC: {auc_xgb_mitigated:.4f})', linewidth=2.5, linestyle='-.')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('All Models - ROC Curves Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('comparison_output/roc_curves_all_models.png', dpi=150)
print("All models ROC curves saved to 'comparison_output/roc_curves_all_models.png'")
plt.show()

# Print AUC Summary
print("\n" + "="*50)
print("AUC SCORES SUMMARY")
print("="*50)
print(f"XGBoost with Race:              {auc_xgb:.4f}")
print(f"XGBoost without Race:           {auc_xgb_no_race:.4f}")
print(f"XGBoost Mitigated:             {auc_xgb_mitigated:.4f}")
print(f"Logistic Regression with Race:  {auc_logreg:.4f}")
print(f"Logistic Regression without Race: {auc_logreg_no_race:.4f}")
print(f"Logistic Regression Mitigated:   {auc_logreg_mitigated:.4f}")
print("="*50)        
