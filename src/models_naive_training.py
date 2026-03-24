import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from utils import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET, DATA_PATH
from data_processing import DataPreprocessor

#Load data
raw_df = pd.read_csv(DATA_PATH)
data_preprocessor = DataPreprocessor(raw_df = raw_df)
mlready_df, _ = data_preprocessor.preprocess()

#Split into train and test
mlready_train_df, mlready_test_df = train_test_split(mlready_df, test_size=0.5, random_state=1234)

#Normalize features
scaler = StandardScaler()
mlready_train_df[NUMERICAL_FEATURES] = scaler.fit_transform(mlready_train_df[NUMERICAL_FEATURES])
mlready_test_df[NUMERICAL_FEATURES] = scaler.transform(mlready_test_df[NUMERICAL_FEATURES])

#X,y for train and test
X_train = mlready_train_df.drop(columns=[TARGET])
y_train = mlready_train_df[TARGET]

X_test = mlready_test_df.drop(columns=[TARGET])
y_test = mlready_test_df[TARGET]

#Hyperparameter tuning for Logistic Regression
param_grid_logreg = {
    'C': [0.01, 0.1, 0.25, 0.5, 1, 2, 4, 10]
}

grid_logreg = GridSearchCV(LogisticRegression(), param_grid_logreg, cv=5, scoring='f1')
grid_logreg.fit(X_train, y_train)
print(f"best parameter Logistic Regression : {grid_logreg.best_params_}")

#Logistic regression training and evaluation
logistic_model = LogisticRegression(C = grid_logreg.best_params_['C'], max_iter=1000)
logistic_model.fit(X_train, y_train)
y_hat_logreg = logistic_model.predict(X_test)
proba_logreg = logistic_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_hat_logreg))

#Hyperparameter tuning for XGBoost
param_grid_xgb = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.005],
    'n_estimators': [50, 80, 100, 500]
}

grid_xgb = GridSearchCV(XGBClassifier(), param_grid_xgb, cv=5, scoring='f1')
grid_xgb.fit(X_train, y_train)
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

xgb_classifier.fit(X_train, y_train) 
y_hat_xgb = xgb_classifier.predict(X_test)
proba_xgb = xgb_classifier.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_hat_xgb))

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
plt.savefig('images/roc_curves.png')