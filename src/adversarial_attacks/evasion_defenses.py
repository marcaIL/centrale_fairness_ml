import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from data_processing import DataPreprocessor
from utils import DATA_PATH, NUMERICAL_FEATURES, TARGET
from adversarial_attacks.evasion_utils import pgd_attack, get_numerical_mask, SmoothedModel

os.makedirs("adversarial_output/reports", exist_ok=True)
os.makedirs("adversarial_output/images", exist_ok=True)

# ---- Load data (same split as training scripts) ----
raw_df = pd.read_csv(DATA_PATH)
mlready_df, _ = DataPreprocessor(raw_df=raw_df).preprocess()
mlready_train_df, mlready_test_df = train_test_split(mlready_df, test_size=0.5, random_state=1234)

scaler = joblib.load('training_output/scalers/scaler_original_dataset.save')
mlready_train_df = mlready_train_df.copy()
mlready_test_df = mlready_test_df.copy()
mlready_train_df[NUMERICAL_FEATURES] = scaler.transform(mlready_train_df[NUMERICAL_FEATURES])
mlready_test_df[NUMERICAL_FEATURES] = scaler.transform(mlready_test_df[NUMERICAL_FEATURES])

X_train = mlready_train_df.drop(columns=[TARGET])
y_train = mlready_train_df[TARGET]
X_test = mlready_test_df.drop(columns=[TARGET])
y_test = mlready_test_df[TARGET]

X_train_np = X_train.values.astype(float)
X_test_np = X_test.values.astype(float)
numerical_mask = get_numerical_mask(X_test.columns)

# Load original naive models
logreg_orig = joblib.load('training_output/models_weights/logreg_naive.save')
xgb_orig = joblib.load('training_output/models_weights/xgb_naive.save')


# ---- D1: Adversarial Training ----
ADV_TRAIN_EPS = 0.3

print("[D1] Generating adversarial examples on training set...")
X_adv_train_logreg = pgd_attack(logreg_orig, X_train_np, epsilon=ADV_TRAIN_EPS, numerical_mask=numerical_mask)
X_adv_train_xgb = pgd_attack(xgb_orig, X_train_np, epsilon=ADV_TRAIN_EPS, numerical_mask=numerical_mask)

# Augment: original + adversarial
X_train_aug_logreg = np.vstack([X_train_np, X_adv_train_logreg])
X_train_aug_xgb = np.vstack([X_train_np, X_adv_train_xgb])
y_train_aug = np.concatenate([y_train.values, y_train.values])

# Retrain with same hyperparameters
print("[D1] Retraining LogReg on augmented data...")
logreg_adv = LogisticRegression(C=logreg_orig.C, max_iter=1000)
logreg_adv.fit(X_train_aug_logreg, y_train_aug)

print("[D1] Retraining XGBoost on augmented data...")
xgb_params = xgb_orig.get_params()
xgb_adv = XGBClassifier(
    n_estimators=xgb_params['n_estimators'],
    learning_rate=xgb_params['learning_rate'],
    max_depth=xgb_params['max_depth'],
    subsample=xgb_params.get('subsample', 0.8),
    colsample_bytree=xgb_params.get('colsample_bytree', 0.8),
    eval_metric='logloss'
)
xgb_adv.fit(X_train_aug_xgb, y_train_aug)


# ---- D3: Randomized Smoothing ----
SMOOTH_SIGMA = 0.25
SMOOTH_N_SAMPLES = 30

print(f"[D3] Setting up randomized smoothing (sigma={SMOOTH_SIGMA}, n_samples={SMOOTH_N_SAMPLES})")
logreg_smooth = SmoothedModel(logreg_orig, sigma=SMOOTH_SIGMA, numerical_mask=numerical_mask, n_samples=SMOOTH_N_SAMPLES)
xgb_smooth = SmoothedModel(xgb_orig, sigma=SMOOTH_SIGMA, numerical_mask=numerical_mask, n_samples=SMOOTH_N_SAMPLES)


# ---- Compare all defenses ----
epsilons = [0.1, 0.2, 0.3, 0.5]

defenses = {
    'logreg': [
        ('none', logreg_orig),
        ('adv_training', logreg_adv),
        ('smoothing', logreg_smooth),
    ],
    'xgb': [
        ('none', xgb_orig),
        ('adv_training', xgb_adv),
        ('smoothing', xgb_smooth),
    ],
}

rows = []
for model_type in ['logreg', 'xgb']:
    for defense_name, model in defenses[model_type]:
        # Clean performance
        proba_clean = model.predict_proba(X_test_np)[:, 1]
        auc_clean = roc_auc_score(y_test, proba_clean)
        pred_clean = (proba_clean >= 0.5).astype(int)
        n_recid1 = (pred_clean == 1).sum()

        print(f"\n[{model_type} / {defense_name}] clean AUC = {auc_clean:.3f}, pred recid=1: {n_recid1}")

        for eps in epsilons:
            print(f"  Attacking with eps={eps:.2f} ...", end=" ", flush=True)
            X_adv = pgd_attack(model, X_test_np, epsilon=eps, numerical_mask=numerical_mask, n_steps=10)
            proba_adv = model.predict_proba(X_adv)[:, 1]
            pred_adv = (proba_adv >= 0.5).astype(int)
            auc_adv = roc_auc_score(y_test, proba_adv)

            flipped = (pred_clean == 1) & (pred_adv == 0)
            flip_rate = flipped.sum() / max(n_recid1, 1)

            rows.append({
                'model': model_type,
                'defense': defense_name,
                'epsilon': eps,
                'flip_rate': flip_rate,
                'auc_clean': auc_clean,
                'auc_adversarial': auc_adv,
            })
            print(f"flip={flip_rate:.3f} | AUC_adv={auc_adv:.3f}")

df = pd.DataFrame(rows)
df.to_csv('adversarial_output/reports/defense_comparison_results.csv', index=False)
print(f"\nSaved: adversarial_output/reports/defense_comparison_results.csv")


# ---- Plot: defense comparison ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
colors = {'none': '#e74c3c', 'adv_training': '#2ecc71', 'smoothing': '#3498db'}
markers = {'none': 'o', 'adv_training': 's', 'smoothing': '^'}

for i, model_type in enumerate(['logreg', 'xgb']):
    ax = axes[i]
    sub = df[df['model'] == model_type]
    for defense in ['none', 'adv_training', 'smoothing']:
        sub_d = sub[sub['defense'] == defense]
        auc_clean = sub_d['auc_clean'].iloc[0]
        label = f'{defense} (clean AUC={auc_clean:.3f})'
        ax.plot(sub_d['epsilon'], sub_d['flip_rate'], f'-{markers[defense]}',
                color=colors[defense], label=label, linewidth=2, markersize=8)
    ax.set_xlabel('Attack epsilon (L-inf)')
    ax.set_ylabel('Flip rate (recid=1 -> 0)')
    ax.set_title(f'{model_type} - Defense comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

plt.suptitle(
    f'Evasion defenses: none vs adv_training (eps={ADV_TRAIN_EPS}) vs smoothing (sigma={SMOOTH_SIGMA})',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig('adversarial_output/images/defense_comparison.png', dpi=150)
plt.close()
print(f"Saved: adversarial_output/images/defense_comparison.png")

