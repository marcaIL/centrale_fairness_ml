import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from privacy_attacks.aia_utils import load_test_data
from adversarial_attacks.evasion_utils import pgd_attack, get_numerical_mask

os.makedirs("adversarial_output/reports", exist_ok=True)
os.makedirs("adversarial_output/images", exist_ok=True)

# Load test data (same split as training scripts)
X_test_with_race, X_test_no_race, y_test, race_priv = load_test_data()

# Load naive models
logreg = joblib.load('training_output/models_weights/logreg_naive.save')
xgb = joblib.load('training_output/models_weights/xgb_naive.save')

numerical_mask = get_numerical_mask(X_test_with_race.columns)
X_test_np = X_test_with_race.values.astype(float)

epsilons = [0.05, 0.1, 0.2, 0.3, 0.5]
models = [("logreg", logreg), ("xgb", xgb)]

rows = []
for model_name, model in models:
    pred_orig = model.predict(X_test_np)
    proba_orig = model.predict_proba(X_test_np)[:, 1]
    auc_orig = roc_auc_score(y_test, proba_orig)
    mask_recid1 = pred_orig == 1

    print(f"\n[Evasion] Model = {model_name} | original AUC = {auc_orig:.3f}")
    print(f"  Samples predicted as recid=1: {mask_recid1.sum()} / {len(pred_orig)}")

    for eps in epsilons:
        X_adv = pgd_attack(model, X_test_np, epsilon=eps, numerical_mask=numerical_mask)
        pred_adv = model.predict(X_adv)
        proba_adv = model.predict_proba(X_adv)[:, 1]

        # Flip rate (recid=1 -> 0)
        flipped = (pred_orig == 1) & (pred_adv == 0)
        flip_rate = flipped.sum() / mask_recid1.sum() if mask_recid1.sum() > 0 else 0

        # AUC under attack
        auc_adv = roc_auc_score(y_test, proba_adv)

        # Mean L2 perturbation
        delta = X_adv - X_test_np
        mean_l2 = np.mean(np.linalg.norm(delta, axis=1))

        # Flip rate by race group
        race_priv_np = race_priv.values
        mask_priv = race_priv_np == 1
        mask_depr = race_priv_np == 0

        flip_priv = flipped[mask_priv].sum() / max((pred_orig[mask_priv] == 1).sum(), 1)
        flip_depr = flipped[mask_depr].sum() / max((pred_orig[mask_depr] == 1).sum(), 1)

        rows.append({
            'model': model_name,
            'epsilon': eps,
            'flip_rate': flip_rate,
            'flip_rate_privileged': flip_priv,
            'flip_rate_deprived': flip_depr,
            'auc_original': auc_orig,
            'auc_adversarial': auc_adv,
            'mean_l2_perturbation': mean_l2,
        })
        print(f"  eps={eps:.2f} | flip={flip_rate:.3f} (priv={flip_priv:.3f}, depr={flip_depr:.3f}) | AUC={auc_adv:.3f}")

df = pd.DataFrame(rows)
df.to_csv('adversarial_output/reports/evasion_attack_results.csv', index=False)
print(f"\nSaved: adversarial_output/reports/evasion_attack_results.csv")


# Plot: flip rate vs epsilon, stratified by race
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for i, model_name in enumerate(['logreg', 'xgb']):
    ax = axes[i]
    sub = df[df['model'] == model_name]
    ax.plot(sub['epsilon'], sub['flip_rate'], '-o', color='black', label='Overall', linewidth=2)
    ax.plot(sub['epsilon'], sub['flip_rate_privileged'], '--s', color='#2ecc71', label='Privileged')
    ax.plot(sub['epsilon'], sub['flip_rate_deprived'], '--^', color='#e74c3c', label='Deprived')
    ax.set_xlabel('Epsilon (L-inf)')
    ax.set_ylabel('Flip rate (recid=1 -> 0)')
    ax.set_title(f'Evasion attack - {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

plt.suptitle('PGD Evasion Attack on naive models - stratified by race', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('adversarial_output/images/evasion_attack_results.png', dpi=150)
plt.close()
print(f"Saved: adversarial_output/images/evasion_attack_results.png")

