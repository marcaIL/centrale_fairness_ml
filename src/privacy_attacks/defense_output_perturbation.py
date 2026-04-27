import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from privacy_attacks.aia_utils import load_test_data, run_aia

os.makedirs("privacy_output/reports", exist_ok=True)
os.makedirs("privacy_output/images", exist_ok=True)

RNG = np.random.default_rng(seed=2026)


# ---- Perturbation functions ----

def perturb_none(proba):
    return proba

def perturb_label_only(proba):
    return (proba >= 0.5).astype(float)

def perturb_rounding(proba, decimals):
    return np.round(proba, decimals=decimals)

def perturb_laplace(proba, scale):
    """Add Laplace noise, clip and renormalize."""
    p1 = proba
    p0 = 1.0 - proba
    n0 = np.clip(p0 + RNG.laplace(0.0, scale, size=p0.shape), 1e-6, None)
    n1 = np.clip(p1 + RNG.laplace(0.0, scale, size=p1.shape), 1e-6, None)
    return n1 / (n0 + n1)


# ---- Config ----

naive_models = [
    ("logreg_naive", "training_output/models_weights/logreg_naive.save"),
    ("xgb_naive",    "training_output/models_weights/xgb_naive.save"),
]

# (strategy_name, param_str, perturbation_fn, auc_is_balanced_acc)
strategies = [
    ("none",        "",       perturb_none,                             False),
    ("label_only",  "",       perturb_label_only,                       True),
    ("rounding",    "k=2",    lambda p: perturb_rounding(p, 2),         False),
    ("rounding",    "k=1",    lambda p: perturb_rounding(p, 1),         False),
    ("rounding",    "k=0",    lambda p: perturb_rounding(p, 0),         True),
    ("laplace",     "b=0.05", lambda p: perturb_laplace(p, 0.05),       False),
    ("laplace",     "b=0.10", lambda p: perturb_laplace(p, 0.10),       False),
    ("laplace",     "b=0.20", lambda p: perturb_laplace(p, 0.20),       False),
    ("laplace",     "b=0.30", lambda p: perturb_laplace(p, 0.30),       False),
]


# ---- Run defense + AIA ----

X_test_with_race, X_test_no_race, y_test, race_priv = load_test_data()

rows = []
for model_name, path in naive_models:
    print(f"\n[Defense] Model = {model_name}")
    model = joblib.load(path)
    clean_proba = model.predict_proba(X_test_with_race)[:, 1]

    for strategy, param, fn, auc_is_bacc in strategies:
        perturbed = fn(clean_proba)

        # Model utility (AUC on is_recid)
        model_auc = roc_auc_score(y_test, perturbed)

        # AIA on the perturbed output
        result = run_aia(
            proba_fn=lambda X, p=perturbed: p,
            X_for_model=X_test_with_race,
            X_attacker=X_test_no_race,
            race_priv=race_priv,
        )
        rows.append({
            'model': model_name,
            'strategy': strategy,
            'param': param,
            'model_auc': model_auc,
            'aia_auc': result['aia_auc'],
            'aia_accuracy': result['aia_accuracy'],
            'baseline_auc': result['baseline_auc'],
            'auc_is_balanced_acc': auc_is_bacc,
        })
        print(
            f"  {strategy:<11s} {param:<7s} | "
            f"model AUC = {model_auc:.3f} | AIA AUC = {result['aia_auc']:.3f} "
            f"(baseline B2 = {result['baseline_auc']:.3f})"
        )

# Save CSV
df = pd.DataFrame(rows)
df.to_csv('privacy_output/reports/defense_perturbation_results.csv', index=False)
print(f"\nSaved: privacy_output/reports/defense_perturbation_results.csv")


# ---- Privacy / utility trade-off plot ----

fig, ax = plt.subplots(figsize=(10, 7))
colors = {'logreg_naive': '#2980b9', 'xgb_naive': '#e67e22'}
markers = {'none': 'o', 'label_only': 'X', 'rounding': 's', 'laplace': '^'}

for model_name in df['model'].unique():
    sub = df[df['model'] == model_name]
    for strategy in sub['strategy'].unique():
        sub2 = sub[sub['strategy'] == strategy]
        ax.scatter(
            sub2['model_auc'], sub2['aia_auc'],
            color=colors[model_name],
            marker=markers[strategy],
            s=110, edgecolor='black', linewidth=0.6,
            label=f'{model_name} - {strategy}',
        )
        for _, r in sub2.iterrows():
            tag = r['param'] if r['param'] else r['strategy']
            ax.annotate(tag, (r['model_auc'], r['aia_auc']),
                        textcoords='offset points', xytext=(5, 4), fontsize=8)

    # Baseline B2 horizontal line per model
    b2 = sub['baseline_auc'].mean()
    ax.axhline(b2, linestyle=':', linewidth=1, color=colors[model_name],
               label=f'{model_name} baseline B2 = {b2:.3f}')

ax.axhline(0.5, color='k', linestyle='--', linewidth=1, label='Random AIA (AUC=0.5)')
ax.set_xlabel('Model AUC on is_recid (utility)')
ax.set_ylabel('AIA AUC on race_priv (privacy leakage)')
ax.set_title(
    'Privacy / Utility trade-off - Output perturbation (naive models)\n'
    'Note: for label_only and rounding k=0, model AUC == balanced_accuracy'
)
ax.grid(True, alpha=0.3)

# Dedup legend
handles, labels = ax.get_legend_handles_labels()
seen = set()
uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
ax.legend(*zip(*uniq), loc='best', fontsize=8)
plt.tight_layout()
plt.savefig('privacy_output/images/privacy_utility_tradeoff.png', dpi=150)
plt.close()
print(f"Saved: privacy_output/images/privacy_utility_tradeoff.png")
