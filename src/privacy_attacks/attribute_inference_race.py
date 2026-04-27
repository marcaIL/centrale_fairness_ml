import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from privacy_attacks.aia_utils import load_test_data, run_aia

# Output folders
os.makedirs("privacy_output/reports", exist_ok=True)
os.makedirs("privacy_output/images", exist_ok=True)

# Load test data (same split as training scripts)
X_test_with_race, X_test_no_race, y_test, race_priv = load_test_data()

# Models to attack: (name, path, uses_race_features)
models = [
    ("logreg_naive",     "training_output/models_weights/logreg_naive.save",            True),
    ("xgb_naive",        "training_output/models_weights/xgb_naive.save",               True),
    ("logreg_no_race",   "training_no_race_output/models_weights/logreg_naive.save",    False),
    ("xgb_no_race",      "training_no_race_output/models_weights/xgb_naive.save",       False),
    ("logreg_mitigated", "training_mitigated_output/models_weights/logreg_naive.save",  True),
    ("xgb_mitigated",    "training_mitigated_output/models_weights/xgb_naive.save",     True),
]

# Run AIA on each model
rows = []
for name, path, uses_race in models:
    print(f"\n[AIA] Attacking {name} ...")
    model = joblib.load(path)
    X_for_model = X_test_with_race if uses_race else X_test_no_race

    result = run_aia(
        proba_fn=lambda X, m=model: m.predict_proba(X)[:, 1],
        X_for_model=X_for_model,
        X_attacker=X_test_no_race,
        race_priv=race_priv,
    )
    result['model'] = name
    rows.append(result)
    print(
        f"  AIA accuracy = {result['aia_accuracy']:.3f} "
        f"(AUC = {result['aia_auc']:.3f}) | "
        f"baseline B2 accuracy = {result['baseline_accuracy']:.3f} "
        f"(AUC = {result['baseline_auc']:.3f})"
    )

# Save results to CSV
df = pd.DataFrame(rows)[
    ['model', 'aia_accuracy', 'aia_balanced_accuracy', 'aia_auc',
     'baseline_accuracy', 'baseline_balanced_accuracy', 'baseline_auc',
     'n_eval', 'pct_privileged_eval']
]
df.to_csv('privacy_output/reports/aia_race_results.csv', index=False)
print(f"\nSaved: privacy_output/reports/aia_race_results.csv")

# Comparison barplot
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(df))
width = 0.38
ax.bar(x - width / 2, df['baseline_auc'], width, label='Baseline B2 (features only)', color='#95a5a6')
ax.bar(x + width / 2, df['aia_auc'], width, label='AIA (features + model output)', color='#c0392b')
ax.axhline(0.5, color='k', linestyle='--', linewidth=1, label='Random (AUC=0.5)')
ax.set_xticks(x)
ax.set_xticklabels(df['model'], rotation=20, ha='right')
ax.set_ylabel('Attacker AUC on race_priv')
ax.set_title('Attribute Inference Attack on race - 3 training strategies')
ax.set_ylim(0.45, 1.0)
ax.grid(True, axis='y', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('privacy_output/images/aia_race_comparison.png', dpi=150)
plt.close()
print(f"Saved: privacy_output/images/aia_race_comparison.png")
