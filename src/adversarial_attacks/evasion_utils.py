import numpy as np
from utils import NUMERICAL_FEATURES


def get_numerical_mask(columns):
    """Boolean mask: True for numerical features."""
    return np.array([c in NUMERICAL_FEATURES for c in columns])


def pgd_attack(model, X, epsilon, numerical_mask, n_steps=10, step_size=None):
    """
    PGD evasion attack via finite differences.
    Pushes predictions towards class 0 (non-recidivist).
    Only perturbs numerical features, within an L-inf epsilon-ball.
    """
    if step_size is None:
        step_size = epsilon / n_steps * 2.5

    X_adv = np.array(X, dtype=float).copy()
    X_orig = X_adv.copy()
    fd_delta = 1e-3

    for _ in range(n_steps):
        base_proba = model.predict_proba(X_adv)[:, 1]
        grad = np.zeros_like(X_adv)

        for j in range(X_adv.shape[1]):
            if not numerical_mask[j]:
                continue
            X_p = X_adv.copy()
            X_p[:, j] += fd_delta
            grad[:, j] = (model.predict_proba(X_p)[:, 1] - base_proba) / fd_delta

        # Step to decrease P(recid=1)
        X_adv -= step_size * np.sign(grad)
        X_adv[:, ~numerical_mask] = X_orig[:, ~numerical_mask]
        # Project to L-inf ball
        X_adv = np.clip(X_adv, X_orig - epsilon, X_orig + epsilon)

    return X_adv


class SmoothedModel:
    """Wraps a model with randomized smoothing (Gaussian noise on numerical features)."""

    def __init__(self, base_model, sigma, numerical_mask, n_samples=30):
        self.base_model = base_model
        self.sigma = sigma
        self.numerical_mask = numerical_mask
        self.n_samples = n_samples

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        probas = []
        for _ in range(self.n_samples):
            X_noisy = X.copy()
            n_num = self.numerical_mask.sum()
            X_noisy[:, self.numerical_mask] += np.random.normal(
                0, self.sigma, (X.shape[0], n_num)
            )
            probas.append(self.base_model.predict_proba(X_noisy))
        return np.mean(probas, axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

