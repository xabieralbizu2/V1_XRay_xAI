# src/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap_auc(y, p, n_boot=2000, seed=0):
    """ROC-AUC with a 95% stratified bootstrap confidence interval."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    stats = [roc_auc_score(y[idx], p[idx]) for idx in
             (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
              for _ in range(n_boot))]
    return roc_auc_score(y, p), *np.percentile(stats, [2.5, 97.5])