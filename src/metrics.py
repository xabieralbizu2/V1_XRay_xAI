# src/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd

def bootstrap_auc(y, p, n_boot=2000, seed=0):
    """ROC-AUC with a 95% stratified bootstrap confidence interval."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    stats = [roc_auc_score(y[idx], p[idx]) for idx in
             (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
              for _ in range(n_boot))]
    return roc_auc_score(y, p), *np.percentile(stats, [2.5, 97.5])

def auc_by_group(df, score: str, group: str = "view", target: str = "target"):
    """Overall and per-group ROC-AUC with bootstrap CIs, as a tidy table."""
    rows = [("overall", *bootstrap_auc(df[target], df[score]))]
    rows += [(g, *bootstrap_auc(d[target], d[score])) for g, d in df.groupby(group)]
    return pd.DataFrame(rows, columns=["subset", "auc", "ci_low", "ci_high"]).assign(model=score)