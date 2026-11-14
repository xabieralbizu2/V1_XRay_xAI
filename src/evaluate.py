"""Operating points and the frozen primary metrics (docs/PROYECTO.md, section 5).

Every model is evaluated through a *decision score*: its test score minus the threshold that gives
90 % sensitivity on validation (one global threshold, or one per view for the post-hoc baseline).
A study is flagged as opacity when its decision score is >= 0. Subtracting a constant within a
view does not change the within-view AUC, so M1 is the same on the raw and the decision score.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def threshold_at_sensitivity(scores, y, sensitivity: float = 0.9) -> float:
    """Highest threshold that keeps at least `sensitivity` of the positives above it."""
    pos = np.sort(np.asarray(scores)[np.asarray(y) == 1])
    return float(np.quantile(pos, 1 - sensitivity))


def per_view_thresholds(df: pd.DataFrame, score: str, sensitivity: float = 0.9) -> dict:
    """One threshold per view, each giving `sensitivity` within that view (set on validation)."""
    return {v: threshold_at_sensitivity(d[score], d.target, sensitivity) for v, d in df.groupby("view")}


def flag_rates(df: pd.DataFrame, score: str, threshold) -> pd.DataFrame:
    """Share flagged as opacity per class and view. `threshold` is a float or {view: float}."""
    thr = df.view.map(threshold) if isinstance(threshold, dict) else threshold
    flagged = df[score] >= thr
    return (df.assign(flagged=flagged)
            .groupby(["class", "view"])
            .agg(n=("flagged", "size"), flagged=("flagged", "mean")))


def normal_fp_gap(df: pd.DataFrame, score: str, threshold) -> float:
    """M3: false-positive rate in Normal AP minus Normal PA, in percentage points."""
    rates = flag_rates(df, score, threshold).flagged
    return 100 * (rates[("Normal", "AP")] - rates[("Normal", "PA")])


def decision_scores(val: pd.DataFrame, test: pd.DataFrame, score: str,
                    per_view: bool = False, sensitivity: float = 0.9) -> np.ndarray:
    """Test score minus its validation threshold (global, or one per view). Flag = result >= 0."""
    if per_view:
        thr = test.view.map(per_view_thresholds(val, score, sensitivity)).to_numpy()
    else:
        thr = threshold_at_sensitivity(val[score], val.target, sensitivity)
    return test[score].to_numpy() - thr


def fast_auc(y, s) -> float:
    """ROC-AUC from the Mann-Whitney U statistic (ties count one half). Same value as sklearn."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = y.sum(), (~y).sum()
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bias_metrics(d, y, is_ap, is_normal) -> dict:
    """M1 (AUC in AP and in PA), M2 (view leakage), M3 (Normal FP gap) and secondary metrics.

    `d` are decision scores, `y` the labels, `is_ap` and `is_normal` boolean arrays.
    Rates are in percent.
    """
    d, y = np.asarray(d), np.asarray(y)
    flag, pos = d >= 0, y == 1
    fpr_ap = float(100 * flag[is_normal & is_ap].mean())
    fpr_pa = float(100 * flag[is_normal & ~is_ap].mean())
    return {
        "auc_ap": fast_auc(y[is_ap], d[is_ap]),
        "auc_pa": fast_auc(y[~is_ap], d[~is_ap]),
        "leakage": fast_auc(is_ap[is_normal], d[is_normal]),
        "m3_gap": fpr_ap - fpr_pa,
        "fpr_normal_ap": fpr_ap,
        "fpr_normal_pa": fpr_pa,
        "sens_ap": float(100 * flag[pos & is_ap].mean()),
        "sens_pa": float(100 * flag[pos & ~is_ap].mean()),
        "auc": fast_auc(y, d),
    }


def bootstrap_bias_metrics(decision: dict, test: pd.DataFrame, ref: str = "E1",
                           n_boot: int = 2000, seed: int = 0):
    """Point estimates with 95 % percentile CIs, and paired differences against `ref`.

    `decision` maps model name -> decision scores on `test`. Every model is evaluated on the
    same resampled studies, so the differences are paired. Thresholds stay fixed (they come
    from validation). Returns (table, deltas), one row per model and metric.
    """
    y = test.target.to_numpy()
    is_ap = (test.view == "AP").to_numpy()
    is_normal = (test["class"] == "Normal").to_numpy()
    point = {m: bias_metrics(d, y, is_ap, is_normal) for m, d in decision.items()}

    rng = np.random.default_rng(seed)
    boots = {m: [] for m in decision}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        for m, d in decision.items():
            boots[m].append(bias_metrics(d[idx], y[idx], is_ap[idx], is_normal[idx]))
    boots = {m: pd.DataFrame(b) for m, b in boots.items()}

    rows, deltas = [], []
    for m in decision:
        for k, v in point[m].items():
            lo, hi = np.percentile(boots[m][k], [2.5, 97.5])
            rows.append({"model": m, "metric": k, "value": v, "ci_low": lo, "ci_high": hi})
            if m != ref:
                lo, hi = np.percentile(boots[m][k] - boots[ref][k], [2.5, 97.5])
                deltas.append({"model": m, "metric": k, "delta": v - point[ref][k],
                               "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def fix_verdict(table: pd.DataFrame, deltas: pd.DataFrame, fixes, ref: str = "E1",
                max_auc_drop: float = 0.01) -> pd.DataFrame:
    """Pre-registered success rule, on point estimates.

    M1: neither within-view AUC drops by more than `max_auc_drop` against `ref`.
    M3: the absolute Normal FP gap is at most half of the reference gap.
    """
    value = table.pivot(index="model", columns="metric", values="value")
    delta = deltas.pivot(index="model", columns="metric", values="delta")
    goal = abs(value.loc[ref, "m3_gap"]) / 2
    out = pd.DataFrame(index=pd.Index(list(fixes), name="model"))
    out["d_auc_ap"] = delta.auc_ap
    out["d_auc_pa"] = delta.auc_pa
    out["m1_ok"] = (out.d_auc_ap >= -max_auc_drop) & (out.d_auc_pa >= -max_auc_drop)
    out["leakage"] = value.leakage
    out["d_leakage"] = delta.leakage
    out["m3_gap"] = value.m3_gap
    out["m3_goal"] = goal
    out["m3_ok"] = out.m3_gap.abs() <= goal
    out["success"] = out.m1_ok & out.m3_ok
    return out
