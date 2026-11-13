"""Operating points: one threshold for all images, or one per view (post-hoc baseline)."""
import numpy as np
import pandas as pd


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
