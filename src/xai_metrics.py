"""Quantitative summaries of attribution maps: lung attribution fraction and pointing game."""
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def lung_attribution_fraction(maps: np.ndarray, masks: np.ndarray):
    """Share of positive attribution inside the lung mask, and that share divided by lung area.

    A map that ignores anatomy has LAF close to the lung area fraction (ratio close to 1).
    """
    a = np.clip(maps, 0, None)
    total = a.sum(axis=(1, 2))
    laf = (a * masks).sum(axis=(1, 2)) / np.maximum(total, 1e-12)
    area = masks.mean(axis=(1, 2))
    laf[total <= 0] = np.nan
    return laf, laf / np.maximum(area, 1e-6)


def box_masks(boxes: pd.DataFrame, patient_ids, size: int = 224, original: int = 1024,
              tol_px: int = 3) -> np.ndarray:
    """Radiologist boxes (x, y, width, height in original pixels) rasterised at `size`.

    `tol_px` enlarges every box (3 px at 224 is about 15 px in the original image).
    """
    out = np.zeros((len(patient_ids), size, size), dtype=bool)
    s = size / original
    by_patient = {pid: d for pid, d in boxes.dropna(subset=["width"]).groupby("patientId")}
    for k, pid in enumerate(patient_ids):
        for b in by_patient.get(pid, pd.DataFrame()).itertuples():
            x0, y0 = int(b.x * s) - tol_px, int(b.y * s) - tol_px
            x1, y1 = int((b.x + b.width) * s) + tol_px + 1, int((b.y + b.height) * s) + tol_px + 1
            out[k, max(y0, 0):min(y1, size), max(x0, 0):min(x1, size)] = True
    return out


def attribution_summary(model, method, dataset, lung_masks: np.ndarray, groups, device,
                        boxes: np.ndarray | None = None, batch_size: int = 16):
    """Per-image LAF (and pointing game when boxes are given) plus a mean map per group.

    Maps are not kept in memory: each one is normalised to sum 1 and added to its group's mean.
    Returns (per-image DataFrame, {group: mean map}).
    """
    n = len(dataset)
    groups = np.asarray(groups)
    rows, sums, counts = [], {}, {}
    for start in tqdm(range(0, n, batch_size), desc="Attributions", leave=False):
        idx = np.arange(start, min(start + batch_size, n))
        x = torch.stack([dataset[i] for i in idx]).to(device)
        maps = np.clip(method(model, x), 0, None)
        laf, ratio = lung_attribution_fraction(maps, lung_masks[idx])
        for k, i in enumerate(idx):
            row = {"i": i, "laf": laf[k], "laf_ratio": ratio[k]}
            if boxes is not None and boxes[i].any():
                peak = np.unravel_index(np.argmax(maps[k]), maps[k].shape)
                lung = lung_masks[i]
                row["hit"] = float(boxes[i][peak])
                row["chance"] = float((boxes[i] & lung).sum() / max(lung.sum(), 1))
            rows.append(row)
            total = maps[k].sum()
            if total > 0:
                g = groups[i]
                sums[g] = sums.get(g, 0) + maps[k] / total
                counts[g] = counts.get(g, 0) + 1
    means = {g: sums[g] / counts[g] for g in sums}
    return pd.DataFrame(rows), means


def mean_ci(values, z: float = 1.96):
    """Mean and normal-approximation 95 % CI, ignoring NaNs."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    return m, m - z * se, m + z * se
