"""Sanity checks for attribution maps: cascading parameter randomisation (Adebayo et al., 2018).

Layer groups are re-initialised from the output towards the input, cumulatively. A method
whose maps barely change while the model is being destroyed is not explaining the model.
"""
import copy

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Cumulative randomisation order (output -> input): label, attributes to re-initialise.
CASCADE = [
    ("classifier", ["classifier"]),
    ("denseblock4", ["features.norm5", "features.denseblock4"]),
    ("denseblock3", ["features.transition3", "features.denseblock3"]),
    ("denseblock2", ["features.transition2", "features.denseblock2"]),
    ("denseblock1", ["features.transition1", "features.denseblock1"]),
    ("conv0", ["features.norm0", "features.conv0"]),
]


def _reset(module: torch.nn.Module) -> None:
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def _get(model: torch.nn.Module, dotted: str) -> torch.nn.Module:
    for name in dotted.split("."):
        model = getattr(model, name)
    return model


def similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation of |maps| and SSIM of min-max normalised |maps|."""
    a, b = np.abs(a), np.abs(b)
    rho = spearmanr(a.ravel(), b.ravel()).correlation
    na = (a - a.min()) / (a.max() - a.min() + 1e-8)
    nb = (b - b.min()) / (b.max() - b.min() + 1e-8)
    return float(np.nan_to_num(rho)), float(structural_similarity(na, nb, data_range=1.0))


def _maps(methods: dict, model, x: torch.Tensor, batch_size: int) -> dict:
    out = {name: [] for name in methods}
    for i in range(0, len(x), batch_size):
        for name, fn in methods.items():
            out[name].append(fn(model, x[i:i + batch_size]))
    return {name: np.concatenate(v) for name, v in out.items()}


@torch.no_grad()
def _logits(model, x: torch.Tensor, batch_size: int) -> np.ndarray:
    return torch.cat([model(x[i:i + batch_size])[:, 0]
                      for i in range(0, len(x), batch_size)]).float().cpu().numpy()


def cascading_randomisation(model, x: torch.Tensor, methods: dict, y=None,
                            batch_size: int = 16, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Attribution maps for the trained model and after each cumulative randomisation step.

    `methods` maps a name to `fn(model, x_batch) -> (N, H, W)` maps. If labels `y` are given,
    the model's AUC on `x` is recorded at every step (it should fall to chance).

    Returns a tidy table (method, step, randomised, spearman, ssim, image_spearman, auc),
    averaged over images, plus the maps of the first image at every step for plotting.
    """
    torch.manual_seed(seed)
    model = copy.deepcopy(model).eval()
    image = x[:, 0].cpu().numpy()
    reference = _maps(methods, model, x, batch_size)
    rows, examples = [], {name: [] for name in methods}

    def record(step: int, label: str, maps: dict) -> None:
        auc = np.nan if y is None else roc_auc_score(y, _logits(model, x, batch_size))
        for name in methods:
            sims = [similarity(reference[name][i], maps[name][i]) for i in range(len(x))]
            img_sim = [spearmanr(np.abs(maps[name][i]).ravel(), image[i].ravel()).correlation
                       for i in range(len(x))]
            rows.append({"method": name, "step": step, "randomised": label,
                         "spearman": np.mean([s[0] for s in sims]),
                         "ssim": np.mean([s[1] for s in sims]),
                         "image_spearman": np.nanmean(img_sim), "auc": auc})
            examples[name].append(maps[name][0])

    record(0, "none", reference)
    for step, (label, attrs) in enumerate(tqdm(CASCADE, desc="Randomising"), start=1):
        for attr in attrs:
            _reset(_get(model, attr))
        record(step, label, _maps(methods, model, x, batch_size))
    return pd.DataFrame(rows), examples
