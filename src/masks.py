"""Lung masks from the TorchXRayVision chest segmentation model (PSPNet trained on ChestX-Det).

Masks are computed once from the 224 px image cache (upsampled to the 512 px the segmenter
expects, then downsampled back) and stored as a uint8 memory-mapped array with the same row
order as the image cache. A stats file flags masks that look wrong for manual review.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
from scipy import ndimage
from torch.utils.data import Dataset
from tqdm import tqdm

from src.dataset import CachedCXRDataset

LUNG_TARGETS = ("Left Lung", "Right Lung")
SEG_SIZE = 512


def load_segmenter() -> torch.nn.Module:
    """PSPNet from TorchXRayVision; expects (N, 1, 512, 512) in [-1024, 1024]."""
    return xrv.baseline_models.chestx_det.PSPNet().eval()


@torch.no_grad()
def segment_lungs(model: torch.nn.Module, x: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Union of both lungs for a batch `x` of shape (N, 1, H, W) in [-1024, 1024].

    Returns boolean masks of shape (N, H, W) at the input resolution.
    """
    n, _, h, w = x.shape
    channels = [model.targets.index(t) for t in LUNG_TARGETS]
    x512 = F.interpolate(x, size=(SEG_SIZE, SEG_SIZE), mode="bilinear", align_corners=False)
    prob = torch.sigmoid(model(x512)[:, channels]).amax(dim=1, keepdim=True)
    prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
    return (prob[:, 0] > threshold).cpu().numpy()


def mask_stats(mask: np.ndarray, min_component: float = 0.01) -> dict:
    """Area fraction and number of sizeable connected components (2 = both lungs found)."""
    labels, n = ndimage.label(mask)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:] / mask.size
    return {
        "area_frac": float(mask.mean()),
        "n_components": int((sizes >= min_component).sum()),
        "largest_frac": float(sizes.max()) if n else 0.0,
    }


def build_mask_cache(cache_path: Path, out_path: Path, device: torch.device,
                     batch_size: int = 16) -> pd.DataFrame:
    """Segment every image of the cache.

    Writes `out_path` (uint8 memmap, same row order as the image cache), its `.index.csv`
    and a `.stats.csv` with one row per image and a `suspect` flag for manual review.
    """
    cache_path, out_path = Path(cache_path), Path(out_path)
    images = np.load(cache_path, mmap_mode="r")
    index = pd.read_csv(cache_path.with_suffix(".index.csv"))
    n, h, w = images.shape
    masks = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(n, h, w))
    model = load_segmenter().to(device)
    stats = []
    for start in tqdm(range(0, n, batch_size), desc="Segmenting lungs"):
        batch = np.asarray(images[start:start + batch_size], dtype=np.float32)
        x = torch.from_numpy((batch / 255.0 * 2.0 - 1.0) * 1024.0)[:, None].to(device)
        m = segment_lungs(model, x)
        masks[start:start + len(m)] = m
        stats.extend(mask_stats(mi) for mi in m)
    masks.flush()
    index.to_csv(out_path.with_suffix(".index.csv"), index=False)
    stats = pd.concat([index, pd.DataFrame(stats)], axis=1)
    stats["suspect"] = ((stats.n_components != 2)
                        | (stats.area_frac < 0.12) | (stats.area_frac > 0.55))
    stats.to_csv(out_path.with_suffix(".stats.csv"), index=False)
    return stats


def load_masks(mask_path: Path, patient_ids, dilate_px: int = 0) -> np.ndarray:
    """Boolean masks (N, H, W) for `patient_ids`, optionally dilated by `dilate_px` pixels."""
    mask_path = Path(mask_path)
    index = pd.read_csv(mask_path.with_suffix(".index.csv"))
    row_of = {pid: i for i, pid in enumerate(index.patientId)}
    masks = np.load(mask_path, mmap_mode="r")
    m = np.asarray(masks[[row_of[p] for p in patient_ids]]).astype(bool)
    if dilate_px:
        m = np.stack([ndimage.binary_dilation(mi, iterations=dilate_px) for mi in m])
    return m


class MaskedCXRDataset(Dataset):
    """Cached images with one region replaced, for the lung-masking tests of Stage 4b.

    mode="lung":    keep the (dilated) lungs, replace everything outside them.
    mode="outside": keep everything outside the lungs, replace the lungs.
    fill="black":   replaced pixels set to -1024 (air in the TorchXRayVision range).
    fill="blur":    replaced pixels taken from a Gaussian-blurred copy of the image, which keeps
                    coarse intensity but removes fine cues such as text and cables.
    """

    def __init__(self, cache_path: Path, mask_path: Path, patient_ids, mode: str = "lung",
                 fill: str = "black", dilate_px: int = 3, sigma: float = 8.0):
        if mode not in ("lung", "outside") or fill not in ("black", "blur"):
            raise ValueError(f"bad mode/fill: {mode!r}, {fill!r}")
        self.images = CachedCXRDataset(cache_path, patient_ids)
        self.masks = load_masks(mask_path, patient_ids, dilate_px=dilate_px)
        self.mode, self.fill, self.sigma = mode, fill, sigma

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> torch.Tensor:
        x = self.images[i]
        keep = self.masks[i] if self.mode == "lung" else ~self.masks[i]
        if self.fill == "black":
            background = torch.full_like(x, -1024.0)
        else:
            background = torch.from_numpy(ndimage.gaussian_filter(x[0].numpy(), self.sigma))[None]
        return torch.where(torch.from_numpy(keep)[None], x, background)
