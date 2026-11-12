# src/dataset.py
"""PyTorch dataset reading the cached 224x224 images."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CachedCXRDataset(Dataset):
    """Returns (1, 224, 224) tensors in the TorchXRayVision range [-1024, 1024]."""

    def __init__(self, cache_path: Path, patient_ids, transform=None):
        self.images = np.load(cache_path, mmap_mode="r")
        index = pd.read_csv(Path(cache_path).with_suffix(".index.csv"))
        row_of = {pid: i for i, pid in enumerate(index.patientId)}
        self.rows = np.array([row_of[pid] for pid in patient_ids])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = np.asarray(self.images[self.rows[i]], dtype=np.float32)
        img = (img / 255.0 * 2.0 - 1.0) * 1024.0
        x = torch.from_numpy(img)[None]
        return self.transform(x) if self.transform else x