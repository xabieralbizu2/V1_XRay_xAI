"""PyTorch dataset reading the cached 224x224 images."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CachedCXRDataset(Dataset):
    """Returns (1, 224, 224) tensors in the TorchXRayVision range [-1024, 1024].

    If `labels` is given, items are (image, label) pairs; otherwise just the image.
    The memory-mapped cache is opened lazily so the dataset can be sent to
    DataLoader workers without copying the whole array.
    """

    def __init__(self, cache_path: Path, patient_ids, labels=None, transform=None):
        self.cache_path = Path(cache_path)
        index = pd.read_csv(self.cache_path.with_suffix(".index.csv"))
        row_of = {pid: i for i, pid in enumerate(index.patientId)}
        self.rows = np.array([row_of[pid] for pid in patient_ids])
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.transform = transform
        self._images = None

    @property
    def images(self) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self.cache_path, mmap_mode="r")
        return self._images

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        img = np.asarray(self.images[self.rows[i]], dtype=np.float32)
        img = (img / 255.0 * 2.0 - 1.0) * 1024.0
        x = torch.from_numpy(img)[None]
        if self.transform is not None:
            x = self.transform(x)
        if self.labels is None:
            return x
        return x, torch.tensor([self.labels[i]])
