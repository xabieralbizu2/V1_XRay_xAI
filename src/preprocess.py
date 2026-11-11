"""Cache RSNA images as a single uint8 array of shape (N, 224, 224)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from skimage.transform import resize
from tqdm import tqdm

from src.data import IMAGES_DIR

IMG_SIZE = 224


def load_dicom_uint8(path: Path) -> np.ndarray:
    """Read a DICOM as float in [0, 255], with bone always bright."""
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = img / (2 ** int(ds.BitsStored) - 1) * 255.0
    if ds.PhotometricInterpretation == "MONOCHROME1":
        img = 255.0 - img
    return img


def build_image_cache(root: Path, patient_ids, out_path: Path, size: int = IMG_SIZE) -> None:
    """Resize every image once and store them in a memory-mapped .npy file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8,
                                      shape=(len(patient_ids), size, size))
    for i, pid in enumerate(tqdm(patient_ids, desc="Caching images")):
        img = load_dicom_uint8(root / IMAGES_DIR / f"{pid}.dcm")
        cache[i] = resize(img, (size, size), anti_aliasing=True,
                          preserve_range=True).round().astype(np.uint8)
    cache.flush()
    pd.Series(list(patient_ids), name="patientId").to_csv(
        out_path.with_suffix(".index.csv"), index=False)