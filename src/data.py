"""Data loading utilities for the RSNA Pneumonia Detection Challenge dataset."""
from pathlib import Path

import pandas as pd
import pydicom
from tqdm import tqdm

LABELS_FILE = "stage_2_train_labels.csv"
CLASSES_FILE = "stage_2_detailed_class_info.csv"
IMAGES_DIR = "stage_2_train_images"
REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [Path("/kaggle/input"), REPO_ROOT / "data"]



def find_data_root(base: str | None = None) -> Path:
    """Locate the folder that contains the RSNA labels file (Kaggle or local)."""
    dirs = [Path(base)] if base else SEARCH_DIRS
    for d in dirs:
        if d.exists():
            matches = list(d.rglob(LABELS_FILE))
            if matches:
                return matches[0].parent
    raise FileNotFoundError(f"{LABELS_FILE} not found in {[str(d) for d in dirs]}. "
                            "See data/README.md to download the data.")


def load_labels(root: Path) -> pd.DataFrame:
    """One row per patient with binary target, number of boxes and detailed class."""
    labels = pd.read_csv(root / LABELS_FILE)  # one row per bounding box
    classes = pd.read_csv(root / CLASSES_FILE).drop_duplicates("patientId")
    df = (
        labels.groupby("patientId")
        .agg(target=("Target", "max"), n_boxes=("Target", "sum"))
        .reset_index()
    )
    return df.merge(classes, on="patientId")


def read_dicom_metadata(root: Path, patient_ids) -> pd.DataFrame:
    """Read view position, sex and age from DICOM headers (no pixel data)."""
    rows = []
    for pid in tqdm(patient_ids, desc="DICOM headers"):
        ds = pydicom.dcmread(root / IMAGES_DIR / f"{pid}.dcm", stop_before_pixels=True)
        rows.append(
            {
                "patientId": pid,
                "view": ds.ViewPosition,
                "sex": ds.PatientSex,
                "age": int(ds.PatientAge),
            }
        )
    return pd.DataFrame(rows)


def build_metadata(root: Path) -> pd.DataFrame:
    """Labels merged with DICOM metadata, one row per patient."""
    df = load_labels(root)
    return df.merge(read_dicom_metadata(root, df.patientId), on="patientId")


def prevalence_report(df: pd.DataFrame) -> dict:
    """Target prevalence overall and by subgroup."""
    age_bins = pd.cut(df.age, [0, 40, 65, 120], labels=["<40", "40-65", ">65"])
    return {
        "overall": df.target.agg(["count", "mean"]),
        "by_view": df.groupby("view").target.agg(["count", "mean"]),
        "by_sex": df.groupby("sex").target.agg(["count", "mean"]),
        "by_age": df.groupby(age_bins, observed=True).target.agg(["count", "mean"]),
        "class_by_view": pd.crosstab(df.view, df["class"], normalize="index").round(3),
    }
