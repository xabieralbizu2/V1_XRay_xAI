# Is the model looking at the lungs? Detecting and fixing shortcuts in a chest X-ray classifier with XAI


**Author:** Xabier Albizu Arias  
**Status:** 🚧 Stage 1 of 6 complete (exploratory analysis)

## TL;DR

Deep learning models for chest X-rays can be right for the wrong reasons: instead of the lung opacity, they may rely on *shortcuts* such as the acquisition view (AP portable vs. PA standing), text markers or medical devices. These shortcuts work inside the training hospital and break in other hospitals.

This project takes a **pretrained DenseNet-121** (TorchXRayVision, trained on CheXpert), adapts it to pneumonia detection on the **RSNA Pneumonia** dataset, and uses **explainability** (Grad-CAM, HiResCAM, Integrated Gradients, occlusion, linear probes) to:

1. **Diagnose** whether the model relies on the view shortcut instead of the lungs.
2. **Validate** the explanations with sanity checks (parameter randomization, deletion/insertion curves, method agreement).
3. **Fix** the model with explanation-guided interventions (lung-masked input and a *Right for the Right Reasons* loss).
4. **Measure** the improvement: in-distribution AUC, worst-group AUC, AUC on a shortcut-inverted stress test and on an external dataset, plus explanation metrics.

The full proposal is in [`docs/PROYECTO.md`](docs/PROYECTO.md).

## Repository structure

```text
xai-cxr-shortcuts/
├── README.md            # this file, updated stage by stage
├── requirements.txt
├── data/README.md       # how to obtain the data (not redistributed)
├── docs/PROYECTO.md     # full project proposal (Spanish)
├── notebooks/           # one notebook per stage, imports from src/
├── results/figures/     # figures used in this README and the report
└── src/
    └── data.py          # data loading and DICOM metadata
```

## Setup

The experiments run on **Kaggle Notebooks** (free GPU, competition data already hosted).

```python
!git clone https://github.com/<user>/xai-cxr-shortcuts
import sys; sys.path.append("/kaggle/working/xai-cxr-shortcuts")
!pip install -q -r /kaggle/working/xai-cxr-shortcuts/requirements.txt
```

To run locally, create a virtual environment, install `requirements.txt` and follow [`data/README.md`](data/README.md).

## Data

[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) (Kaggle, 2018): frontal chest X-rays from the NIH ChestX-ray14 dataset, annotated by radiologists with bounding boxes of pneumonia-related lung opacities. DICOM headers provide view position, sex and age.

## Progress log

### Stage 1 — Exploratory analysis by subgroup ✅

**Goal.** Check whether the dataset contains a spurious correlation the model could exploit, before training anything.

**How.** `src/data.py` builds one row per patient from the labels file and the DICOM headers (no pixel data). Notebook: `notebooks/01_eda.ipynb`.

**Results.** 26,684 patients, overall prevalence of lung opacity 22.5%.

| Subgroup | Patients | Prevalence |
|---|---:|---:|
| AP view | 12,173 | **38.3%** |
| PA view | 14,511 | **9.3%** |
| Female | 11,518 | 21.7% |
| Male | 15,166 | 23.1% |
| Age < 40 | 9,044 | 25.5% |
| Age 40–65 | 14,167 | 20.7% |
| Age > 65 | 3,468 | 22.5% |

| View | Lung Opacity | No Lung Opacity / Not Normal | Normal |
|---|---:|---:|---:|
| AP | 38.3% | 48.2% | 13.4% |
| PA | 9.3% | 41.0% | 49.7% |

**Conclusions.**

- **The view is a strong potential shortcut.** AP images have 4.1× the prevalence of PA images. A rule that only looks at the view would reach an AUC of about 0.71 (computed analytically from the counts above) without seeing the lungs.
- AP images are rarely normal (13% vs. 50% in PA): AP portable films are taken on sicker, bedridden patients. "Looking like an AP film" is therefore highly predictive of "not normal".
- Sex and age show small differences in prevalence; they are not strong shortcuts but will be used for the fairness analysis.
- 5 patients have invalid ages (0 or > 120) and will be excluded.

**Decision.** The shortcut hypothesis is plausible, so the project continues with the main plan.

### Stage 2 — Splits and baselines (E0 zero-shot, B1 metadata only) ⏳

### Stage 2a — Splits, metadata-only baseline (B1) and image cache ✅

**Splits.** Patient-level 70/15/15 split stratified by target × view (5 patients with invalid ages removed). All experiments use the same `splits.csv`.

| Split | Patients | Prevalence | AP share |
|---|---:|---:|---:|
| Train | 18,675 | 22.5% | 45.6% |
| Validation | 4,002 | 22.5% | 45.6% |
| Test | 4,002 | 22.5% | 45.6% |

**B1: logistic regression on view, sex and age only (no image).** Test ROC-AUC with 95% bootstrap CI:

| Subset | AUC |
|---|---|
| Overall | 0.727 [0.709, 0.746] |
| Within AP | 0.552 [0.524, 0.578] |
| Within PA | 0.545 [0.504, 0.588] |

**Conclusions.**
- Without looking at the image, metadata alone reaches an AUC of 0.73. Almost all of it comes from the view: within each view the AUC drops to ~0.55.
- This "high overall, near-chance within group" pattern is the signature of a shortcut. It is the reference used to evaluate the image models: an image model whose AUC drops sharply within views is partly exploiting the view.
- Visual inspection shows that AP images carry visible cues (ECG leads, lines, tubes, text markers) even when the label is negative.

**Image cache.** 26,679 DICOMs resized once to 224×224 `uint8` (~1.3 GB, memory-mapped) in `data/processed/`.

### Stage 2b — Zero-shot pretrained model (E0) ✅

DenseNet-121 from TorchXRayVision trained **only on CheXpert** (`densenet121-res224-chex`, no NIH/RSNA images, so no test leakage), evaluated as is on the RSNA test split. Test ROC-AUC with 95% bootstrap CI:

| Model | Overall | Within AP | Within PA |
|---|---|---|---|
| B1 (metadata only) | 0.727 [0.709, 0.746] | 0.552 [0.524, 0.578] | 0.545 [0.504, 0.588] |
| E0 `Lung Opacity` output | **0.779** [0.762, 0.794] | 0.695 [0.670, 0.719] | 0.747 [0.713, 0.778] |
| E0 `Pneumonia` output | 0.749 [0.732, 0.766] | 0.692 [0.667, 0.715] | 0.724 [0.687, 0.760] |

**Conclusions.**
- The pretrained model relies on image content: its AUC stays at 0.70–0.75 within each view, while the metadata baseline collapses to ~0.55. The overall-to-within-view drop is ~0.06 for E0 versus ~0.18 for B1.
- This is expected: E0 never saw RSNA, so it could not learn this dataset's view–label correlation.
- AP images are harder (0.695 vs 0.747): AP negatives are rarely normal (13%), so separating pneumonia from other findings is harder than in PA.
- The `Lung Opacity` output matches the RSNA label best and is used as the E0 reference.

**Hypothesis for Stage 3.** Fine-tuning on RSNA (E1) will raise the overall AUC but also let the model absorb the view shortcut, widening the overall-to-within-view gap compared with E0.

### Stage 3 — Fine-tuning (E1) and explanations ⏳

### Stage 4 — Sanity checks of the explanations ⏳

### Stage 5 — Explanation-guided interventions (E2, E3) ⏳

### Stage 6 — Before/after evaluation and report ⏳

## References

- Zech, J. R. et al. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs. *PLOS Medicine*.
- DeGrave, A. J., Janizek, J. D. and Lee, S.-I. (2021). AI for radiographic COVID-19 detection selects shortcuts over signal. *Nature Machine Intelligence*.
- Cohen, J. P. et al. (2022). TorchXRayVision: A library of chest X-ray datasets and models. *MIDL*.
- Ross, A. S., Hughes, M. C. and Doshi-Velez, F. (2017). Right for the right reasons. *IJCAI*.
- Adebayo, J. et al. (2018). Sanity checks for saliency maps. *NeurIPS*.

## License

Code: MIT (to be confirmed). Data: subject to the RSNA Pneumonia Detection Challenge rules.
