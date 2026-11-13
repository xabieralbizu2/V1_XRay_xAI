# Does the model look at the lungs? Diagnosing and correcting view bias in a chest X-ray classifier with explainability

**Final project, Explainable AI** · Master in Advanced Artificial Intelligence, ICAI · 2026/2027
**Author:** Xabier Albizu Arias
**Status:** v0.2 (12/11/2026). Stages 1–3 done. The evaluation protocol is frozen from this version on (section 5).

> **In one sentence.** A chest X-ray model can score well for the wrong reasons, such as the acquisition view, text markers or devices, instead of the lungs. This project uses explainability to find out *where* a view-related bias comes from and tries to remove it without losing diagnostic performance.

## 1. Problem

Frontal chest X-rays are taken in two views:

- **PA (posteroanterior):** standing, in the radiology room. Used for patients who can walk.
- **AP (anteroposterior):** in bed, with a portable machine. Used for patients too sick to stand (ICU, emergency, inpatients).

So the view is a proxy for how sick the patient is. In RSNA, lung opacity is about 4× more frequent in AP than in PA. AP images also *look* different. Some differences are external (text such as "AP/PORTABLE", cables, tubes, bed edges). Others come from the physics of the projection (magnified heart, poorer inspiration, hazier lung bases).

A model can learn "looks like AP → probably opacity". That works inside the training data but fails when the patient mix changes, for example in an ICU (almost all AP) or an outpatient clinic (almost all PA), or in a hospital that labels images differently.

**Stakeholder:** the validation committee of a radiology department deciding whether to adopt an externally trained model. Explainability serves three purposes here: debugging (find the bias), validation before deployment, and calibrated trust for the radiologist. Under the EU AI Act this is a high-risk system (arts. 13–14: transparency and human oversight).

## 2. Data and setup

- **Dataset:** RSNA Pneumonia Detection Challenge (Kaggle, 2018), with 26,679 frontal images after removing invalid ages. There are three classes: *Lung Opacity* (positive), *No Lung Opacity / Not Normal* (negative, other findings) and *Normal* (negative, no findings). The DICOM headers give view, sex and age. Radiologists drew bounding boxes for the opacities.
- **Splits:** 70/15/15 by patient, stratified by label × view, seed 42. Only `splits/patient_splits.csv` (id + split, no labels) is versioned.
- **Preprocessing:** DICOM → 224 × 224 uint8 cache (MONOCHROME1 inverted), scaled to [-1024, 1024] as TorchXRayVision expects.
- **Pretrained model:** TorchXRayVision DenseNet-121 `densenet121-res224-chex`. It was trained only on CheXpert (Stanford), so it has never seen RSNA/NIH images and there is no test leakage. We use raw logits (`op_threshs = None`).
- **Compute:** training on Kaggle (GPU T4), evaluation locally (Apple MPS).

## 3. Models

| ID | Model | Purpose | Status |
|---|---|---|---|
| B1 | Logistic regression on view, sex, age (no image) | How far you get without looking at the lungs | Done |
| E0 | Pretrained DenseNet, *Lung Opacity* output, no training | Starting point | Done |
| E1 | E0 fine-tuned on RSNA (standard training) | Model to explain | Done |
| E2 | E1 trained on lung-masked inputs | Fix 1: remove access to cues outside the lungs | Stage 5 |
| E3 | E1 + *Right for the Right Reasons* loss (penalises input gradients outside the lungs) | Fix 2: teach where to look | Stage 5 |
| E4 | E1 with view-balanced reweighting (view independent of label in training) | Fix 3: classical robustness baseline, no XAI | Stage 5 |

**E1 training:** single-logit head initialised from the *Lung Opacity* unit; AdamW (backbone 1e-4, head 1e-3, weight decay 1e-4); 1-epoch warm-up + cosine; augmentation (±10° rotation, ±5 % shift, ±10 % scale, brightness/contrast, **no horizontal flip**); early stopping on validation AUC (patience 3); 12 epochs; 3 seeds. E2–E4 use the same recipe.

## 4. Findings so far

**Stage 1 (EDA).** Opacity prevalence is about 4× higher in AP than in PA, so the view can act as a shortcut.

**Stage 2 (baselines, test set).**

| Model | Overall AUC | AP only | PA only |
|---|---|---|---|
| B1 | 0.727 | 0.552 | 0.545 |
| E0 | 0.779 | 0.695 | 0.747 |

B1 reaches 0.73 almost entirely through the view: within a single view it is near chance.

**Stage 3 (E1, test set, mean of 3 seeds).**

| Model | Overall AUC | AP only | PA only |
|---|---|---|---|
| E1 | **0.865** | **0.817** | **0.824** |

E1 is a much better classifier than E0, including within each view. Seeds agree (std ≤ 0.001). Validation AUC goes from 0.791 at epoch 0 to 0.870.

**But E1 depends on the view:**

- A naive stress test (aligned 0.96 vs inverted 0.52) turned out to be **confounded by difficulty**. AP negatives are mostly *Not Normal* (only 14 % Normal, vs 48 % in PA), and PA lesions are half the size (median box area 58k vs 110k px²). With *Normal* negatives only, inverted AUC recovers to 0.81.
- **View leakage** (among *Normal* studies only, how well the score separates AP from PA; 0.5 = ignores the view): E0 0.627 [0.587–0.666] → **E1 0.785 [0.755–0.814]**. Fine-tuning amplified the view dependence.
- **Practical impact.** At a screening threshold (90 % sensitivity, set on validation; probability 0.106), **18.4 % of Normal AP studies are flagged vs 2.8 % of Normal PA studies**. A healthy AP patient is 6.6× more likely to get a false alarm.

**Lesson learned.** The original diagnostic (overall vs within-view AUC drop) cannot see this bias. Within-view AUC only compares patients of the same view, so it is blind to a constant "AP bonus". The bias only shows up when comparing across views. This is why the metrics in section 5 replace it.


## 5. Frozen evaluation protocol

From v0.2 on, these are the **only** primary metrics. Anything else is reported as secondary or exploratory, and none of them changes after seeing results.

| # | Metric | Definition | E1 value | Goal for a fix |
|---|---|---|---|---|
| M1 | Within-view AUC | ROC-AUC computed separately in AP and in PA | 0.817 / 0.824 | Must not drop by more than 0.01 |
| M2 | View leakage | Among *Normal* studies, AUC of the score for AP vs PA | 0.785 | Move towards 0.5 |
| M3 | Normal false-positive gap | FPR in Normal AP − FPR in Normal PA, at the 90 %-sensitivity threshold of each model (set on validation) | 18.4 − 2.8 = 15.6 pp | Reduce by at least half (≤ 7.8 pp) |

**Rules:**

- All decisions (hyperparameters, λ for E3, which variant to keep) are made on **validation**. The test set is used **once**, at the end, for the final table.
- 3 seeds per model. Report the mean across seeds and 95 % bootstrap CIs (2,000 resamples). Model-to-model comparisons use a **paired** bootstrap on the same images.
- A fix counts as **successful** if M3 meets its goal and M1 does not drop by more than 0.01.

**Secondary metrics:** overall AUC, sensitivity by view at the same threshold, stress test with Normal negatives, Lung Attribution Fraction (section 6), pointing game against the radiologists' boxes.

## 6. Where does the bias come from? Hypotheses and decisive tests

| Hypothesis | Meaning | Test | Prediction if true |
|---|---|---|---|
| **H1** | The bias comes from cues **outside** the lungs (text, cables, borders) | Inference with only the lungs visible (outside set to background) | M2 drops to ≤ 0.65 |
| **H2** | The bias comes from AP projection physics **inside** the lungs | Same test | M2 stays ≥ 0.75 |
| **H3** | *Normal* labels are less reliable in AP (missed subtle opacities) | Cannot be tested without new radiologist reads | Reported as a limitation |

- **Complementary test:** inference with only the region **outside** the lungs visible. If M2 is still high, external cues are informative on their own.
- **Validity control:** masked images are out of distribution for E1. If within-view AUC with lung-only input drops by more than 0.05, the inference test is **inconclusive**, and E2 (trained on masked inputs) becomes the deciding test. E2 keeping M2 ≥ 0.75 supports H2.
- **Intermediate results** (0.65 < M2 < 0.75) mean both sources contribute. Report as such.

**Consequence for Stage 5, decided now:**

- H1 supported → E2 and E3 are the main fixes, and E4 is the comparison.
- H2 supported → location-based fixes (E2, E3) are expected to fail, and E4 is the main fix. E2 is still run to confirm the prediction.

## 7. Plan

**Stage 4: explain and test (local, MPS).**

1. **Lung masks** with the TorchXRayVision segmentation model, computed once and cached. Manually review 30–50 at random and record the failures.
2. **Sanity checks first.** Run the cascading parameter randomisation test (Adebayo et al., 2018) on ~200 images. Any method whose maps do not change when the weights are randomised is dropped before interpretation.
3. **Attributions:** Grad-CAM, HiResCAM and Integrated Gradients on E1, with E0 for comparison. Focus on Normal AP studies with high scores versus Normal PA.
4. **Lung Attribution Fraction (LAF):** share of positive attribution inside the lung mask, divided by the lung area fraction. Report it per view and per class.
5. **Deletion/insertion curves** and agreement between methods, compared against random ordering.
6. **Linear probe** on the 1024-d embedding to predict view. This tests whether the representation encodes the view, and compares E0 with E1.
7. **Decisive masking test** from section 6.

**Stage 5: fix (Kaggle, about 1 h per model with 3 seeds).** Train E2, E3 and E4 as decided in section 6. Select on validation.

**Stage 6: final evaluation and report.** One pass over the test set for all models, the final table (M1–M3 + secondary), critical discussion and README.

**Optional, if time allows:** an external test set (e.g. SIIM-FISABIO-RSNA COVID-19) to check whether the fixes improve transfer to another institution.

## 8. Limitations (to discuss in the report)

- H3 (label noise in Normal AP) cannot be ruled out.
- Single source dataset; no external validation unless the optional step is done.
- Masks are automatic, and their errors propagate to LAF and to E2/E3.
- Grad-CAM has 7 × 7 resolution, so "inside the lungs" is a coarse statement.
- A map on the lungs does not prove the model uses the clinically correct feature. Non-spatial cues (global contrast, detector noise) are invisible to heatmaps.
- Heatmaps can be misread by clinicians as "this is where the pneumonia is" (automation bias).

## 9. Key references

- Adebayo, J. et al. (2018). Sanity checks for saliency maps. *NeurIPS*.
- Cohen, J. P. et al. (2022). TorchXRayVision: A library of chest X-ray datasets and models. *MIDL*.
- DeGrave, A. J., Janizek, J. D. and Lee, S.-I. (2021). AI for radiographic COVID-19 detection selects shortcuts over signal. *Nature Machine Intelligence*.
- Draelos, R. L. and Carin, L. (2020). Use HiResCAM instead of Grad-CAM for faithful explanations of CNNs. arXiv:2011.08891.
- Geirhos, R. et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*.
- Oakden-Rayner, L. et al. (2020). Hidden stratification causes clinically meaningful failures in machine learning for medical imaging. *ACM CHIL*.
- Ross, A. S., Hughes, M. C. and Doshi-Velez, F. (2017). Right for the right reasons. *IJCAI*.
- Selvaraju, R. R. et al. (2017). Grad-CAM. *ICCV*.
- Shih, G. et al. (2019). Augmenting the NIH chest radiograph dataset with expert annotations of possible pneumonia. *Radiology: AI*.
- Sundararajan, M., Taly, A. and Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*.
- Zech, J. R. et al. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs. *PLOS Medicine*.

## 10. Changelog

| Version | Date | Change |
|---|---|---|
| v0.1 | 2026-11-11 | Initial proposal (Spanish). |
| v0.2 | 2026-11-12 | Translated to English and simplified. Added Stage 1–3 results. Replaced the within-view AUC drop with three frozen primary metrics (M1–M3). Added pre-registered hypotheses H1–H3 with decision rules. Added E4 (view-balanced reweighting). External test made optional. |
