1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
# Does the model look at the lungs? Diagnosing and correcting view bias in a chest X-ray classifier with explainability

**Final project, Explainable AI** · Master in Advanced Artificial Intelligence, ICAI · 2026/2027
**Author:** Xabier Albizu Arias
**Status:** v0.3 (14/11/2026). Stages 1–5 done. Stage 6 (single test evaluation) pending. Nothing is added from here on.

## 1. Question

Frontal chest X-rays come in two views. **PA** is taken standing, in the radiology room, on patients who can walk. **AP** is taken in bed with a portable machine, on patients too sick to stand. So the view is a proxy for how sick the patient is: in RSNA, lung opacity is 4× more frequent in AP (38 %) than in PA (9 %).

The project asks three questions about a pneumonia classifier:

1. Does it use the view as a shortcut?
2. Where in the image does it get the view from?
3. Can the shortcut be removed without losing diagnostic performance?

**Why it matters.** A shortcut like "looks like AP → probably opacity" works inside the training data but produces false alarms on healthy AP patients and fails when the patient mix changes (an ICU is almost all AP; an outpatient clinic is almost all PA).

## 2. Setup

- **Data:** RSNA Pneumonia Detection Challenge, 26,679 frontal images. Three classes: *Lung Opacity* (positive), *No Lung Opacity / Not Normal* (negative, other findings), *Normal* (negative, no findings). Radiologist boxes for the opacities. View, sex and age from the DICOM headers.
- **Splits:** 70/15/15 by patient, stratified by label × view, seed 42. All decisions are made on validation. The test set is used once, in Stage 6.
- **Pretrained model:** TorchXRayVision DenseNet-121 `densenet121-res224-chex`, trained only on CheXpert, so it has never seen RSNA images.
- **Training recipe (all fine-tuned models):** single-logit head initialised from the *Lung Opacity* unit; AdamW (backbone 1e-4, head 1e-3); warm-up + cosine; augmentation without horizontal flip; 12 epochs, early stopping (patience 3); 3 seeds; Kaggle T4.

## 3. Models

| ID | Model | Role |
|---|---|---|
| E0 | Pretrained DenseNet, *Lung Opacity* output, no training | Starting point |
| E1 | E0 fine-tuned on RSNA | **The model we explain** |
| E2 | E1 trained and evaluated on lung-only images (outside set to black) | Fix by location: remove everything outside the lungs |
| E4 | E1 trained with a conditional view adversary | Fix by independence: the score must not reveal the view, given the label |
| Per-view threshold | E1 unchanged, one threshold for AP and another for PA | Trivial post-hoc baseline, no training |

- **E2.** Lung masks from the TorchXRayVision segmentation model, converted to the convex hull of each lung and dilated 5 px, so dense consolidations and peripheral lesions are kept. The 1.9 % of masks that look wrong are replaced by the mean hull.
- **E4.** A small network tries to predict the view from E1's logit and the true label. A gradient reversal layer trains E1 to defeat it, so the score carries no view information beyond what the label explains (Zhang et al., 2018; applied to this exact confounder by Janizek et al., 2020). The strength λ was chosen on validation (section 5).
- E2 and E4 keep the checkpoint with the best validation **within-view AUC**, because overall AUC rewards the shortcut.
- **Dropped:** E3 (*Right for the Right Reasons*, penalising attributions outside the lungs) once Stage 4 showed the bias is not outside the lungs; view-balanced reweighting, which Janizek et al. (2020) found ineffective for this confounder.

## 4. Metrics (frozen since Stage 3)

| # | Metric | Question it answers | Goal for a fix |
|---|---|---|---|
| M1 | AUC within AP and within PA | Does it still detect opacity well? | Neither drops by more than 0.01 vs E1 |
| M2 | **View leakage**: among *Normal* studies, AUC of the score for AP vs PA | Does the score reveal the view in healthy patients? (0.5 = no) | Move towards 0.5 |
| M3 | **Normal false-positive gap**: % of Normal AP flagged − % of Normal PA flagged, at each model's 90 %-sensitivity threshold (set on validation) | Do healthy AP patients get more false alarms? | At most half of E1's gap |

A fix **succeeds** if it meets M1 and M3. Results are the mean logit of the 3 seeds, with 95 % bootstrap CIs (2,000 resamples) and paired bootstrap against E1.

**Why not the usual metric.** Within-view AUC only compares patients of the same view, so it cannot see a constant "AP bonus". M2 and M3 compare across views, which is where the bias shows.

## 5. Findings

### Stage 1–2: the shortcut is available (test)

- AP has 38 % opacity vs 9 % in PA, and only 13 % of AP studies are Normal vs 50 % of PA.
- A model with only view, sex and age (B1) reaches overall AUC 0.727 but 0.55 within each view: it is pure shortcut.
- E0 (no training): overall AUC 0.779, AP 0.695, PA 0.747.

### Stage 3: fine-tuning creates the bias (test)

- E1 is a much better classifier: overall AUC 0.865 [0.852, 0.877], AP 0.817, PA 0.824. Seeds agree (std ≤ 0.001).
- **But it depends on the view.** Leakage among Normal studies goes from 0.627 [0.587, 0.666] in E0 to **0.785 [0.755, 0.814]** in E1 (paired difference +0.158 [0.126, 0.190]; seeds 0.780, 0.790, 0.784).
- **Clinical impact:** at a 90 %-sensitivity threshold, **18.4 % of Normal AP studies are flagged vs 2.8 % of Normal PA** (M3 = 15.6 pp). A healthy AP patient is 6.6× more likely to get a false alarm.

### Stage 4: where the bias comes from (validation)

- **Explanations were checked first.** Grad-CAM and HiResCAM pass the parameter-randomisation sanity check; Integrated Gradients fails and was dropped.
- **Heatmaps do not settle it.** Grad-CAM puts much of the attribution outside the lungs; HiResCAM puts it inside. HiResCAM does point at the lesions (pointing game 0.72 in AP vs 0.46 by chance).
- **Intervening on the image does.** Blurring everything outside the lungs (which removes text and cables) leaves leakage unchanged (0.794 → 0.790). Showing only the lungs on a black background lowers it to 0.70, but within-view AUC drops 0.14, so the input is too unusual for E1 and the test is inconclusive. E2 decides.
- The view is linearly readable from the embedding of both E0 and E1 (probe AUC ≈ 0.998). What changes with fine-tuning is that the head uses it: the mean logit gap between Normal AP and Normal PA grows from 0.76 to 1.22.

### Stage 5: fixes (validation)

| Model | AUC AP | AUC PA | Leakage (M2) | Overall AUC |
|---|---|---|---|---|
| E1 (reference) | 0.804 | 0.850 | 0.799 | 0.869 |
| E2, 3 seeds | 0.79–0.80 | 0.825 | 0.821–0.827 | 0.857 |
| E4, λ = 0.3, seed 0 | 0.803 | 0.850 | **0.466** | 0.858 |
| E4, λ = 1, seed 0 | 0.798 | 0.846 | 0.419 | 0.837 |
| E4, λ = 3, 3 seeds | 0.796–0.799 | 0.841–0.845 | 0.373–0.417 | 0.822 |

- **E2 does not remove the bias.** With only the lungs visible, leakage stays at 0.82. The bias lives in the lung region (H2), not in text, cables or borders, so the lung-cropping fix of Aslani et al. (2022) does not apply here. E2 is kept as the diagnostic test, not as a fix.
- **E4 removes the leakage without losing within-view AUC** at λ = 0.3.
- **Stronger λ over-corrects:** leakage falls below 0.5, which means healthy PA studies now score higher than healthy AP ones. The reason: the adversary only knows *opacity yes/no*. AP negatives are mostly *Not Normal* (sicker patients), PA negatives are half *Normal*. Making the two negative groups look alike pushes all AP scores down, so Normal AP ends below Normal PA. The negative class hides two subgroups that differ by view (hidden stratification).
- **λ choice.** Rule: among λ > 0 whose within-view AUC is at most 0.01 below λ = 0, take the one with validation leakage closest to 0.5 → **λ = 0.3**.
- **Check:** λ = 0 reproduces E1 seed 0 exactly, epoch by epoch, so every model is compared with the same code.

### Hypotheses: outcome

| Hypothesis | Outcome |
|---|---|
| H1: bias from cues outside the lungs (text, cables, borders) | **Rejected.** Removing them changes nothing (Stage 4, E2). |
| H2: bias from the lung region itself (AP projection physics, lung shape) | **Supported.** Lung-only training keeps leakage at 0.82. |
| H3: *Normal* labels less reliable in AP | Not testable without new radiologist reads. Limitation. |

## 6. What remains

1. Train E4 (λ = 0.3) seeds 1 and 2.