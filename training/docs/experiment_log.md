# Experiment log

Notes kept while building and then stress-testing the MILK10k model released here. The numbers are pooled
out-of-fold (OOF) scores on the 5,240 training lesions unless stated otherwise. The scores of every run listed below are
recomputed from its OOF file in `results/experiments_scores.csv` and in notebooks 05–06.

_This model was developed for a research competition and is not a medical device._

## 1. Problem and metric

For each lesion we predict P(malignant) and an 11-class diagnosis, from a dermoscopic image, a clinical
close-up and a few metadata fields. The score is `0.7·pAUC80 + 0.3·MacroAUC₁₁ − 0.1·FairGap`; see the README
section "The metric" and the docstring of `src/milk10k/metrics.py`. pAUC80 carries most of the weight, so the
question that matters is how well the malignant lesions are ranked in the high-sensitivity region.

## 2. Common setup

- **Folds.** One fixed 5-fold split (`data/folds.csv`), stratified by diagnosis and grouped by `patient_id`. The release
  has no patient IDs, so `patient_id := lesion_id`, and lesions from one person can land in different folds. CV is
  optimistic for that reason.
- **Scoring.** Always the pooled OOF score. Fold means and seed means are reported separately and never mixed up
  with it.
- **Two environments.** Early runs used a local RTX 3050 with bf16. Later runs used a Kaggle T4 with fp16, which has
  no native bf16. Scores from the two are not directly comparable, so every comparison stays within one environment.

## 3. Phase 1: building the model

Starting point: two ResNet18 encoders (`resnet18.a1_in1k`, one per image type), a tabular MLP, and a
two-layer Transformer over four pooled tokens. Training uses 224 px, 10 epochs, focal loss + BCE, AdamW 3e-4 and
cosine decay. Each experiment changes one factor.

| Run | Change | Seeds, environment | Score (seed mean) | Δ | Decision and reason |
|---|---|---|---:|---:|---|
| Baseline | Seeded 224 px | 2, local | 0.6232 | — | Parent. The two seeds differ by 0.0155, so a local change must beat 2 × 0.0155 = 0.031 |
| P1a | 288 px | 2, local | 0.6400 | +0.0168 vs baseline | Promising but below 0.031. Became the reference recipe |
| P1b | 384 px | 2, local | 0.6344 | −0.0056 vs P1a | Dropped: both seeds lower than at 288 px |
| P3b-mel | MEL weight 3 in the binary loss | 2, local | 0.6118 | −0.0281 vs P1a | Dropped: pAUC80 fell in both seeds |
| P3c | Blend of binary and class heads (OOF only) | 1, local | 0.6292 | −0.0017 vs baseline seed 42 | Dropped |
| Clinical crop-to-fill | Evaluation-time crop instead of letterbox | 2, local | 0.6408 | +0.0009 vs P1a | Noise, not adopted |
| **P1a-kg** | P1a recipe on Kaggle | 3, Kaggle | **0.6295** (0.6446 / 0.6204 / 0.6237) | — | **Reference** for every later decision |
| P2a | ConvNeXt-T backbone | 2, Kaggle | 0.5928 | −0.0368 | Dropped, *for this recipe*: its training loss was still falling at epoch 10 |
| P4a | Two spatial cross-attention blocks between the image streams | 2, Kaggle | 0.5341 | −0.0955 | Dropped. Also much worse without the clinical image (pAUC80 0.4126 vs 0.5251) |
| Clinical removed | P1a-kg checkpoints evaluated without the clinical image | 3, Kaggle | 0.6182 | −0.0114 | The clinical image helps |
| TTA | Four flip views at inference | 3, Kaggle | 0.6346 | +0.0051 | Unconfirmed |
| **Final ensemble** | Mean of the three P1a-kg seeds (each lesion scored by its held-out models) | 3, Kaggle | **0.6505** | +0.0210 vs single-seed mean | Released model. Optional TTA variant: 0.6555 |

Not run:
- an EfficientNet-B2 or B3 backbone, because nothing showed the backbone was the bottleneck;
- a focal-γ sweep and a global positive-class weight;
- crop-based TTA.

Where the final gain comes from: +0.0210 of the +0.0273 over the local baseline is seed ensembling (CI
[+0.0123, +0.0380]). The remainder crosses environments and cannot be attributed to the resolution change.

## 4. Deciding with a gate

Three seeds of the same recipe scored 0.6446, 0.6204 and 0.6237. A single-seed comparison can therefore swing by
more than 0.02 with no real change, so we fixed a rule before running any candidate on Kaggle:

- σ = sample standard deviation of the three P1a-kg scores = **0.013136**.
- For a candidate with n seeds: Δ = mean(candidate scores) − mean(reference scores), and
  `T(n) = max(0.01, 2σ·√(1/n + 1/3))`, so **T(2) = 0.0240** and **T(3) = 0.0215**.
- Keep if Δ > T. Drop if Δ ≤ 0. In between, the candidate is promising: add a seed and test again.

Paired bootstrap intervals (1,000 lesion resamples, seed 0) are reported next to every comparison. They only cover
lesion sampling for fitted models, not seed-to-seed variation, so they never replace the gate. The final ensemble
(+0.020968 vs the single-seed mean) is just below T(3) = 0.021451. It is released because it is the reference recipe,
averaged; it is not a confirmed training improvement.

## 5. Phase 2: what limits the model?

Before trying anything new we looked for evidence of where the remaining error comes from.

- **Convergence.** 14 of the 15 best epochs fall at epochs 8–10, but only 3 at epoch 10. From epoch 8 to 10 the mean
  fold score moved +0.0109, −0.0069 and −0.0006 for the three seeds, while the training loss kept falling. That
  is a late plateau, not undertraining, so "train longer" lost priority.
- **Hard positives.** 690 of the 3,454 malignant lesions fall below the ≥80%-sensitivity cut of the final ensemble.
  427 of them (61.9%) are hard for **all three** seeds and 621 for at least two. Ensembling moved 311 of the 1,000
  seed-level hard lesions out of the tail, but none of the 427 shared ones.
  - **Melanoma is the weak spot:** 295 of 450 melanomas are hard. BCC contributes 285 hard cases out of 2,522, and SCCKA 104 of 473.
  - **Which benign lesions outrank them:** BKL, NV and AKIEC account for 92.4% of the lost pAUC80 area (42.1%, 26.6% and 23.7%), with 222,424, 140,570 and 125,323 benign-over-hard-malignant pairs.
  - **Severity:** 193 hard lesions sit below more than half of all benign lesions, and all but one of those are shared by every seed.
- **Preprocessing.** All images are 600×450. Dermoscopy is stretched by 1.333× when resized to a square, and the
  clinical letterbox pads 25% of the canvas. A matched visual review of 40 persistent-hard and 40 easy malignant
  lesions (24 MEL, 12 BCC, 4 SCCKA, matched on class, site and skin-tone code) showed no difference in framing,
  crop loss or distortion between the two groups. It did show that the `GaussNoise` augmentation visibly
  corrupts training images in both streams. We confirmed the noise survives into the normalised tensors (0.634–0.976 RMS).
  It is selected for about 6.7% of dermoscopic and 10% of clinical training views.
- **FairGap.** The code 1–2 group has only 111 lesions (38 malignant, 73 benign). Across the three reference seeds,
  FairGap's standard deviation is 0.0691, against 0.0163 for pAUC80 and 0.0026 for macro-AUC. The final ensemble's
  FairGap of 0.169 comes from that small group scoring *higher* (pAUC80 0.7228 vs 0.5535 and 0.5899). It is noise
  around a small sample, not evidence about fairness. The direction of the source skin-tone coding is also unverified.

## 6. Phase 2: one controlled test, a negative result

The GaussNoise finding was the only concrete defect, so we tested exactly that.
- **Change:** `GaussNoise` replaced by a no-op in its `OneOf` slot. This keeps the blur probabilities unchanged.
- **Everything else identical:** recipe, folds and seeds 42/43. The software environment was identical to the reference, down to the installed package list.
- **Training time:** 38.2 and 35.2 min per seed.

| | Score | pAUC80 | Macro-AUC | FairGap | pAUC80 codes 1–2 / 3–4 / 5–6 |
|---|---:|---:|---:|---:|---|
| Noise off, seed 42 | 0.6339 | 0.5417 | 0.8717 | 0.0678 | 0.502 / 0.527 / 0.569 |
| Noise off, seed 43 | 0.6262 | 0.5314 | 0.8687 | 0.0633 | 0.492 / 0.517 / 0.555 |
| Noise off, mean | 0.6301 | 0.5365 | 0.8702 | 0.0655 | 0.497 / 0.522 / 0.562 |
| Reference P1a-kg, mean of 3 | 0.6295 | 0.5395 | 0.8788 | 0.1174 | 0.601 / 0.525 / 0.553 |

- **Gate result:** Δ = +0.0005, far below T(2) = 0.0240. The CI for the difference of means is [−0.016, +0.015].
- **Components:** pAUC80 and macro-AUC both **fell** in both seeds.
- **Where the gain comes from:** the tiny score gain comes entirely from FairGap. Its +0.0052 offsets −0.0021 from pAUC80 and −0.0026 from macro-AUC. FairGap shrank because the code 1–2 group was pulled *down* (0.601 → 0.497), not because the weaker groups improved.
- **Seed 44:** not run. At n = 3 a keep would have needed a seed scoring above 0.6928, far above anything observed.
- **Decision:** recorded as a **negative result**; the released model keeps its augmentation unchanged.

Lessons:
1. A visibly destructive augmentation can still help, presumably as regularisation. Seeing a defect justifies an ablation, not an expected gain.
2. Decompose any score change into its components before believing it, especially a change driven by FairGap on a small group.

## 7. Future work (Phase 3)

Under the gate, a two-seed candidate must beat the reference seed mean by about **+0.024**. No single change here
came close, so the next steps should target large effects rather than refinements. A realistic expectation is
0.66–0.68; 0.7 would be a stretch.

1. **Representation.** The hard positives are stable across seeds: 427/690 hard for all three. They are mostly
   melanoma ranked below BKL, NV and AKIEC, so more seeds of the same backbone will not fix them. Candidates:
   - an EfficientNet-B3 dermoscopy encoder, keeping the clinical ResNet;
   - a lower backbone learning rate (3e-4 → 1e-4) with the head kept at 3e-3, which needs separate learning-rate settings.
2. **A ranking loss for the pAUC tail.** pAUC80 is the weakest and heaviest component (0.5716 vs macro-AUC 0.8911).
   A pairwise softplus loss between all benign logits and the lowest-scoring 20% of malignant logits in a batch
   targets that region directly.
3. **Groundwork:**
   - verify the direction of the skin-tone coding;
   - train clinical-only and tabular-only models, which have never been run;
   - bootstrap the per-group pAUC80 to see how stable FairGap really is on 111 lesions.
4. **Deprioritised:**
   - longer training: the curves show a plateau, not undertraining;
   - different crops: the matched review found no framing problem;
   - self-supervised initialisation: we could not establish that suitable public weights were eligible and free of overlap with MILK10k.
