# Follow-up training and baseline results — 2026-10-21

Current model: **run41_balanced_validation**. Starting comparison: Run20 on exactly the same 800 reserved clips.

| Metric | Starting Run20 | Current |
|---|---:|---:|
| Test recordings | 800 | 800 |
| Accuracy | 79.63% | 81.25% |
| Gunshot recall | 84.78% | 82.61% |
| Gunshot precision | 45.17% | 47.50% |
| Gunshot F1 | 58.94% | 60.32% |
| ROC AUC | 0.8750 | 0.9061 |
| Decision threshold | 0.210 | 0.245 |
| Gunshots detected | 117 | 114 |
| Gunshots missed | 21 | 24 |
| False alarms | 142 | 126 |
| Correct non-gunshots | 520 | 536 |

Accuracy improved by +1.63 percentage points. Gunshot recall changed by -2.17 percentage points. The stronger target of improved accuracy and fewer false alarms without a recall loss was not met.

## New data

| Dataset | Type | Clips | Gunshot / non-gunshot | Train / validation / held-out test |
|---|---|---:|---:|---|
| VietnamForest | mixed | 809 | 87 / 722 | 510 / 79 / 220 |
| RealImpactMeasured | negative | 600 | 0 / 600 | 300 / 100 / 200 |
| MIMIIValve | negative | 500 | 0 / 500 | 200 / 100 / 200 |
| ReaLISEDIndoor | negative | 2336 | 0 / 2336 | 1434 / 482 / 420 |
| AlbuferaField | mixed; training only after pretest | 226 | 55 / 171 | 226 / 0 / 0 |
| IndependentPops | negative | 429 | 0 / 429 | 255 / 74 / 100 |

There are 4,900 new original or measured microphone clips across 6 corpora. No synthetic variants were generated to meet the 200-clip minimum. RealImpact channels share hammer strikes and are separated by physical object; they are not 600 independent impacts. Albufera clips all come from one continuous recording, so the entire corpus was used only for training after its initial pretest. Its after-training retest is explicitly seen and must not be presented as unseen accuracy.

Albufera initially: 2/55 gunshots detected, 75.66% accuracy. Current seen retest: 55/55 detected, 87.17% accuracy. Comparison uses the reserved baseline and 1,140 never-trained held-out clips from the source-separated new corpora.

## Per baseline dataset

| Dataset | Clips | Starting TP / FN / FP / TN | Current TP / FN / FP / TN |
|---|---:|---|---|
| FSD50K_mixed | 100 | 37 / 13 / 9 / 41 | 36 / 14 / 9 / 41 |
| GunshotTriangulation | 88 | 80 / 8 / 0 / 0 | 78 / 10 / 0 / 0 |
| PlasticBagPop | 612 | 0 / 0 / 133 / 479 | 0 / 0 / 117 / 495 |

## Complete trial history

Runs21–24 are new-corpus cycles. Runs25 onward are pooled training, calibration or model-selection revisions using those corpora; Albufera was added in Run28 and IndependentPops in Run38. Model revisions are not additional independent datasets.

| Run | Baseline accuracy | Gunshot recall | False alarms | Retained trees / head stage |
|---|---:|---:|---:|---:|
| run21_vietnam | 79.38% | 84.78% | 144 | 30 |
| run22_realimpact | 79.38% | 84.78% | 144 | 0 |
| run23_mimii | 79.38% | 84.78% | 144 | 0 |
| run24_realised | 79.38% | 84.78% | 144 | 0 |
| run25_joint_calibrated | 79.38% | 84.78% | 144 | 0 |
| run26_pooled_recall | 79.38% | 84.78% | 144 | 0 |
| run27_bounded_updates | 79.63% | 82.61% | 139 | 2 |
| run28_albufera_bounded | 79.63% | 82.61% | 139 | 0 |
| run29_soft_prior | 79.63% | 82.61% | 139 | 0 |
| run30_adaptation_tolerance | 79.50% | 84.78% | 143 | 4 |
| run31_continued_bounded | 79.50% | 84.78% | 143 | 0 |
| run32_spectrogram_detail | 79.50% | 84.78% | 143 | 0 |
| run33_equivalent_validation | 79.50% | 84.78% | 143 | 1 |
| run34_spectrogram_adaptation | 79.25% | 84.78% | 145 | 12 |
| run35_continued_spectrogram | 79.25% | 84.78% | 145 | 0 |
| run36_feature_correction | 79.25% | 84.78% | 145 | 0 |
| run37_blended_correction | 78.75% | 83.33% | 147 | 5 |
| run38_independent_pops | 85.88% | 75.36% | 79 | 191 |
| run39_validation_sensitivity | 83.13% | 81.16% | 109 | 0 |
| run40_knowledge_blend | 79.75% | 86.23% | 143 | 0 |
| run41_balanced_validation | 81.25% | 82.61% | 126 | 0 |

## Revisions

- Corrected unstable continued boosting: tiny Hessians from extreme probabilities could make one tree flip probabilities almost completely. New Newton leaf values are bounded; regression tests verify the per-stage logit bound and unchanged earlier trees.
- Retained earlier knowledge in frozen classifier priors; clipped extreme prior confidence to permit gradual adaptation.
- Tested finer stage selection and validation-only threshold calibration. Pooled validation recall and macro-recall guards protect earlier positive examples; negative rejection is the selection objective.
- Adaptation revisions allow explicitly recorded small validation-score tolerances and use new training-domain fit as a secondary objective. These revisions do not claim independent validation of that training-only domain.
- Added optional peak-aligned spectrogram mean/max summaries alongside CNN and waveform features. Old bundles remain loadable; the prediction CLI now defaults to the current saved bundle.
- Runs39 onward calibrate sensitivity and optionally blend the learned system with the frozen Run20 expert; blend weight and threshold are selected on development validation only. These are calibration revisions, not new dataset cycles.
- IndependentPops uses author-labelled public Freesound previews, excludes all known FSD50K source IDs and reserved uploaders, and separates whole uploaders across train, validation and test. Labels use original author descriptions, not generated WavCaps captions.

## Evaluation limits

None of the 800 reserved recordings entered training, validation or within-run threshold/model selection. Repeated baseline evaluations did guide stopping and strategy changes. This makes the baseline an adaptively reused development benchmark; it is not an unbiased new final test. No statistical-significance claim is made. Gunshot Triangulation contains 88 microphone clips of 22 firing events. A higher aggregate accuracy must be read alongside recall, precision and each dataset result.

## Sources

- Vietnam: https://zenodo.org/records/13893977 — Vietnam-only subset; Belize audio excluded.
- RealImpact: https://github.com/samuel-clarke/RealImpact — Clarke et al., CVPR 2023; measured, hammer-deconvolved object impacts.
- MIMII: https://zenodo.org/records/3384388 — one SNR version, distinct machine IDs across splits.
- ReaLISED: https://zenodo.org/records/6488321 — Mohino-Herranz et al., 2022; television excluded; classes separated; original room/session IDs unavailable.
- Albufera: https://github.com/gpinyero/EUSIPCO2023 — author-labelled shooting/background clips, training-only after pretest.
- IndependentPops: original Freesound URLs and licences in dataset/run38_independent_pops/ATTRIBUTIONS.md; discovery metadata from https://huggingface.co/datasets/cvssp/WavCaps. One original source ID per clip.

The 56-WAV prepared firearm library was excluded for failing the 200-clip minimum. Redundant archives were removed after extraction/checksum records; selected audio, manifests, models and predictions remain local. No commit or push was performed.

Current bundle: `runs/run41_balanced_validation/bundle.json`. Prediction: `.venv/bin/python predict_impulse_refined.py /absolute/audio.wav`.
