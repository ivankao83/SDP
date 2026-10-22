# New-source training and baseline comparison — 2026-10-22

Current model: **run49_validation_balance**. Comparison: Run41, same 800 reserved recordings.

Practical target (+5 percentage points in accuracy, no recall loss, fewer false alarms): **met**. This is not a statistical-significance claim.

| Metric | Starting Run41 | Current |
|---|---:|---:|
| Test recordings | 800 | 800 |
| Accuracy | 81.25% | 86.88% |
| Gunshot recall | 82.61% | 83.33% |
| Gunshot precision | 47.50% | 58.38% |
| Gunshot F1 | 60.32% | 68.66% |
| ROC AUC | 0.9061 | 0.9445 |
| Decision threshold | 0.2450 | 0.3250 |
| Gunshots detected | 114 | 115 |
| Gunshots missed | 24 | 23 |
| False alarms | 126 | 82 |
| Correct non-gunshots | 536 | 580 |

## Per baseline dataset

| Dataset | Clips | Before TP / FN / FP / TN | Current TP / FN / FP / TN |
|---|---:|---|---|
| FSD50K_mixed | 100 | 36 / 14 / 9 / 41 | 38 / 12 / 8 / 42 |
| GunshotTriangulation | 88 | 78 / 10 / 0 / 0 | 77 / 11 / 0 / 0 |
| PlasticBagPop | 612 | 0 / 0 / 117 / 495 | 0 / 0 / 74 / 538 |

## New sources and held-out tests

| Source | Type | Unique clips | Train / validation / held-out | Held-out gunshots detected before → after | Held-out false alarms before → after |
|---|---|---:|---|---|---|
| AydemirGunshots | positive | 320 | 200 / 40 / 80 | 38 → 67 | 0 → 0 |
| ClapRIRPhone | negative | 280 | 160 / 40 / 80 | 0 → 0 | 33 → 27 |
| IndependentFieldMixed | mixed | 306 | 183 / 63 / 60 | 10 → 9 | 8 → 6 |

The 906-clip pool before each source's own training cycle: 79.58% accuracy. Its current retest: 92.27%, **including recordings used for training and validation**. This is a seen-pool result.

Pretests use the frozen model available before each source cycle. Only the 220 held-out new-source clips provide the separate never-trained comparison: 62.27% → 78.64% accuracy.

IndependentFieldMixed is a newly curated 306-clip collection, not a separately published benchmark. Original Freesound author descriptions establish real firearm discharge or non-firearm material/action labels. Only CC0 or attribution-only previews were used; generated captions were not used as labels. Whole uploaders stay in one split; complete positive recordings were preserved to avoid cropping away quiet shots.

AydemirGunshots selects 40 filenames per source folder from the published corpus. The 360 selected files contained 40 exact decoded duplicates, leaving 320 unique clips. Entire firearm-model groups are separated; M4/M16 share one training family. Original video IDs are unavailable, so model grouping cannot establish all video or cross-corpus independence. The source licence field is Unknown; local audio is excluded from Git.

ClapRIRPhone uses 280 author-recorded phone handclaps, with paired inferred RIR outputs excluded. Whole rooms define 160 training, 40 validation and 80 test recordings. Seven rooms are available; performer identity is not provided. Git blob and author export hashes were checked before resampling. The author does not assert a separate asset dataset licence; audio remains local.

## Training and model revisions

| Run | Baseline accuracy | Recall | False alarms |
|---|---:|---:|---:|
| run42_new_sources | 81.63% | 82.61% | 123 |
| run43_pattern_heads | 83.88% | 81.16% | 103 |
| run45_adapted_system | 83.63% | 79.71% | 103 |
| run46_room_handclaps | 89.13% | 80.43% | 60 |
| run47_sensitivity | 85.38% | 86.96% | 99 |
| run48_balanced_sensitivity | 86.00% | 85.51% | 92 |
| run49_validation_balance | 86.88% | 83.33% | 82 |

Run42 is the new-source training cycle with earlier-data replay and frozen CNN pattern embeddings. Run43 tests alternative learned correction heads on those approved development splits. Run44 fine-tunes a copy of the prior CNN using balanced replay and the frozen full system as a teacher; its epoch is selected on validation only. The combined adapted system is evaluated in Run45. Run46 adds the new room-separated ClapRIRPhone collection, pretesting it with Run45 before training; later model revisions may use these same approved development splits. Runs47–49 are validation-only sensitivity calibration revisions with no new weight training. These model revisions are not additional datasets.

## Verification and limits

Paired clip changes: 50 previously wrong predictions corrected; 5 previously correct predictions became wrong.

All 800 baseline recordings are excluded from training and within-run validation, threshold, epoch and correction-head selection. Repeated baseline results guide continuation and strategy changes, making this an adaptively reused development benchmark rather than an unbiased final test. Correlated microphone/event recordings also limit inferential claims. No statistical-significance claim is made.

Production inference was checked against saved predictions for each baseline dataset. The 800 paths, labels and manifest SHA256 are unchanged; confusion counts and ROC AUC were recomputed independently.

## Sources

- IndependentFieldMixed original author links and licences: dataset/run42_independent_firearms/ATTRIBUTIONS.md.
- Discovery index: https://huggingface.co/datasets/cvssp/WavCaps (original author descriptions, not generated captions).
- ClapRIRPhone source: https://github.com/Akinesia112/ClapRIR; original phone recordings only, not inferred RIRs.
- Published gunshot corpus: https://www.kaggle.com/datasets/emrahaydemr/gunshot-audio-dataset.
- Citation: Tuncer, Dogan, Akbal and Aydemir (2021), An Automated Gunshot Audio Classification Method Based On Finger Pattern Feature Generator And Iterative Relieff Feature Selector.

Original audio and model files remain local. No Git commit or push was performed.
