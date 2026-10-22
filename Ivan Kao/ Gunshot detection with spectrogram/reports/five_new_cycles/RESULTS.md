# Five new dataset cycles and reserved baseline comparison

Completed runs 10–14, continuing from run9 through every selected trained checkpoint. Threshold stayed at 0.21.

On the identical 800 reserved test recordings, accuracy improved from 51.00% to 73.38%, and false alarms fell from 381 to 193 (49.34% fewer). Gunshot detections fell from 127/138 to 118/138; recall declined from 92.03% to 85.51%. ROC AUC also declined. This is a false-alarm/sensitivity tradeoff, not improvement in every metric.

## Five cycles

Before and after scores use each cycle’s identical held-out clips. Only train splits optimized weights; validation selected the epoch. Earlier training examples were replayed. Negative-only correct rate means correct rejection, not evidence of gunshot recall. Counts include audio excerpts and are not counts of independent source recordings.

| Run | New dataset | New train | New validation | Held-out test | Correct rate before | After | False alarms before → after | Missed gunshots before → after |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | [AudioMNIST](https://github.com/soerenab/AudioMNIST) | 2000 | 500 | 500 | 95.60% | 99.80% | 22 → 1 | 0 → 0 |
| 11 | [DCASEOffice](https://dcase.community/challenge2016/task-sound-event-detection-in-synthetic-audio) | 176 | 44 | 54 | 77.78% | 94.44% | 12 → 3 | 0 → 0 |
| 12 | [DEMAND](https://zenodo.org/records/1227121) | 360 | 90 | 60 | 100.00% | 100.00% | 0 → 0 | 0 → 0 |
| 13 | [TropicalForest](https://data.mendeley.com/datasets/x48cwz364j/3) | 319 | 80 | 120 | 71.67% | 91.67% | 3 → 5 | 31 → 5 |
| 14 | [M16Firecrackers](https://data.mendeley.com/datasets/kbgw323248/1) | 110 | 37 | 72 | 70.83% | 88.89% | 21 → 8 | 0 → 0 |

The first three corpora are negative-only; TropicalForest and M16Firecrackers contain both classes. One exact duplicate forest clip was excluded before inference.

## Reserved 800-file comparison

These three datasets were never used for optimization, validation, threshold tuning, or checkpoint selection. Their baseline predictions were preserved; their final predictions were generated only after the fifth model was selected.

| Metric | Baseline run9 | After run14 |
|---|---:|---:|
| Accuracy | 51.00% | 73.38% |
| Gunshot recall | 92.03% | 85.51% |
| Gunshot precision | 25.00% | 37.94% |
| Gunshot F1 | 39.32% | 52.56% |
| ROC AUC | 0.8411 | 0.8264 |
| Decision threshold | 0.21 | 0.21 |
| Gunshots detected | 127 | 118 |
| Gunshots missed | 11 | 20 |
| False alarms | 381 | 193 |
| Correct non-gunshots | 281 | 469 |
| False-positive rate | 57.55% | 29.15% |

| Reserved dataset | Recordings | Baseline | After five cycles |
|---|---:|---|---|
| FSD50K mixed | 100 | 70.00% accuracy; 39/50 detections; 19 false alarms | 76.00% accuracy; 38/50 detections; 12 false alarms |
| Gunshot Triangulation positive | 88 | 88/88 detections | 80/88 detections |
| Plastic-Bag Pop negative | 612 | 250/612 correctly rejected; 362 false alarms | 431/612 correctly rejected; 181 false alarms |

## Per-cycle complete metrics

### run10_audiomnist — AudioMNIST

Selected trained epoch: 6. Total optimization examples including replay: 2543.

| Metric | Before | After |
|---|---:|---:|
| Accuracy | N/A | N/A |
| Gunshot recall | N/A | N/A |
| Gunshot precision | N/A | N/A |
| Gunshot F1 | N/A | N/A |
| ROC AUC | N/A | N/A |
| Decision threshold | 0.21 | 0.21 |
| Gunshots detected | 0 | 0 |
| Gunshots missed | 0 | 0 |
| False alarms | 22 | 1 |
| Correct non-gunshots | 478 | 499 |
| False-positive rate | 4.40% | 0.20% |

### run11_office — DCASEOffice

Selected trained epoch: 3. Total optimization examples including replay: 819.

| Metric | Before | After |
|---|---:|---:|
| Accuracy | N/A | N/A |
| Gunshot recall | N/A | N/A |
| Gunshot precision | N/A | N/A |
| Gunshot F1 | N/A | N/A |
| ROC AUC | N/A | N/A |
| Decision threshold | 0.21 | 0.21 |
| Gunshots detected | 0 | 0 |
| Gunshots missed | 0 | 0 |
| False alarms | 12 | 3 |
| Correct non-gunshots | 42 | 51 |
| False-positive rate | 22.22% | 5.56% |

### run12_demand — DEMAND

Selected trained epoch: 3. Total optimization examples including replay: 1103.

| Metric | Before | After |
|---|---:|---:|
| Accuracy | N/A | N/A |
| Gunshot recall | N/A | N/A |
| Gunshot precision | N/A | N/A |
| Gunshot F1 | N/A | N/A |
| ROC AUC | N/A | N/A |
| Decision threshold | 0.21 | 0.21 |
| Gunshots detected | 0 | 0 |
| Gunshots missed | 0 | 0 |
| False alarms | 0 | 0 |
| Correct non-gunshots | 60 | 60 |
| False-positive rate | 0.00% | 0.00% |

### run13_forest — TropicalForest

Selected trained epoch: 10. Total optimization examples including replay: 1162.

| Metric | Before | After |
|---|---:|---:|
| Accuracy | 71.67% | 91.67% |
| Gunshot recall | 48.33% | 91.67% |
| Gunshot precision | 90.62% | 91.67% |
| Gunshot F1 | 63.04% | 91.67% |
| ROC AUC | 0.8856 | 0.9597 |
| Decision threshold | 0.21 | 0.21 |
| Gunshots detected | 29 | 55 |
| Gunshots missed | 31 | 5 |
| False alarms | 3 | 5 |
| Correct non-gunshots | 57 | 55 |
| False-positive rate | 5.00% | 8.33% |

### run14_m16_firecrackers — M16Firecrackers

Selected trained epoch: 1. Total optimization examples including replay: 1153.

| Metric | Before | After |
|---|---:|---:|
| Accuracy | 70.83% | 88.89% |
| Gunshot recall | 100.00% | 100.00% |
| Gunshot precision | 69.57% | 85.71% |
| Gunshot F1 | 82.05% | 92.31% |
| ROC AUC | 1.0000 | 1.0000 |
| Decision threshold | 0.21 | 0.21 |
| Gunshots detected | 48 | 48 |
| Gunshots missed | 0 | 0 |
| False alarms | 21 | 8 |
| Correct non-gunshots | 3 | 16 |
| False-positive rate | 87.50% | 33.33% |

## Boundaries and limitations

- AudioMNIST holds out complete speakers (40 train, 10 validation, 10 test).
- DCASE uses isolated office events for development and separate author evaluation-source mixtures for test; clips retain up to three seconds around the largest impulse. Original-session identifiers for isolated training events are unavailable.
- DEMAND uses one microphone channel per scene, with three-second excerpts every ten seconds; 12 scenes train, three validation, two test. Its 60 test clips come from two original recordings.
- Forest recording days are disjoint across splits. The sample is capped at ten files per day/class to reduce repeated-event dominance.
- M16/firecracker excerpts use channel 1 and three seconds around the absolute peak for both labels. Distances 50/60/70 m train, 80 m validation, 90/100 m test. Shared recording-session metadata beyond distance is unavailable.
- The three reserved test sets passed exact decoded-audio overlap checks. Complete shared-source independence from SESA cannot be established because its original source IDs were unavailable.
- The 88 Gunshot Triangulation microphone clips represent 22 firing events.
- Selected epochs did not meet every original-domain validation retention target. Actual final test recall fell, as shown above. No tuning or additional training followed this final test.

## Saved artifacts

- Each run directory holds its checkpoint, manifests, training history, selection, and before/after predictions.
- `reports/run9_external_test/` preserves the baseline; `reports/run14_external_test/` contains the final predictions.
- `verification.json` records independent prediction/count/hash and split-boundary checks.
- Temporary DCASE ZIP archives were removed after preserving selected audio, freeing 454.1 MB.
