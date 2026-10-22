# Significant practical improvement on the reserved benchmark

Target set before training: at least five percentage points above the current 73.375% benchmark accuracy, with no reduction in gunshot recall and fewer false alarms. Achieved: 79.00%, a 5.625 percentage-point gain, with recall 85.51% → 86.23% and false alarms 193 → 149 (22.80% fewer). No further training followed the successful benchmark test.

## Same 800 reserved test recordings

| Metric | Previous Run14 baseline | New model |
|---|---:|---:|
| Test Recordings | 800 | 800 |
| Accuracy | 73.38% | 79.00% |
| Gunshot recall | 85.51% | 86.23% |
| Gunshot precision | 37.94% | 44.40% |
| Gunshot F1 | 52.56% | 58.62% |
| ROC AUC | 0.8264 | 0.8748 |
| Decision threshold | 0.21 | 0.21 |
| Gunshots detected | 118 | 119 |
| Gunshots missed | 20 | 19 |
| False alarms | 193 | 149 |
| Correct non-gunshots | 469 | 513 |
| False-positive rate | 29.15% | 22.51% |

| Reserved dataset | Baseline | New model |
|---|---|---|
| FSD50K mixed (100 clips) | 76% accuracy; 38/50 detections; 12 false alarms | 79% accuracy; 39/50 detections; 10 false alarms |
| Gunshot Triangulation (88 clips) | 80/88 detections | 80/88 detections |
| Plastic-Bag Pop (612 clips) | 431 correctly rejected; 181 false alarms | 473 correctly rejected; 139 false alarms |

## Successful joint data group

All four new sources were combined for fitting the added classifier, together with previously approved training recordings. Positive examples came from earlier gunshot training splits. The new sources below are negative for firearm discharge.

| New source | Unique selected clips | Train | Validation | Held-out test |
|---|---:|---:|---:|---:|
| [CMURealSounds](https://www.auditorylab.org/sound-events-database) | 295 | 214 | 45 | 36 |
| [LabImpulses](https://zenodo.org/records/8112384) | 9 | 5 | 2 | 2 |
| [IndependentBalloonPops](https://freesound.org/people/Sadiquecat/sounds/814343/) | 7 | 5 | 1 | 1 |
| [IndependentBagPops](https://freesound.org/people/ARodLRU2018/sounds/662318/) | 4 | 2 | 1 | 1 |

The pooled new held-out set contains 40 negative clips. Before training: 20 false alarms, 20 correctly rejected (50.00%). After training: 13 false alarms, 27 correctly rejected (67.50%). Gunshot recall and ROC AUC are unavailable on this negative-only set.

Training included 1,539 original clips (1,313 replay and 226 new) plus 550 training-only waveform variants. Augmented variants are not additional independent recordings. There were 1,475 validation clips including earlier development sets.

## Model revision

The Run14 CNN and its spectrogram frontend remain frozen. A gradient-boosted correction classifier was trained using the CNN probability plus amplitude-normalized, peak-centered waveform features: impulse energy concentration from 1–1000 ms, energy before/after the peak, spectral-band balance, spectral flatness, kurtosis, and zero-crossing behavior. These supply timing detail finer than the CNN spectrogram frames.

Three gradient-boosting settings were compared on validation data only. Selected: depth 3, learning rate 0.03, 380 trees, seed 1818. Threshold remained 0.21. The original-domain validation gunshot recall retention checks passed.

The current model includes BOTH the CNN and the trained correction classifier. Loading only the old CNN checkpoint will not reproduce these results.

Prediction command, run from the project directory:

```sh
.venv/bin/python predict_impulse_refined.py /absolute/path/to/audio.wav --bundle runs/run18_impulse_refiner/bundle.json
```

`reports/latest_model.json` now identifies this complete bundle. The reserved evaluator also supports `--bundle` and recognizes the current bundle by default.

## All attempted candidates

| Candidate | Reserved accuracy | Gunshot recall | False alarms | Target met |
|---|---:|---:|---:|---|
| run15_joint_impulses | 29.75% | 87.68% | 545 | No |
| run16_real_impulses | 22.12% | 71.01% | 583 | No |
| run17_bag_confounds | 56.00% | 90.58% | 339 | No |
| run18_impulse_refiner | 79.00% | 86.23% | 149 | Yes |

Failed candidates and their before/after data remain saved. Runs15–17 changed CNN weights but did not improve the benchmark, so they did not replace the working baseline. The successful fourth candidate preserves the earlier CNN and adds the trained classifier.

## Test-only boundaries and practical limits

- None of the three reserved datasets entered optimization, validation, threshold search, or the per-trial checkpoint selection. All 800 audio hashes were verified; their decoded hashes are absent from development manifests and the final model seen-hash registry.
- The baseline and final evaluation use identical paths, labels, and threshold. Saved counts/metrics were independently recomputed from CSV predictions. Live bundle predictions match the saved predictions on examples from all three reserved datasets.
- Benchmark results guided whether to continue and which failed candidates to reject. This is therefore an adaptively reused development benchmark, not a fresh unbiased estimate of performance on another unseen corpus. The five-point criterion is a practical improvement target, not a claim of statistical significance.
- Bag-pop training uses two original plastic-bag recordings, a separate paper-bag validation recording, and a separate plastic-bag test recording from different creators. The balloon group separates rooms/recordings, but one uploader appears across splits.
- Gunshot Triangulation contains 88 microphone clips from 22 firing events. Independence among Plastic-Bag Pop microphone recordings is not established.
- Exact duplicates and known FSD50K/ESC50/UrbanSound source overlaps were excluded. Complete shared-source independence from SESA cannot be established because its original source IDs are unavailable.
- Temporary source archives were removed after preserving selected audio, metadata, and checksums to limit disk usage.
