# Runs 19–20 results

Two new negative corpora, each with over 1,000 usable clips and at least 200 held-out test clips. All splits were fixed before testing or training; speakers are separated within each corpus. The same test clips were scored before and after training; only separate training clips optimized weights.

| Run | Dataset | Usable clips | Train | Validation | Test | False alarms before | False alarms after training | Correct rejection before | Correct rejection after |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | SpeechCommands (negative) | 1000 | 600 | 200 | 200 | 0 | 0 | 100.00% | 100.00% |
| 20 | VocalSound (negative) | 1037 | 598 | 219 | 220 | 12 | 8 | 94.55% | 96.36% |

Speech Commands training tried 120 additional trees. Its best trained candidate used 30 added trees, but validation did not improve over the previous model, so those changes were not carried forward. VocalSound training used replay of earlier approved training data, including the new Speech Commands training split. Validation selected 100 added trees before the baseline was evaluated. The original CNN and all 380 prior refiner trees stayed unchanged.

Three nonfinite VocalSound recordings were excluded. There are 2,037 usable new clips overall: 1,198 training, 419 validation and 420 held-out testing. No augmentations are counted as new independent clips.

## Same 800-recording test-only baseline

| Metric | Previous Run18 | Run20 |
|---|---:|---:|
| Accuracy | 79.00% | 79.63% |
| Gunshot recall | 86.23% | 84.78% |
| Gunshot precision | 44.40% | 45.17% |
| Gunshot F1 | 58.62% | 58.94% |
| ROC AUC | 0.8748 | 0.8750 |
| Decision threshold | 0.21 | 0.21 |
| Gunshots detected | 119 | 117 |
| Gunshots missed | 19 | 21 |
| False alarms | 149 | 142 |
| Correct non-gunshots | 513 | 520 |

Accuracy increased 0.625 percentage points and false alarms decreased by 7 (4.70%). Recall fell 1.45 percentage points: 2 additional gunshots were missed. This is a modest tradeoff, not a significant across-the-board improvement. No claim of statistical significance is made.

## Per baseline dataset

| Dataset | Type | Clips | Before TP / FN / FP / TN | After TP / FN / FP / TN |
|---|---|---:|---|---|
| FSD50K_mixed | mixed | 100 | 39 / 11 / 10 / 40 | 37 / 13 / 9 / 41 |
| GunshotTriangulation | positive | 88 | 80 / 8 / 0 / 0 | 80 / 8 / 0 / 0 |
| PlasticBagPop | negative | 612 | 0 / 0 / 139 / 473 | 0 / 0 / 133 / 479 |

## Sources and interpretation

- Speech Commands: https://www.tensorflow.org/tutorials/audio/simple_audio; GitHub: https://github.com/tensorflow/datasets/blob/master/docs/catalog/speech_commands.md
- VocalSound: https://github.com/YuanGongND/vocalsound; compact download mirror https://zenodo.org/records/14649751 (CC-BY-SA-4.0). Cite Gong, Yu and Glass, ICASSP 2022.

Gunshot recall, gunshot precision, gunshot F1 and ROC AUC are not informative for the negative-only new subsets; correct rejection and false-alarm counts are reported instead. Baseline gunshot triangulation has 88 microphone clips of 22 shots. Repeated historical baseline evaluations mean it is an adaptively reused benchmark, not a new unbiased estimate of generalization. Baseline audio was never used for training, validation, threshold search or checkpoint selection.

Run19 trained-candidate baseline (rejected earlier on validation): 79.50% accuracy, 86.23% recall, 145 false alarms. It was not adopted based on this test result.

Final Run20 model on all 420 new held-out negatives: 8 false alarms; 98.10% correctly rejected. This uses one final model for all clips.

The threshold remains 0.21. Selected audio, split manifests, hashes, feature caches, checkpoints and raw test predictions are retained. Redundant download archives totaling 377.7 MB were deleted after verifying checksums and extraction. No Git commit or push was made in these runs.

Inference: `.venv/bin/python predict_impulse_refined.py /absolute/file.wav --bundle runs/run20_vocal_sound/bundle.json`
