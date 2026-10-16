# Training and evaluating the MAD model

Recommended checkpoint: `runs/mixed_full_spectrum/best.pt` (threshold 0.81).
It achieved MAD gunshot recall 88.93%, precision 90.55%, and accuracy 94.50%.
On the original test set, recall was 100% and accuracy 99.94%. See
`runs/mixed_full_spectrum/RESULTS.md` for the comparison and evaluation limits.

The original exp1 model had high accuracy on CADRE/ESC-50 but detected only 8 of
280 MAD test gunshot recordings. The new training path learns from MAD's training
audio with a small CPU-friendly CNN and all 128 mel bands. It retains the existing
3-second window, 1-second hop, and recording-level maximum pooling. No positive
recording is assumed to contain gunshots in every window.

`train_mad.py` removes training videos that also occur in the official test CSV,
then uses video-grouped validation within the remaining training recordings.
YouTube timestamps are ignored when determining video identity. Exact decoded
audio duplicates between training and validation are removed from training.
The checkpoint is selected by validation gunshot F1; the threshold is selected
on the same validation split and stored in the checkpoint. Class-weighted loss
addresses the imbalance between gunshots and other sounds. `evaluate_mad.py`
checks source-video and decoded-audio overlap before computing test metrics.

From this directory:

```sh
.venv/bin/python -u train_mad.py --out runs/mad_full_spectrum
.venv/bin/python -u evaluate_mad.py --data dataset/MAD/MAD_dataset --checkpoint runs/mad_full_spectrum/best.pt --out runs/mad_full_spectrum/eval_mad
.venv/bin/python gunshot.py predict --checkpoint runs/mad_full_spectrum/best.pt --audio /absolute/path/recording.wav --output predictions.csv
.venv/bin/python -m unittest test_mad test_variable_length test_prepare_esc50
```

For a model covering both MAD and the existing dataset:

```sh
.venv/bin/python -u train_mad.py --include-original dataset --out runs/mixed_full_spectrum --epochs 15
.venv/bin/python -u evaluate_mad.py --data dataset/MAD/MAD_dataset --checkpoint runs/mixed_full_spectrum/best.pt --out runs/mixed_full_spectrum/eval_mad
OMP_NUM_THREADS=2 .venv/bin/python gunshot.py evaluate --data dataset --checkpoint runs/mixed_full_spectrum/best.pt --out runs/mixed_full_spectrum/eval_original --batch-size 16
```

The mixed run retains the existing dataset's train/validation assignments. It
selects the threshold and checkpoint to maximize the **minimum** validation
gunshot F1 across the two datasets. Per-domain validation metrics are saved in
`best_validation_by_domain.json`. The first MAD-only model improved MAD scores
but regressed on the original dataset, which motivated training on both sources.
Both test benchmarks have therefore informed development decisions; neither is
an untouched final benchmark. Their audio is excluded from training.

Use a fresh output directory for another training run. Feature files are cached
under `dataset/MAD/full_spectrum_cache_v1`. Original weights and original test
results remain in `runs/exp1`. New weights are specific to the compact CNN;
the checkpoint config identifies which architecture to load.

MAD's shooting label is mapped to gunshot; other classes are provisional
negatives. Shelling and mixed-event clips require review. Scores describe MAD
recordings, not false alarms per hour in a live environment. The MAD test split
has already been evaluated for the original model, so it is a reused benchmark;
a separate untouched dataset is needed to establish broader generalization.

Data source and attribution: Kim, Yoon, and Jung (2024), *A Military Audio Dataset
for Situational Awareness and Surveillance*.
https://github.com/kaen2891/military_audio_dataset
https://www.kaggle.com/datasets/junewookim/mad-dataset-military-audio-dataset
