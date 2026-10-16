# C3GD training and held-out evaluation

C3GD v2 was downloaded from https://zenodo.org/records/22286299. The archive MD5
matched `a7d1961da8b7733c7019853b80d706d4`. It contains 12,112 one-second gunshot
recordings. Attribution: Sinclair Gurny and Ryan Quinn, *Certus Caliber
Classification Gunshot Dataset*, CC BY 4.0.

The fixed collection-event split is:

| Use | Collection events | Original recordings |
|---|---|---:|
| Training | `oh_farm`, `nj_farm` | 6,019 |
| Validation | `ny_gravel_pit` | 1,990 |
| Testing only | `nj_farm_2` | 4,103 |

All microphones, channels, and shots from an event stay in that split. The two
New Jersey events are separate collection sessions at the same general venue;
this is an event-separated test, not a wholly new geographic location. Existing
original/MAD training and validation assignments are retained.

Training starts from `runs/mixed_full_spectrum/best.pt`. Each C3GD recording is
mixed with a negative background drawn only from its assigned split, paired
with that same background without the shot. Durations alternate between one and
three seconds. Both members receive the same bandwidth treatment. Deterministic
augmentation uses SNRs from -5 to 20 dB, 12/16/24 kHz bandwidth variants, and two
synthetic echo delays. This creates 12,038 C3GD training examples and 3,980 C3GD
validation examples. Together with the previous data, the totals are 22,098
training and 6,441 validation examples. These include synthetic pairs and are
not counts of unique natural recordings.

Checkpoint and threshold selection maximize the lowest validation gunshot F1
across original, MAD, and augmented C3GD data. A second training candidate
balances sampling by dataset and binary class to improve retention of earlier
data. Choose between these candidates using validation scores only, before
running test inference. Preserve both candidate histories.

```sh
.venv/bin/python -u train_c3gd.py
.venv/bin/python -u train_c3gd.py --out runs/c3gd_balanced --domain-balanced --learning-rate 0.0001 --epochs 10
.venv/bin/python -m unittest test_c3gd test_mad test_variable_length test_prepare_esc50
```

The first interrupted attempt, `runs/c3gd_transfer_invalid_labels`, is explicitly
invalid and excluded from model selection. It exposed a MAD multiclass-to-binary
mapping bug in the new loader. The mapping and class weighting were corrected,
regression tests added, and training restarted from the unchanged run #2 weights.

ShotSpotter is excluded from training, validation, backgrounds, and threshold
selection. Test it only after selecting and freezing a checkpoint. C3GD and
ShotSpotter are positive-only datasets, so report recording recall, not
two-class accuracy or precision. Original and MAD test sets provide separate
checks involving negative recordings. These older benchmarks and ShotSpotter
have already informed development, so their future evaluations are reused
benchmarks rather than untouched final estimates.

Example test commands, replacing the run directory with the validation-selected
candidate:

```sh
.venv/bin/python -u evaluate_c3gd.py --run runs/c3gd_balanced
.venv/bin/python -u evaluate_shotspotter.py --data dataset/shotspotter --out runs/c3gd_balanced/eval_shotspotter --checkpoints runs/mixed_full_spectrum/best.pt runs/c3gd_balanced/best.pt
.venv/bin/python -u evaluate_mad.py --data dataset/MAD/MAD_dataset --checkpoint runs/c3gd_balanced/best.pt --out runs/c3gd_balanced/eval_mad
OMP_NUM_THREADS=2 .venv/bin/python gunshot.py evaluate --data dataset --checkpoint runs/c3gd_balanced/best.pt --out runs/c3gd_balanced/eval_original --batch-size 16
```

Training saves source manifests, augmentation parameters, validation scores,
and training/validation audio hashes. Evaluation rejects exact decoded-audio
matches against those hashes. Hash checks do not prove absence of all
near-duplicates. Earlier checkpoints remain unchanged.
