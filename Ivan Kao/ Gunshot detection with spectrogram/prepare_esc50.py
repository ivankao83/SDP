"""Convert extracted ESC-50 into fold-safe, overlapping negative examples.
Source: https://github.com/karolpiczak/ESC-50
Dataset: Karol J. Piczak, ESC: Dataset for Environmental Sound Classification,
ACM Multimedia 2015, https://doi.org/10.1145/2733373.2806390.
Upstream dataset is CC BY-NC (ESC-10 subset: CC BY); retain per-clip attribution
when redistributing audio. This code does not redistribute the dataset.
Default folds: train 1-3, validation 4, test 5; windows at 0-3, 1-4, 2-5 seconds.
These overlapping windows are correlated. Add labeled gunshots separately.
Audit incidental events and evaluate on same-environment positive/negative audio
so recording-source differences do not become a shortcut for classification.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 24000
SAMPLES = 3 * SR


def prepare(source, output, val_fold=4, test_fold=5):
    source, output = Path(source), Path(output)
    if val_fold == test_fold or val_fold not in range(1, 6) or test_fold not in range(1, 6):
        raise ValueError('Validation/test folds must be different integers from 1 to 5.')
    with (source/'meta'/'esc50.csv').open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required = {'filename', 'fold', 'category', 'src_file'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f'Metadata must contain {required}')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty ESC-50 metadata.')
    folds_by_source, filenames = {}, set()
    for row in rows:
        name, fold = row['filename'], int(row['fold'])
        if Path(name).name != name or name in filenames or fold not in range(1, 6):
            raise ValueError(f'Invalid or duplicate metadata entry: {name}')
        filenames.add(name)
        if not row['src_file'].strip():
            raise ValueError(f'Missing source group: {name}')
        if folds_by_source.setdefault(row['src_file'], fold) != fold:
            raise ValueError(f"Source {row['src_file']} spans folds; refusing leakage.")
        if 'gun' in row['category'].lower():
            raise ValueError(f'Unexpected potential gunshot class: {row["category"]}')
        if not (source/'audio'/name).is_file():
            raise FileNotFoundError(source/'audio'/name)
    # Never overwrite existing negative data, including an incomplete earlier run.
    for split in ['train', 'val', 'test']:
        folder = output/split/'non_gunshot'
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f'{folder} is nonempty; choose a fresh output folder.')
    for name in ['esc50_manifest.csv', 'esc50_summary.json']:
        if (output/name).exists():
            raise ValueError(f'{output/name} already exists.')
    for split in ['train', 'val', 'test']:
        (output/split/'non_gunshot').mkdir(parents=True, exist_ok=True)
        (output/split/'gunshot').mkdir(parents=True, exist_ok=True)
    counts = Counter()
    fields = ['output_path', 'source_filename', 'source_id', 'category', 'fold', 'split', 'start_s', 'end_s', 'label']
    with (output/'esc50_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            fold = int(row['fold'])
            split = 'test' if fold == test_fold else 'val' if fold == val_fold else 'train'
            y, _ = librosa.load(source/'audio'/row['filename'], sr=SR, mono=True)
            if not np.isfinite(y).all() or abs(len(y)-5*SR) > 24:
                raise ValueError(f"Expected finite five-second audio: {row['filename']}")
            y = librosa.util.fix_length(y, size=5*SR)
            for start in [0, 1, 2]:
                relative = Path(split)/'non_gunshot'/f"esc50_{Path(row['filename']).stem}_s{start}.wav"
                sf.write(output/relative, y[start*SR:start*SR+SAMPLES], SR, subtype='FLOAT')
                writer.writerow(dict(output_path=relative.as_posix(), source_filename=row['filename'],
                                     source_id=row['src_file'], category=row['category'], fold=fold,
                                     split=split, start_s=start, end_s=start+3, label=0))
                counts[split] += 1
    summary = dict(source='https://github.com/karolpiczak/ESC-50', input_files=len(rows),
                   output_windows=dict(counts), val_fold=val_fold, test_fold=test_fold,
                   sample_rate=SR, window_seconds=3, offsets_seconds=[0, 1, 2],
                   status='negative data prepared; gunshot examples still required')
    (output/'esc50_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Extracted ESC-50 folder containing audio/ and meta/')
    parser.add_argument('--out', default='dataset')
    parser.add_argument('--val-fold', type=int, default=4)
    parser.add_argument('--test-fold', type=int, default=5)
    args = parser.parse_args()
    prepare(args.source, args.out, args.val_fold, args.test_fold)
