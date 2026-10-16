"""Test a frozen model on the predeclared, held-out C3GD collection event.

C3GD contains gunshot recordings only. Scores are positive-recording recall;
negative-class accuracy, false-alarm rate, precision, and ROC AUC are unavailable.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import gunshot as g


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, default=Path('dataset/C3GD/C3GD'))
    p.add_argument('--run', type=Path, default=Path('runs/c3gd_transfer'))
    a = p.parse_args()
    out = a.run / 'eval_c3gd'
    if out.exists():
        raise ValueError('Evaluation output already exists.')
    path = a.run / 'best.pt'
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    cfg = ckpt['config']
    threshold = cfg['threshold']
    net = g.model_for(cfg)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    torch.set_num_threads(2)
    rows = json.loads((a.run / 'c3gd_test_manifest.json').read_text())
    held_events = {r['event_id'] for r in rows}
    if held_events & set(ckpt['c3gd_training_events'] + ckpt['c3gd_validation_events']):
        raise ValueError('Test event overlaps training/validation.')
    seen_hashes, seen_sources = set(ckpt['seen_hashes']), set(ckpt['seen_sources'])
    predictions = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            group = f"c3gd:{row['event_id']}:{row['file_id']}:{row['clip_id']}"
            if group in seen_sources:
                raise ValueError('Test shot source overlaps training/validation.')
            y = g.read_clip(a.data / 'data' / row['filename'])
            if hashlib.sha256(y.tobytes()).hexdigest() in seen_hashes:
                raise ValueError('Test audio exactly matches training/validation audio.')
            x = g.features(np.stack([g.mel(w, cfg['frontend']) for _, w in g.audio_windows(y)]), cfg)
            prob = net(x).flatten().max().sigmoid().item()
            predictions.append({'filename': row['filename'], 'event_id': row['event_id'],
                                'class_name': row['class_name'], 'mic_id': row['mic_id'],
                                'is_phone': row['is_phone'], 'true_label': 1,
                                'gunshot_probability': prob, 'predicted_gunshot': int(prob > threshold)})
            if (i + 1) % 500 == 0:
                print(f'Tested {i + 1}/{len(rows)} C3GD held-out recordings', flush=True)
    out.mkdir()
    with (out / 'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    def summary(records):
        detected = sum(r['predicted_gunshot'] for r in records)
        return {'recordings': len(records), 'detected': detected, 'missed': len(records) - detected,
                'recording_gunshot_recall': detected / len(records)}
    report = {'mode': 'test_only', 'test_events': sorted(held_events), 'threshold': threshold,
              'checkpoint_sha256_before': checksum, 'checkpoint_sha256_after': hashlib.sha256(path.read_bytes()).hexdigest(),
              'metric_unit': 'recording', **summary(predictions),
              'limitations': 'Positive-only dataset. No precision, two-class accuracy, or false-alarm rate can be measured.'}
    assert report['checkpoint_sha256_before'] == report['checkpoint_sha256_after']
    for field in ['class_name', 'mic_id', 'is_phone']:
        groups = defaultdict(list)
        for row in predictions:
            groups[row[field]].append(row)
        report['by_' + field] = {k: summary(v) for k, v in sorted(groups.items())}
    g.write_json(out / 'test_metrics.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
