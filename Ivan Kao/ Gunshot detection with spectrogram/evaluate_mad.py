"""Evaluate the existing binary checkpoint on MAD's official test recordings.

Uses the checkpoint's original frontend and recording max pooling. MAD label 1
is gunshot; all other labels are provisional negatives (mixed events need review).
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import gunshot as g

CLASSES = ['communication', 'gunshot', 'footsteps', 'shelling', 'vehicle', 'helicopter', 'fighter']


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=Path('runs/exp1/best.pt'))
    p.add_argument('--out', type=Path, default=Path('runs/exp1/eval_mad'))
    a = p.parse_args()
    torch.set_num_threads(2)
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    cfg = ckpt['config']
    threshold = cfg.get('threshold', .5)
    net = g.model_for(cfg)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    with (a.data / 'test.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    test_sources = {parse_qs(urlparse(r['youtube url']).query).get('v', [r['youtube url']])[0] for r in rows}
    if test_sources & set(ckpt.get('seen_sources', [])):
        raise ValueError('Test source videos overlap training/validation sources.')
    missing = [r['path'] for r in rows if not (a.data / r['path']).is_file()]
    if missing:
        raise ValueError(f'{len(missing)} missing audio files, e.g. {missing[0]}')
    a.out.mkdir(parents=True, exist_ok=True)
    labels, probs, original = [], [], []
    seen = set(ckpt['seen_hashes'])
    with (a.out / 'test_predictions.csv').open('w', newline='') as f, torch.inference_mode():
        writer = csv.writer(f)
        writer.writerow(['path', 'mad_class', 'true_label', 'gunshot_probability', 'predicted_label', 'duration_seconds'])
        for i, row in enumerate(rows):
            y = g.read_clip(a.data / row['path'])
            if hashlib.sha256(y.tobytes()).hexdigest() in seen:
                raise ValueError(f'Training/validation duplicate: {row["path"]}')
            # Keep only one recording's features in RAM at a time.
            spectra = np.stack([g.mel(w, cfg['frontend']) for _, w in g.audio_windows(y)])
            x = g.features(spectra, cfg)
            probability = torch.cat([net(b).flatten() for b in x.split(64)]).max().sigmoid().item()
            cls = int(row['label'])
            label = int(cls == 1)
            labels.append(label)
            probs.append(probability)
            original.append(cls)
            writer.writerow([row['path'], CLASSES[cls], label, probability, int(probability > threshold), len(y) / g.SR])
            if (i + 1) % 100 == 0:
                print(f'Evaluated {i + 1}/{len(rows)} recordings', flush=True)
    labels, probs, original = np.array(labels), np.array(probs), np.array(original)
    report = g.metrics(labels, probs, threshold)
    report.update(metric_unit='recording', pooling='max_window_probability', dataset='MAD official test split',
                  checkpoint=str(a.checkpoint.resolve()), binary_mapping='MAD label 1 positive; all other labels negative',
                  checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest())
    report['per_mad_class'] = {
        name: {'recordings': int((original == i).sum()),
               'predicted_gunshot': int(((original == i) & (probs > threshold)).sum())}
        for i, name in enumerate(CLASSES)
    }
    g.write_json(a.out / 'test_metrics.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
