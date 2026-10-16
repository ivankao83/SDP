"""Train a full-spectrum binary MAD model with video-disjoint validation.

The official test split is used only to exclude shared source videos, never for
training, checkpoint selection, or threshold selection. Labels other than
shooting are provisional negatives; shelling/mixed clips require review.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gunshot as g


def source(row):
    if row.get('domain') == 'original':
        return 'original:' + row['path']
    return parse_qs(urlparse(row['youtube url']).query).get('v', [row['youtube url']])[0]


class Recordings(Dataset):
    def __init__(self, rows, cache):
        self.rows, self.cache = rows, cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return torch.from_numpy(np.load(self.cache / (row['cache_key'] + '.npy')).astype(np.float32)), int(row['label'] == '1')


def select_threshold(y, probabilities, domains=None):
    # Optimize validation F1; never inspect test probabilities for selection.
    candidates = np.linspace(.05, .95, 91)
    reports = [g.metrics(y, probabilities, float(t)) for t in candidates]
    if domains is None:
        scores = [r['gunshot']['f1-score'] for r in reports]
    else:
        scores = [min(g.metrics(y[domains == d], probabilities[domains == d], float(t))['gunshot']['f1-score']
                      for d in set(domains)) for t in candidates]
    winner = max(range(len(reports)), key=lambda i: scores[i])
    return float(candidates[winner]), reports[winner]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, default=Path('dataset/MAD/MAD_dataset'))
    p.add_argument('--out', type=Path, default=Path('runs/mad_full_spectrum'))
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--include-original', type=Path, help='Also use existing train/val audio from this dataset root')
    a = p.parse_args()
    if (a.out / 'best.pt').exists():
        raise ValueError('Choose a new output directory; best.pt already exists.')
    torch.set_num_threads(2)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.data.parent / 'full_spectrum_cache_v1'
    cache.mkdir(exist_ok=True)
    with (a.data / 'training.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    with (a.data / 'test.csv').open(newline='') as f:
        test_rows = list(csv.DictReader(f))
    test_sources = set(map(source, test_rows))
    excluded = sorted(set(map(source, rows)) & test_sources)
    rows = [r for r in rows if source(r) not in test_sources]
    mad_count = len(rows)
    for row in rows:
        row['domain'] = 'mad'
    if a.include_original:
        for split in ['train', 'val']:
            for cls, label in g.CLASSES.items():
                for path in sorted((a.include_original / split / cls).rglob('*')):
                    if path.suffix.lower() in {'.wav', '.flac', '.ogg'}:
                        rows.append({'path': str(path.resolve()), 'label': str(label),
                                     'domain': 'original', 'original_split': split})
    cfg = dict(model='compact_cnn', experiment=2, bands=128, frontend='notebook',
               sample_rate=g.SR, samples=g.SAMPLES, median_reference=1e-10,
               threshold=.5, seed=42, class_to_index=g.CLASSES,
               window_hop_samples=g.SR, pooling='max_logit_per_recording',
               metric_unit='recording', initialization='random')
    hashes = {}
    for i, row in enumerate(rows):
        key = hashlib.sha256(row['path'].encode()).hexdigest()
        row['cache_key'] = key
        meta = cache / (key + '.json')
        if not meta.exists():
            y = g.read_clip(a.data / row['path'])
            spectra = np.stack([g.mel(w, cfg['frontend']) for _, w in g.audio_windows(y)])
            x = g.features(spectra, cfg).numpy().astype(np.float16)
            np.save(cache / (key + '.npy'), x)
            g.write_json(meta, {'hash': hashlib.sha256(y.tobytes()).hexdigest()})
        hashes[row['path']] = json.loads(meta.read_text())['hash']
        if (i + 1) % 500 == 0:
            print(f'Prepared {i + 1}/{len(rows)} recordings', flush=True)
    labels = np.array([int(r['label'] == '1') for r in rows[:mad_count]])
    groups = np.array([source(r) for r in rows[:mad_count]])
    train_idx, val_idx = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(rows[:mad_count], labels, groups))
    tr, va = [rows[i] for i in train_idx], [rows[i] for i in val_idx]
    tr += [r for r in rows[mad_count:] if r['original_split'] == 'train']
    va += [r for r in rows[mad_count:] if r['original_split'] == 'val']
    val_hashes = {hashes[r['path']] for r in va}
    tr = [r for r in tr if hashes[r['path']] not in val_hashes]
    assert not set(map(source, tr)) & set(map(source, va))
    assert not (set(map(source, tr)) | set(map(source, va))) & test_sources
    g.write_json(a.out / 'split_manifest.json', {'train': tr, 'val': va,
                 'excluded_test_sources': excluded, 'seed': 42})
    print(f'Train {len(tr)}, validation {len(va)}; excluded {len(excluded)} test-shared videos', flush=True)
    train_loader = DataLoader(Recordings(tr, cache), batch_size=16, shuffle=True, collate_fn=g.collate_recordings)
    val_loader = DataLoader(Recordings(va, cache), batch_size=16, collate_fn=g.collate_recordings)
    net = g.model_for(cfg)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=.5)
    positive = sum(r['label'] == '1' for r in tr)
    weight = torch.tensor((len(tr) - positive) / positive)
    best, stale, history = -1., 0, []
    for epoch in range(1, a.epochs + 1):
        net.train()
        total = 0.
        for bags, y in train_loader:
            optimizer.zero_grad()
            logits = g.recording_logits(net, bags, torch.device('cpu'))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=weight)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
            total += loss.item() * len(y)
        vl, y, prob = g.run_epoch(net, val_loader, torch.device('cpu'))
        domains = np.array([r['domain'] for r in va])
        threshold, report = select_threshold(y, prob, domains)
        domain_reports = {d: g.metrics(y[domains == d], prob[domains == d], threshold) for d in set(domains)}
        score = min(r['gunshot']['f1-score'] for r in domain_reports.values())
        row = dict(epoch=epoch, train_loss=total / len(tr), val_loss=vl,
                   val_gunshot_f1=score, val_gunshot_recall=report['gunshot']['recall'],
                   val_gunshot_precision=report['gunshot']['precision'], threshold=threshold,
                   val_domain_f1={d: r['gunshot']['f1-score'] for d, r in domain_reports.items()})
        history.append(row)
        g.write_json(a.out / 'history.json', history)
        print(json.dumps(row), flush=True)
        scheduler.step(vl)
        if score > best:
            best, stale = score, 0
            cfg['threshold'] = threshold
            torch.save({'state_dict': net.state_dict(), 'config': dict(cfg), 'epoch': epoch,
                        'seen_hashes': list(set(hashes.values())),
                        'seen_sources': sorted(set(map(source, tr + va)))}, a.out / 'best.pt')
            g.write_json(a.out / 'config.json', cfg)
            g.write_json(a.out / 'best_validation_metrics.json', report)
            g.write_json(a.out / 'best_validation_by_domain.json', domain_reports)
        else:
            stale += 1
        if stale >= 5:
            break
    print(f'Training complete; best validation F1 {best:.4f}. Test remains unevaluated.', flush=True)


if __name__ == '__main__':
    main()
