"""Add C3GD training data without using ShotSpotter for training or selection.

Entire C3GD collection events are assigned to train/validation/test before
inference. Previous original/MAD train/validation assignments are preserved.
Each new positive is paired with background from its own split at the same
duration/bandwidth. This avoids a padding or recording-bandwidth label shortcut.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import gunshot as g
from train_mad import select_threshold

EVENT_SPLITS = {'oh_farm': 'train', 'nj_farm': 'train',
                'ny_gravel_pit': 'val', 'nj_farm_2': 'test'}


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event_split(row):
    return EVENT_SPLITS[row['event_id']]


def binary_label(row):
    label = int(row['label'])
    allowed = range(7) if row['domain'] == 'mad' else (0, 1)
    if label not in allowed:
        raise ValueError(f'Unexpected {row["domain"]} label: {label}')
    return int(label == 1)


def bandwidth(y, sample_rate):
    if sample_rate == g.SR:
        return y.astype(np.float32)
    low = librosa.resample(y, orig_sr=g.SR, target_sr=sample_rate)
    return librosa.util.fix_length(librosa.resample(low, orig_sr=sample_rate, target_sr=g.SR),
                                   size=len(y)).astype(np.float32)


def make_pair(shot, background, rng, length, snr, sample_rate):
    """Same-length positive/negative pair with identical background/bandwidth."""
    background = np.asarray(background, dtype=np.float32)
    start = int(rng.integers(0, max(1, len(background) - length + 1)))
    noise = librosa.util.fix_length(background[start:start + length], size=length).copy()
    if np.sqrt(np.mean(noise ** 2)) < 1e-8:
        noise = rng.normal(0, .001, size=length).astype(np.float32)
    foreground = np.zeros(length, dtype=np.float32)
    shot = np.asarray(shot[:length], dtype=np.float32)
    offset = int(rng.integers(0, length - len(shot) + 1))
    foreground[offset:offset + len(shot)] = shot
    # Fixed reproducible randomized echoes simulate variation in propagation.
    for delay, gain in [(int(.045 * g.SR), .20), (int(.11 * g.SR), .08)]:
        if delay < length:
            foreground[delay:] += gain * foreground[:-delay].copy()
    signal_rms = np.sqrt(np.mean(foreground ** 2))
    if signal_rms < 1e-10:
        raise ValueError('C3GD positive is silent.')
    noise_rms = np.sqrt(np.mean(noise ** 2))
    foreground *= noise_rms * (10 ** (snr / 20)) / signal_rms
    positive, negative = bandwidth(foreground + noise, sample_rate), bandwidth(noise, sample_rate)
    # Prevent clipping while preserving the signal/background relationship.
    peak = max(float(np.abs(positive).max()), float(np.abs(negative).max()), 1.)
    return positive / peak, negative / peak


def save_feature(path, y, cfg):
    spectra = np.stack([g.mel(w, cfg['frontend']) for _, w in g.audio_windows(y)])
    np.save(path, g.features(spectra, cfg).numpy().astype(np.float16))


class CachedRecordings(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return torch.from_numpy(np.load(row['feature_path']).astype(np.float32)), binary_label(row)


def validation(net, loader, rows):
    loss, y, probability = g.run_epoch(net, loader, torch.device('cpu'))
    domains = np.array([row['domain'] for row in rows])
    threshold, report = select_threshold(y, probability, domains)
    by_domain = {domain: g.metrics(y[domains == domain], probability[domains == domain], threshold)
                 for domain in sorted(set(domains))}
    score = min(r['gunshot']['f1-score'] for r in by_domain.values())
    return loss, threshold, report, by_domain, score


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--c3gd', type=Path, default=Path('dataset/C3GD/C3GD'))
    p.add_argument('--previous', type=Path, default=Path('runs/mixed_full_spectrum'))
    p.add_argument('--out', type=Path, default=Path('runs/c3gd_transfer'))
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--domain-balanced', action='store_true')
    p.add_argument('--learning-rate', type=float, default=.0002)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Choose a fresh output directory.')
    torch.set_num_threads(2)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    checkpoint = a.previous / 'best.pt'
    checksum = fingerprint(checkpoint)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = dict(ckpt['config'])
    cfg.update(initialization='run2_checkpoint', initialization_sha256=checksum,
               training_datasets=['original', 'MAD', 'C3GD'], c3gd_event_splits=EVENT_SPLITS,
               seed=42, epochs=a.epochs, learning_rate=a.learning_rate,
               sampling='balanced_domain_and_class' if a.domain_balanced else 'recording_shuffle',
               selection='maximum minimum validation gunshot F1 across original, MAD, C3GD augmented',
               augmentation={'snr_db': [-5, 20], 'sample_rates': [12000, 16000, 24000],
                             'paired_durations_seconds': [1, 3], 'echoes_seconds': [.045, .11]})
    previous = json.loads((a.previous / 'split_manifest.json').read_text())
    splits = {'train': [], 'val': []}
    base_cache = Path('dataset/MAD/full_spectrum_cache_v1')
    background_pools = {}
    for split in splits:
        for row in previous[split]:
            feature_path = base_cache / (row['cache_key'] + '.npy')
            if not feature_path.is_file():
                raise FileNotFoundError(feature_path)
            splits[split].append({**row, 'feature_path': str(feature_path.resolve())})
        background_pools[split] = [r['path'] for r in previous[split]
                                   if r['domain'] == 'original' and r['label'] == '0']
        if not background_pools[split]:
            raise ValueError('Missing split-specific negative backgrounds.')
    with (a.c3gd / 'metadata.csv').open(newline='') as f:
        c3rows = list(csv.DictReader(f))
    if set(r['event_id'] for r in c3rows) != set(EVENT_SPLITS):
        raise ValueError('Unexpected C3GD events; review the source split protocol.')
    if any(not (a.c3gd / 'data' / r['filename']).is_file() for r in c3rows):
        raise FileNotFoundError('Incomplete C3GD audio download.')
    a.out.mkdir(parents=True)
    cache = Path('dataset/C3GD/paired_cache_v1')
    cache.mkdir(exist_ok=True)
    protocol = {'mode': 'training_with_source_separated_C3GD', 'seed': 42,
                'previous_checkpoint_sha256': checksum, 'event_splits': EVENT_SPLITS,
                'shotspotter_used_for_training': False, 'shotspotter_used_for_validation': False,
                'shotspotter_used_for_threshold_selection': False,
                'c3gd_recordings_by_event': dict(Counter(r['event_id'] for r in c3rows)),
                'c3gd_test_audio_read_by_training': False,
                'threshold_selection': cfg['selection'], 'augmentation': cfg['augmentation'],
                'sampling': cfg['sampling'], 'learning_rate': a.learning_rate,
                'note': 'C3GD validation uses synthetic positive/background pairs, not natural negative recordings.'}
    g.write_json(a.out / 'protocol.json', protocol)
    g.write_json(a.out / 'c3gd_test_manifest.json', [r for r in c3rows if event_split(r) == 'test'])
    hashes = {'train': set(), 'val': set()}
    for split in hashes:
        for row in previous[split]:
            hashes[split].add(json.loads((base_cache / (row['cache_key'] + '.json')).read_text())['hash'])
    c3_sources = {'train': set(), 'val': set(), 'test': set()}
    rng = np.random.default_rng(42)
    c3_counter = {'train': 0, 'val': 0}
    for i, row in enumerate(c3rows):
        split = event_split(row)
        group = f"c3gd:{row['event_id']}:{row['file_id']}:{row['clip_id']}"
        c3_sources[split].add(group)
        if split == 'test':
            continue
        path = a.c3gd / 'data' / row['filename']
        shot = g.read_clip(path)
        digest = hashlib.sha256(shot.tobytes()).hexdigest()
        if digest in set(ckpt['seen_hashes']):
            raise ValueError(f'C3GD matches previously used audio: {path}')
        hashes[split].add(digest)
        background_path = background_pools[split][int(rng.integers(len(background_pools[split])))]
        background = g.read_clip(background_path)
        length = g.SR if c3_counter[split] % 2 == 0 else g.SAMPLES
        snr, sr = float(rng.uniform(-5, 20)), int(rng.choice([12000, 16000, 24000]))
        positive, negative = make_pair(shot, background, rng, length, snr, sr)
        for label, audio in [('1', positive), ('0', negative)]:
            feature_path = cache / f'{i}_{label}.npy'
            if not feature_path.exists():
                save_feature(feature_path, audio, cfg)
            hashes[split].add(hashlib.sha256(audio.tobytes()).hexdigest())
            splits[split].append({'path': str(path.resolve()) if label == '1' else background_path,
                                  'label': label, 'domain': 'c3gd_augmented', 'event_id': row['event_id'],
                                  'source_group': group, 'background_path': background_path,
                                  'duration_seconds': length / g.SR, 'snr_db': snr if label == '1' else None,
                                  'augmented_sample_rate': sr, 'feature_path': str(feature_path.resolve())})
        c3_counter[split] += 1
        if sum(c3_counter.values()) % 500 == 0:
            print(f'Prepared {sum(c3_counter.values())}/8009 C3GD source recordings', flush=True)
    assert not hashes['train'] & hashes['val'], 'Decoded audio crosses train/validation splits.'
    assert not c3_sources['train'] & c3_sources['val']
    assert not (c3_sources['train'] | c3_sources['val']) & c3_sources['test']
    g.write_json(a.out / 'split_manifest.json', splits)
    protocol['training_examples'] = len(splits['train'])
    protocol['validation_examples'] = len(splits['val'])
    protocol['c3gd_original_recordings_used'] = c3_counter
    g.write_json(a.out / 'protocol.json', protocol)
    print(json.dumps(protocol), flush=True)
    sampler = None
    if a.domain_balanced:
        strata = Counter((r['domain'], binary_label(r)) for r in splits['train'])
        sampling_weights = [1 / strata[(r['domain'], binary_label(r))] for r in splits['train']]
        sampler = WeightedRandomSampler(sampling_weights, len(splits['train']), replacement=True)
    loaders = {s: DataLoader(CachedRecordings(rows), batch_size=16,
                             shuffle=s == 'train' and sampler is None,
                             sampler=sampler if s == 'train' else None,
                             collate_fn=g.collate_recordings) for s, rows in splits.items()}
    net = g.model_for(cfg)
    net.load_state_dict(ckpt['state_dict'])
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.learning_rate, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=.5)
    positives = sum(binary_label(r) for r in splits['train'])
    weight = torch.tensor((len(splits['train']) - positives) / positives)
    assert 0 < positives < len(splits['train']) and weight > 0
    if a.domain_balanced:
        weight = torch.tensor(1.)  # The sampler already balances positive/negative examples.
    seen_hashes = sorted(set(ckpt['seen_hashes']) | hashes['train'] | hashes['val'])
    seen_sources = sorted(set(ckpt['seen_sources']) | c3_sources['train'] | c3_sources['val'])
    best, stale, history = -1., 0, []
    for epoch in range(a.epochs + 1):
        total = 0.
        if epoch:
            net.train()
            for bags, labels in loaders['train']:
                assert torch.all((labels == 0) | (labels == 1)), 'BCE requires binary labels.'
                optimizer.zero_grad()
                logits = g.recording_logits(net, bags, torch.device('cpu'))
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.)
                optimizer.step()
                total += loss.item() * len(labels)
        vl, threshold, report, domains, score = validation(net, loaders['val'], splits['val'])
        row = {'epoch': epoch, 'train_loss': total / len(splits['train']) if epoch else None,
               'validation_loss': vl, 'validation_min_domain_f1': score,
               'validation_domain_f1': {k: v['gunshot']['f1-score'] for k, v in domains.items()},
               'threshold': threshold}
        history.append(row)
        g.write_json(a.out / 'history.json', history)
        print(json.dumps(row), flush=True)
        if score > best:
            best, stale = score, 0
            cfg['threshold'] = threshold
            torch.save({'state_dict': net.state_dict(), 'config': dict(cfg), 'epoch': epoch,
                        'seen_hashes': seen_hashes, 'seen_sources': seen_sources,
                        'c3gd_training_events': ['oh_farm', 'nj_farm'],
                        'c3gd_validation_events': ['ny_gravel_pit']}, a.out / 'best.pt')
            g.write_json(a.out / 'config.json', cfg)
            g.write_json(a.out / 'best_validation_by_domain.json', domains)
            g.write_json(a.out / 'best_validation_metrics.json', report)
        else:
            stale += 1
        scheduler.step(vl)
        if epoch and stale >= 3:
            break
    assert fingerprint(checkpoint) == checksum, 'Previous checkpoint changed.'
    print(f'Finished. Best minimum validation F1 {best:.4f}. No test inference performed.', flush=True)


if __name__ == '__main__':
    main()
