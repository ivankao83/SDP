"""Continue run #3 with environmental negatives and source-disjoint replay.

The new corpus is development data after its initial frozen-model test. No
ShotSpotter or published test recordings enter optimization or selection.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import gunshot as g
from train_c3gd import CachedRecordings, binary_label, fingerprint, make_pair, save_feature
from evaluate_false_alarms import clip_bounds

HOLDOUT = {'MS-SNSD': ['Babble', 'CopyMachine'], 'DEMAND': ['NPARK', 'SPSQUARE']}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, default=Path('runs/run4_background_revision'))
    parser.add_argument('--epochs', type=int, default=6)
    a = parser.parse_args()
    if a.out.exists():
        raise ValueError('Output must be new.')
    torch.set_num_threads(2)
    random.seed(404)
    np.random.seed(404)
    torch.manual_seed(404)
    rng = np.random.default_rng(404)
    prev = Path('runs/c3gd_balanced/best.pt')
    checksum = fingerprint(prev)
    checkpoint = torch.load(prev, map_location='cpu', weights_only=True)
    cfg = dict(checkpoint['config'])
    cfg.update(initialization='run3_checkpoint', initialization_sha256=checksum,
               learning_rate=5e-5, seed=404, epochs=a.epochs,
               training_datasets=['original_replay', 'MAD_replay', 'C3GD_replay', 'MS-SNSD', 'DEMAND'],
               selection='validation negative rejection with old-domain recall/F1 floors; see protocol.json')
    a.out.mkdir(parents=True)
    cache = a.out / 'temporary_features'
    cache.mkdir()
    prior = json.loads(Path('runs/c3gd_balanced/split_manifest.json').read_text())
    splits = {'train': [], 'val': []}
    hashes = {'train': set(), 'val': set()}
    source_sets = {'train': set(), 'val': set()}
    def add(split, row, audio):
        path = cache / f'{split}_{len(splits[split])}.npy'
        save_feature(path, audio, cfg)
        splits[split].append({**row, 'feature_path': str(path.resolve())})
        hashes[split].add(hashlib.sha256(audio.tobytes()).hexdigest())
    # Replay only examples already assigned to train/validation in prior runs.
    for split in splits:
        for domain in ['original', 'mad']:
            for label in [0, 1]:
                pool = [r for r in prior[split] if r['domain'] == domain and binary_label(r) == label]
                count = min(400, len(pool)) if split == 'train' else len(pool)
                for index in sorted(rng.choice(len(pool), count, replace=False)):
                    row = pool[index]
                    path = Path(row['path'])
                    if domain == 'mad':
                        path = Path('dataset/MAD/MAD_dataset') / path
                    add(split, row, g.read_clip(path))
                print(f'Prepared {split} {domain} class {label}: {count}', flush=True)
        # Regenerate a bounded source-separated subset of paired C3GD replay.
        pool = [r for r in prior[split] if r['domain'] == 'c3gd_augmented' and binary_label(r) == 1]
        count = 400 if split == 'train' else 300
        for index in sorted(rng.choice(len(pool), min(count, len(pool)), replace=False)):
            row = pool[index]
            pair = make_pair(g.read_clip(Path(row['path'])), g.read_clip(Path(row['background_path'])),
                             rng, round(row['duration_seconds'] * g.SR), row['snr_db'], row['augmented_sample_rate'])
            for label, audio in zip(['1', '0'], pair):
                add(split, {**row, 'label': label}, audio)
        print(f'Prepared {split} C3GD replay pairs: {count}', flush=True)
    for dataset in HOLDOUT:
        for row in json.loads(Path(f'dataset/false_alarm_run4/{dataset}/manifest.json').read_text()):
            split = 'val' if row['category'] in HOLDOUT[dataset] else 'train'
            source_sets[split].add(row['path'])
            audio = g.read_clip(Path(row['path']))
            hashes[split].add(hashlib.sha256(audio.tobytes()).hexdigest())
            if hashlib.sha256(audio.tobytes()).hexdigest() in set(checkpoint['seen_hashes']):
                raise ValueError('New background overlaps previous development audio.')
            for start, end in clip_bounds(len(audio)):
                add(split, {'path': row['path'], 'label': '0', 'domain': 'new_background',
                            'dataset': dataset, 'category': row['category'],
                            'start_seconds': start / g.SR, 'end_seconds': end / g.SR}, audio[start:end])
    assert not source_sets['train'] & source_sets['val']
    assert not hashes['train'] & hashes['val'], 'Decoded audio crosses splits.'
    g.write_json(a.out / 'split_manifest.json', splits)
    protocol = {'initialization': str(prev), 'initialization_sha256': checksum,
                'new_corpus_status': 'Development after initial frozen run3 test, not independent final test.',
                'background_validation_categories': HOLDOUT,
                'split_rule': 'Entire categories/environments held out; no file windows cross train/validation.',
                'replay_rule': 'Original and MAD reuse prior splits; C3GD uses only prior train/val collection events.',
                'shotspotter_used_for_training_or_selection': False,
                'test_audio_used_for_training_or_selection': False,
                'selection': 'Maximize 0.5*negative-source-rejection-from-clip-scores + 0.5*negative-clip-rejection + 0.2*minimum old-domain F1; require old-domain recall and F1 within 0.05 of epoch0 at inherited threshold.',
                'selection_window_limitation': 'Source scores pool separate 10-second clip scores. Continuous full-file rechecks also inspect windows crossing clip boundaries.',
                'learning_rate': 5e-5, 'seed': 404,
                'examples': {s: dict(Counter(r['domain'] for r in rows)) for s, rows in splits.items()},
                'background_source_files': {s: len(v) for s, v in source_sets.items()}}
    g.write_json(a.out / 'protocol.json', protocol)
    print(json.dumps(protocol), flush=True)
    counts = Counter((r['domain'], binary_label(r)) for r in splits['train'])
    weights = [(3 if r['domain'] == 'new_background' else 1) / counts[(r['domain'], binary_label(r))] for r in splits['train']]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    loaders = {s: DataLoader(CachedRecordings(rows), batch_size=16,
                             sampler=sampler if s == 'train' else None,
                             collate_fn=g.collate_recordings) for s, rows in splits.items()}
    net = g.model_for(cfg)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer = torch.optim.AdamW(net.parameters(), lr=5e-5, weight_decay=.01)
    domains = np.array([r['domain'] for r in splits['val']])
    background = domains == 'new_background'
    old_domains = ['original', 'mad', 'c3gd_augmented']
    baseline, best, history = None, -float('inf'), []
    for epoch in range(a.epochs + 1):
        train_loss = None
        if epoch:
            train_loss, _, _ = g.run_epoch(net, loaders['train'], torch.device('cpu'), optimizer)
        loss, y, probabilities = g.run_epoch(net, loaders['val'], torch.device('cpu'))
        if baseline is None:
            baseline = {d: g.metrics(y[domains == d], probabilities[domains == d], cfg['threshold']) for d in old_domains}
            g.write_json(a.out / 'inherited_validation_metrics.json', baseline)
        candidates = []
        for threshold in sorted(set(np.linspace(.05, .95, 91).tolist() + [checkpoint['config']['threshold']])):
            reports = {d: g.metrics(y[domains == d], probabilities[domains == d], threshold) for d in old_domains}
            eligible = all(reports[d]['gunshot'][k] >= baseline[d]['gunshot'][k] - .05 for d in old_domains for k in ['recall', 'f1-score'])
            if not eligible:
                continue
            clip_rejection = float(np.mean(probabilities[background] <= threshold))
            file_flags = {}
            for r, p in zip(splits['val'], probabilities):
                if r['domain'] == 'new_background':
                    file_flags[r['path']] = file_flags.get(r['path'], False) or p > threshold
            file_rejection = 1 - sum(file_flags.values()) / len(file_flags)
            score = .5 * clip_rejection + .5 * file_rejection + .2 * min(v['gunshot']['f1-score'] for v in reports.values())
            candidates.append((score, threshold, reports, clip_rejection, file_rejection))
        row = {'epoch': epoch, 'train_loss': train_loss, 'validation_loss': loss, 'eligible': bool(candidates)}
        if candidates:
            score, threshold, reports, clip_rejection, file_rejection = max(candidates, key=lambda v: v[0])
            row.update(score=score, threshold=threshold, negative_clip_rejection=clip_rejection,
                       negative_source_rejection_from_clip_scores=file_rejection,
                       gunshot_recall_by_domain={d: r['gunshot']['recall'] for d, r in reports.items()},
                       gunshot_f1_by_domain={d: r['gunshot']['f1-score'] for d, r in reports.items()})
            if score > best:
                best = score
                cfg['threshold'] = threshold
                torch.save({**checkpoint, 'state_dict': net.state_dict(), 'config': dict(cfg), 'epoch': epoch,
                            'seen_hashes': sorted(set(checkpoint['seen_hashes']) | hashes['train'] | hashes['val']),
                            'seen_sources': sorted(set(checkpoint['seen_sources']) | source_sets['train'] | source_sets['val'])}, a.out / 'best.pt')
                g.write_json(a.out / 'config.json', cfg)
                g.write_json(a.out / 'best_validation_metrics.json', reports)
                g.write_json(a.out / 'best_validation_selection.json', row)
        history.append(row)
        g.write_json(a.out / 'history.json', history)
        print(json.dumps(row), flush=True)
    assert fingerprint(prev) == checksum
    print('Finished; previous checkpoint unchanged. New checkpoint frozen.', flush=True)


if __name__ == '__main__':
    main()
