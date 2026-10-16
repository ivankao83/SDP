"""Gunshot classification: paper-guided reconstruction, not original author code.
Sources: https://github.com/gpinyero/EUSIPCO2023
https://eurasip.org/Proceedings/Eusipco/Eusipco2023/pdfs/0000206.pdf
Assumptions: torchvision single-channel models with random initialization;
per-clip full-spectrum min-max normalization before cropping; training-only
mean of per-clip power medians; LeakyReLU slope .01; scheduler factor .1;
best validation-loss checkpoint. Median/max reference shifts can cancel under
per-clip min-max scaling. Original training code/weights were not released.
Notebook frontend produces 161 frames; paper159 is an inferred alternative.
Accepts finite, nonempty recordings of any duration in data/{train,val,test}/
{gunshot,non_gunshot}. Each positive file must contain at least one gunshot.
Three-second windows at a one-second hop cover each recording; short inputs are
right-zero-padded. Max pooling combines window logits into one recording label
(multiple-instance learning), not one positive label per window. This is an
adaptation beyond the paper's fixed-clip protocol. Metrics are per recording.
Longer files have more opportunities for false alarms; evaluate realistic lengths.
Padding is identical for both classes, but matching their duration distributions
is still important to avoid duration/padding shortcuts. Split by source/session.
Features are held in RAM; memory and compute grow with total recording duration.
Use --batch-size 1 for long files. No automatic trimming or time compression.
Run with --help for commands. No trained weights included.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

SR, SAMPLES = 24000, 72000
CLASSES = {'non_gunshot': 0, 'gunshot': 1}
EXPERIMENTS = {1: ('max', False, False), 2: ('max', False, True),
               3: ('max', True, False), 4: ('max', True, True),
               5: ('medians', True, False), 6: ('medians', True, True)}


def mel(y, frontend):
    # Notebook uses 1024-point FFT with librosa's centered padding (161 frames).
    # Alternative reproduces the paper's stated 159 frames, using FFT=900.
    return librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=1024 if frontend == 'notebook' else 900,
        win_length=900, hop_length=450, window='hann', center=frontend == 'notebook',
        power=2.0, n_mels=128, fmin=0, fmax=12000, htk=False, norm='slaney')


def read_clip(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if not np.isfinite(y).all() or not len(y):
        raise ValueError(f'Empty or nonfinite audio: {path}')
    return y


def window_starts(length, hop=SR):
    if length < 1 or not 1 <= hop <= SAMPLES:
        raise ValueError('Audio must be nonempty; hop must be between one sample and three seconds.')
    starts = list(range(0, max(length-SAMPLES+1, 1), hop))
    if length > SAMPLES and starts[-1] != length-SAMPLES:
        starts.append(length-SAMPLES)
    return starts


def audio_windows(y, hop=SR):
    for start in window_starts(len(y), hop):
        yield start, librosa.util.fix_length(y[start:start+SAMPLES], size=SAMPLES)


def recording_features(spectra, cfg):
    return [features(windows, cfg) for windows in spectra]


def load_split(root, split, frontend):
    arrays, labels, names, hashes = [], [], [], []
    for cls, label in CLASSES.items():
        folder = Path(root) / split / cls
        files = sorted(p for p in folder.rglob('*') if p.suffix.lower() in {'.wav', '.flac', '.ogg'})
        if not files:
            raise ValueError(f'No WAV/FLAC/OGG files under {folder}')
        for p in files:
            y = read_clip(p)
            arrays.append(np.stack([mel(window, frontend) for _, window in audio_windows(y)]))
            labels.append(label)
            names.append(str(p.resolve()))
            hashes.append(hashlib.sha256(y.tobytes()).hexdigest())
    return arrays, np.array(labels), names, hashes


def features(spectra, cfg):
    reference, normalize, _ = EXPERIMENTS[cfg['experiment']]
    out = []
    for s in spectra:
        db = librosa.power_to_db(s, ref=np.max if reference == 'max' else cfg['median_reference'],
                                 amin=1e-10, top_db=80.0)
        # Explicit assumption: per-clip scaling over the full spectrogram before cropping.
        if normalize:
            low, high = float(db.min()), float(db.max())
            db = (db-low) / max(high-low, 1e-8)
        out.append(db[:cfg['bands']])
    return torch.from_numpy(np.stack(out)[:, None].astype(np.float32))


def replace_relu(module):
    for name, child in module.named_children():
        if isinstance(child, (nn.ReLU, nn.ReLU6)):
            setattr(module, name, nn.LeakyReLU(0.01, inplace=True))
        else:
            replace_relu(child)


def model_for(cfg):
    name = cfg['model']
    if name == 'compact_cnn':
        # Full-spectrum input; early averaging keeps CPU inference inexpensive.
        return nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Dropout(.25), nn.Linear(64, 1))
    if name == 'vgg16':
        net = models.vgg16(weights=None)
        net.features[0] = nn.Conv2d(1, 64, 3, padding=1)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
    elif name == 'resnet34':
        net = models.resnet34(weights=None)
        net.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        net.fc = nn.Linear(net.fc.in_features, 1)
    else:
        net = models.mobilenet_v2(weights=None)
        net.features[0][0] = nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
    if EXPERIMENTS[cfg['experiment']][2]:
        replace_relu(net)
    return net


def device_for(name):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)


def collate_recordings(batch):
    bags, labels = zip(*batch)
    return list(bags), torch.tensor(labels, dtype=torch.float32)


def loader(x, labels, batch, shuffle=False):
    if isinstance(x, torch.Tensor):  # Also support single-window test callers.
        return DataLoader(TensorDataset(x, torch.tensor(labels, dtype=torch.float32)),
                          batch_size=batch, shuffle=shuffle, num_workers=0)
    return DataLoader(list(zip(x, labels)), batch_size=batch, shuffle=shuffle,
                      num_workers=0, collate_fn=collate_recordings)


def recording_logits(net, bags, device):
    if isinstance(bags, torch.Tensor):
        return net(bags.to(device)).flatten()
    sizes = [len(bag) for bag in bags]
    # One loss/label per recording. A positive bag needs only one positive window.
    window_logits = net(torch.cat(bags).to(device)).flatten()
    return torch.stack([part.max() for part in window_logits.split(sizes)])


def run_epoch(net, batches, device, optimizer=None):
    net.train(optimizer is not None)
    total, probabilities, labels = 0., [], []
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in batches:
            y = y.to(device)
            logits = recording_logits(net, x, device)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            probabilities.extend(logits.detach().sigmoid().cpu().tolist())
            labels.extend(y.cpu().tolist())
    return total / len(labels), np.array(labels, dtype=int), np.array(probabilities)


def metrics(y, p, threshold=.5):
    pred = (p > threshold).astype(int)
    report = classification_report(y, pred, labels=[0, 1], target_names=['non_gunshot', 'gunshot'],
                                   output_dict=True, zero_division=0)
    report['confusion_matrix_rows_true_columns_predicted'] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    report['roc_auc'] = float(roc_auc_score(y, p)) if len(set(y)) == 2 else None
    report['threshold'] = threshold
    return report


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding='utf-8')


def train(a):
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'best.pt').exists():
        raise ValueError('Output already contains best.pt. Choose a new --out directory.')
    cfg = dict(model=a.model, experiment=a.experiment, bands=a.bands, frontend=a.frontend,
               seed=a.seed, sample_rate=SR, samples=SAMPLES, epochs=a.epochs, batch_size=a.batch_size,
               learning_rate=1e-4, scheduler_patience=10, scheduler_factor=.1,
               class_to_index=CLASSES, initialization='random', threshold=.5,
               window_hop_samples=SR, pooling='max_logit_per_recording', padding='right_zero',
               metric_unit='recording')
    tr = load_split(a.data, 'train', a.frontend)
    va = load_split(a.data, 'val', a.frontend)
    if set(tr[3]) & set(va[3]):
        raise ValueError('Identical decoded audio occurs in train and val; remove leakage.')
    cfg['median_reference'] = max(float(np.mean([np.median(windows) for windows in tr[0]])), 1e-10)
    xtr, xva = recording_features(tr[0], cfg), recording_features(va[0], cfg)
    train_batches = loader(xtr, tr[1], a.batch_size, True)
    val_batches = loader(xva, va[1], a.batch_size)
    device = device_for(a.device)
    net = model_for(cfg).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.1, patience=10)
    write_json(out/'config.json', cfg)
    write_json(out/'split_manifest.json', {'train': tr[2], 'val': va[2]})
    history, best = [], float('inf')
    for epoch in range(1, a.epochs+1):
        tl, _, _ = run_epoch(net, train_batches, device, optimizer)
        vl, y, p = run_epoch(net, val_batches, device)
        row = dict(epoch=epoch, train_loss=tl, val_loss=vl,
                   val_gunshot_f1=metrics(y, p)['gunshot']['f1-score'], lr=optimizer.param_groups[0]['lr'])
        scheduler.step(vl)
        history.append(row)
        write_json(out/'history.json', history)
        print(json.dumps(row), flush=True)
        if vl < best:
            best = vl
            torch.save({'state_dict': net.state_dict(), 'config': cfg, 'epoch': epoch,
                        'seen_hashes': tr[3]+va[3]}, out/'best.pt')
            write_json(out/'best_validation_metrics.json', metrics(y, p))
    print(f'Saved {out / "best.pt"}. Run evaluate separately on the held-out test set.')


def restore(a):
    device = device_for(a.device)
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    net = model_for(ckpt['config'])
    net.load_state_dict(ckpt['state_dict'])
    return net.to(device), ckpt, device


def evaluate(a):
    net, ckpt, device = restore(a)
    data = load_split(a.data, 'test', ckpt['config']['frontend'])
    if set(data[3]) & set(ckpt['seen_hashes']):
        raise ValueError('Test audio duplicates training/validation audio; use an independent test set.')
    loss, y, p = run_epoch(net, loader(recording_features(data[0], ckpt['config']), data[1], a.batch_size), device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    threshold = ckpt['config'].get('threshold', .5)
    report = metrics(y, p, threshold)
    report['loss'] = loss
    report['metric_unit'] = 'recording'
    report['pooling'] = 'max_window_probability'
    write_json(out/'test_metrics.json', report)
    with (out/'test_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['path', 'true_label', 'gunshot_probability', 'predicted_label'])
        w.writerows(zip(data[2], y, p, (p > threshold).astype(int)))
    print(json.dumps(report, indent=2))


def predict(a):
    net, ckpt, device = restore(a)
    y = read_clip(a.audio)
    if not np.isfinite(a.hop) or not 0 < a.hop <= 3:
        raise ValueError('--hop must be greater than 0 and at most 3 seconds.')
    hop = round(a.hop * SR)
    if hop < 1:
        raise ValueError('--hop is smaller than one sample.')
    net.eval()
    with Path(a.output).open('w', newline='', encoding='utf-8') as f, torch.inference_mode():
        w = csv.writer(f)
        w.writerow(['start_s', 'end_s', 'padded', 'gunshot_probability', 'label'])
        for start, segment in audio_windows(y, hop):
            x = features([mel(segment, ckpt['config']['frontend'])], ckpt['config']).to(device)
            p = net(x).sigmoid().item()
            w.writerow([start/SR, min(start+SAMPLES, len(y))/SR, len(y)<SAMPLES,
                        p, 'gunshot' if p > ckpt['config'].get('threshold', .5) else 'non_gunshot'])
    print(f'Saved window predictions to {a.output}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--out', default='runs/vgg16_exp6_60')
    t.add_argument('--model', choices=['vgg16', 'resnet34', 'mobilenet_v2', 'compact_cnn'], default='vgg16')
    t.add_argument('--experiment', type=int, choices=range(1, 7), default=6)
    t.add_argument('--bands', type=int, choices=[60, 83, 128], default=60)
    t.add_argument('--frontend', choices=['notebook', 'paper159'], default='notebook')
    t.add_argument('--epochs', type=int, default=85)
    t.add_argument('--seed', type=int, default=42)
    e = sub.add_parser('evaluate')
    e.add_argument('--data', required=True)
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--out', default='test_results')
    r = sub.add_parser('predict')
    r.add_argument('--checkpoint', required=True)
    r.add_argument('--audio', required=True)
    r.add_argument('--hop', type=float, default=1.)
    r.add_argument('--output', default='predictions.csv')
    for parser in [t, e, r]:
        parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or mps')
    for parser in [t, e]:
        parser.add_argument('--batch-size', type=int, default=32,
                            help='Recordings per batch; reduce for long files (all their windows are processed).')
    a = p.parse_args()
    if getattr(a, 'batch_size', 1) < 1 or getattr(a, 'epochs', 1) < 1:
        p.error('Batch size and epochs must be positive.')
    {'train': train, 'evaluate': evaluate, 'predict': predict}[a.command](a)


if __name__ == '__main__':
    main()

