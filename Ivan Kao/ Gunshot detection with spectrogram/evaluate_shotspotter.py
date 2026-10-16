"""TEST ONLY: fixed checkpoints on ShotSpotter live-fire sensor recordings.

No training, optimizer, checkpoint selection, or threshold search. All recordings
are positive by dataset metadata. Recording detection rate is recall, not a
two-class accuracy estimate. Sensor recordings from the same firing test are
correlated. A positive recording score does not localize an individual shot.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import gunshot as g


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate(rows):
    detected = sum(r['predicted_gunshot'] for r in rows)
    return {'recordings': len(rows), 'detected': detected, 'missed': len(rows) - detected,
            'recording_gunshot_recall': detected / len(rows) if rows else None}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('runs/test_only_shotspotter'))
    p.add_argument('--checkpoints', type=Path, nargs='+',
                   default=[Path('runs/exp1/best.pt'), Path('runs/mixed_full_spectrum/best.pt')])
    p.add_argument('--reference-threshold', type=float,
                   help='Optional prespecified shared threshold for diagnostic comparison only; never selected on test results.')
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Choose a fresh output directory to preserve prior test results.')
    files = sorted(a.data.rglob('*.wav'))
    if not files:
        raise ValueError('No WAV files found.')
    metadata = []
    for path in files:
        audio_meta = json.loads(path.with_suffix('.json').read_text())
        survey = json.loads(next(path.parent.glob('*.survey.json')).read_text())
        info = sf.info(path)
        if int(survey['num_rounds']) < 1:
            raise ValueError(f'Expected positive live-fire recording: {path}')
        metadata.append({'path': path.relative_to(a.data).as_posix(),
                         'test_id': survey['test_id'], 'firing_position': survey['firing_position'],
                         'firearm': survey['firearm'], 'rounds_in_test': survey['num_rounds'],
                         'sensor': audio_meta['serial_number'],
                         'native_sample_rate': info.samplerate, 'duration_seconds': info.duration,
                         'raw_audio_sha256': sha(path)})
    torch.set_num_threads(2)
    models = []
    for checkpoint in a.checkpoints:
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
        net = g.model_for(ckpt['config'])
        net.load_state_dict(ckpt['state_dict'])
        net.eval()
        models.append({'checkpoint': checkpoint, 'sha256': sha(checkpoint), 'net': net,
                       'config': ckpt['config'], 'seen': set(ckpt['seen_hashes']),
                       'threshold': ckpt['config'].get('threshold', .5), 'predictions': []})
    a.out.mkdir(parents=True)
    protocol = {'mode': 'test_only', 'training_performed': False, 'threshold_tuning_performed': False,
                'checkpoint_selection_performed': False,
                'dataset': 'ShotSpotter Pittsburgh live-fire tests 2018-12-18',
                'source': 'https://github.com/ShotSpotter/research.accuracy-of-gunshot-location',
                'label_basis': 'All sensor recordings are positive according to live-fire metadata.',
                'prespecified_reference_threshold': a.reference_threshold,
                'metric_unit': 'sensor_recording', 'pooling': 'maximum probability over 3-second windows at 1-second hop',
                'checkpoints': [{'path': str(m['checkpoint']), 'sha256_before': m['sha256'],
                                 'threshold': m['threshold']} for m in models]}
    # Record the fixed evaluation protocol before running inference.
    g.write_json(a.out / 'protocol.json', protocol)
    g.write_json(a.out / 'dataset_manifest.json', metadata)
    with torch.inference_mode():
        for i, (path, meta) in enumerate(zip(files, metadata)):
            y = g.read_clip(path)
            digest = hashlib.sha256(y.tobytes()).hexdigest()
            if any(digest in m['seen'] for m in models):
                raise ValueError(f'Audio matches a checkpoint training/validation recording: {path}')
            spectra_by_frontend = {}
            for m in models:
                frontend = m['config']['frontend']
                if frontend not in spectra_by_frontend:
                    spectra_by_frontend[frontend] = np.stack([g.mel(w, frontend) for _, w in g.audio_windows(y)])
                x = g.features(spectra_by_frontend[frontend], m['config'])
                probs = torch.cat([m['net'](batch).flatten() for batch in x.split(64)]).sigmoid()
                probability = probs.max().item()
                m['predictions'].append({**meta, 'true_label': 1,
                                         'gunshot_probability': probability,
                                         'threshold': m['threshold'],
                                         'predicted_gunshot': int(probability > m['threshold']),
                                         'evaluated_windows': len(probs)})
            if (i + 1) % 100 == 0:
                print(f'Tested {i + 1}/{len(files)} recordings with {len(models)} frozen models', flush=True)
    reports = []
    for m in models:
        after = sha(m['checkpoint'])
        if after != m['sha256']:
            raise RuntimeError('Checkpoint file changed during evaluation.')
        rows = m['predictions']
        name = m['checkpoint'].parent.name
        with (a.out / f'{name}_predictions.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = {'model': name, 'checkpoint': str(m['checkpoint']),
                  'checkpoint_sha256_before': m['sha256'], 'checkpoint_sha256_after': after,
                  'checkpoint_unchanged': True, 'threshold': m['threshold'], **rate(rows)}
        if a.reference_threshold is not None:
            reference_rows = [{**r, 'predicted_gunshot': int(r['gunshot_probability'] > a.reference_threshold)} for r in rows]
            report['reference_threshold_diagnostic'] = {'threshold': a.reference_threshold, **rate(reference_rows)}
        for key in ['firearm', 'firing_position']:
            groups = defaultdict(list)
            for row in rows:
                groups[row[key]].append(row)
            report[f'by_{key}'] = {k: rate(v) for k, v in sorted(groups.items())}
        tests = defaultdict(list)
        for row in rows:
            tests[row['test_id']].append(row)
        report['firing_tests'] = len(tests)
        report['tests_with_at_least_one_sensor_flagged'] = sum(any(r['predicted_gunshot'] for r in rs) for rs in tests.values())
        report['tests_with_no_sensor_flagged'] = sum(not any(r['predicted_gunshot'] for r in rs) for rs in tests.values())
        report['accuracy'] = None
        report['precision'] = None
        report['roc_auc'] = None
        report['false_positive_rate'] = None
        report['unavailable_reason'] = 'No labeled negative recordings; report positive-recording recall only.'
        reports.append(report)
    final = {**protocol, 'decoded_exact_overlap_with_training_validation': 0,
             'recordings': len(files), 'unique_raw_audio_hashes': len({r['raw_audio_sha256'] for r in metadata}),
             'total_sensor_recording_seconds': sum(r['duration_seconds'] for r in metadata),
             'limitations': ['Multiple sensors record the same firing test, so recordings are correlated.',
                             'Maximum recording scores do not establish which individual shots were detected.',
                             'No negative class is supplied; false alarms and two-class accuracy are unmeasured.',
                             'Exact-audio hash checks cannot rule out all shared-source or near-duplicate audio.'],
             'results': reports}
    g.write_json(a.out / 'test_metrics.json', final)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == '__main__':
    main()
