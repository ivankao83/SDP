"""Run #4: TEST ONLY on fixed presumed-negative background recordings.

No optimizer, training, threshold search, or checkpoint selection. Report
negative-only clip rejection and candidate false alarms, not gunshot recall,
two-class accuracy, or AUC. Source recordings require review to confirm alarms
are false; environmental-noise labels are not exhaustive event annotations.
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


def merge_alerts(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def clip_bounds(length, chunk=10 * g.SR, minimum_tail=g.SAMPLES):
    if length < 1:
        raise ValueError('Audio must be nonempty.')
    bounds = [[start, min(start + chunk, length)] for start in range(0, length, chunk)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < minimum_tail:
        bounds[-2][1] = bounds[-1][1]
        bounds.pop()
    return bounds


def negative_summary(clips, recordings):
    flagged = sum(row['flagged'] for row in clips)
    seconds = sum(row['duration_seconds'] for row in recordings)
    alerts = sum(row['alert_events'] for row in recordings)
    return {'source_recordings': len(recordings), 'duration_seconds': seconds,
            'ten_second_clips_including_short_tails': len(clips),
            'flagged_clips_candidate_false_alarms': flagged,
            'correctly_rejected_clips_assuming_negative_labels': len(clips) - flagged,
            'candidate_false_alarm_clip_rate': flagged / len(clips) if clips else None,
            'negative_clip_rejection_rate': (len(clips) - flagged) / len(clips) if clips else None,
            'recordings_with_any_flag': sum(row['flagged'] for row in recordings),
            'candidate_alert_events': alerts,
            'candidate_alert_events_per_hour': alerts / (seconds / 3600) if seconds else None,
            'alerted_audio_fraction': sum(row['alerted_seconds'] for row in recordings) / seconds if seconds else None}


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, default=Path('dataset/false_alarm_run4'))
    p.add_argument('--out', type=Path, default=Path('runs/run4_false_alarm_test'))
    p.add_argument('--checkpoints', type=Path, nargs='+', default=[Path('runs/mixed_full_spectrum/best.pt'), Path('runs/c3gd_balanced/best.pt')])
    p.add_argument('--development-manifest', type=Path,
                   help='Explicitly evaluate reused development data, reporting train/validation separately.')
    a = p.parse_args()
    development_splits = {}
    if a.development_manifest:
        for split, rows in json.loads(a.development_manifest.read_text()).items():
            for row in rows:
                if row.get('domain') == 'new_background':
                    if row['path'] in development_splits and development_splits[row['path']] != split:
                        raise ValueError('Background source crosses development splits.')
                    development_splits[row['path']] = split
    if a.out.exists():
        raise ValueError('Choose a fresh output folder; preserve previous test results.')
    inputs = []
    for dataset in ['MS-SNSD', 'DEMAND']:
        inputs.extend(json.loads((a.data / dataset / 'manifest.json').read_text()))
    torch.set_num_threads(2)
    models = []
    for path in a.checkpoints:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        net = g.model_for(ckpt['config'])
        net.load_state_dict(ckpt['state_dict'])
        net.eval()
        threshold = ckpt['config'].get('threshold', .5)
        models.append({'path': path, 'name': path.parent.name, 'cfg': ckpt['config'],
                       'threshold': threshold, 'net': net, 'seen': set(ckpt['seen_hashes']),
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'clips': [], 'recordings': [], 'windows': [], 'alerts': []})
    a.out.mkdir(parents=True)
    protocol = {'mode': 'development_recheck' if development_splits else 'test_only', 'training_performed': False,
                'threshold_tuning_performed': False, 'checkpoint_selection_performed': False,
                'dataset_selection': 'All 51 MS-SNSD noise_test WAVs; DEMAND ch01 in six prespecified environments.',
                'input_recordings': len(inputs), 'expected_labels': 'presumed non_gunshot',
                'label_limitation': 'Environmental-noise files have not been individually gunshot-audited; alarms are candidates for review.',
                'clip_unit': 'Nonoverlapping approximately 10-second clips. Final tails shorter than 3 seconds are merged into the preceding clip; a whole source shorter than 3 seconds remains one clip.',
                'window_unit': '3-second windows at 1-second hop with full recording tail coverage.',
                'alert_event_definition': 'Merge overlapping flagged 3-second windows within each source recording.',
                'checkpoints': [{'path': str(m['path']), 'sha256_before': m['sha256'], 'threshold': m['threshold']} for m in models]}
    if development_splits:
        protocol['data_status'] = 'Reused training/validation data; these scores are not independent test performance.'
        protocol['development_manifest'] = str(a.development_manifest)
    g.write_json(a.out / 'protocol.json', protocol)
    g.write_json(a.out / 'input_manifest.json', inputs)
    exclusions, evaluated = [], []
    with torch.inference_mode():
        for i, record in enumerate(inputs):
            path = Path(record['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError(f'Input changed since download: {path}')
            y = g.read_clip(path)
            digest = hashlib.sha256(y.tobytes()).hexdigest()
            overlap = [m['name'] for m in models if digest in m['seen']]
            if overlap and str(path) not in development_splits:
                exclusions.append({'path': str(path), 'reason': 'Exact decoded training/validation overlap', 'models': overlap})
                continue  # Exclude from both models for a common comparison set.
            evaluated.append(record)
            starts = g.window_starts(len(y))
            shared = {}
            for model in models:
                cfg = model['cfg']
                key = (cfg['frontend'], cfg['experiment'], cfg['bands'], cfg.get('median_reference'))
                if key not in shared:
                    spectra = np.stack([g.mel(window, cfg['frontend']) for _, window in g.audio_windows(y)])
                    shared[key] = g.features(spectra, cfg)
                x = shared[key]
                probabilities = torch.cat([model['net'](batch).flatten() for batch in x.split(64)]).sigmoid().tolist()
                by_start = dict(zip(starts, probabilities))
                threshold = model['threshold']
                base = {'dataset': record['dataset'], 'category': record['category'], 'source_path': str(path)}
                if development_splits:
                    base['development_split'] = development_splits.get(str(path), 'unseen')
                intervals = []
                for start, prob in zip(starts, probabilities):
                    end = min(start + g.SAMPLES, len(y))
                    flag = int(prob > threshold)
                    model['windows'].append({**base, 'start_seconds': start / g.SR, 'end_seconds': end / g.SR,
                                             'gunshot_probability': prob, 'threshold': threshold, 'flagged': flag})
                    if flag:
                        intervals.append((start / g.SR, end / g.SR))
                merged = merge_alerts(intervals)
                for start, end in merged:
                    model['alerts'].append({**base, 'start_seconds': start, 'end_seconds': end,
                                            'duration_seconds': end - start})
                model['recordings'].append({**base, 'duration_seconds': len(y) / g.SR,
                                            'gunshot_probability': max(probabilities), 'threshold': threshold,
                                            'flagged': int(max(probabilities) > threshold),
                                            'alert_events': len(merged), 'alerted_seconds': sum(end - start for start, end in merged)})
                for clip_start, clip_end in clip_bounds(len(y)):
                    local_starts = g.window_starts(clip_end - clip_start)
                    clip_probabilities = []
                    for local in local_starts:
                        absolute = clip_start + local
                        if clip_end - clip_start >= g.SAMPLES or clip_start == 0:
                            clip_probabilities.append(by_start[absolute])
                        else:
                            tail = np.stack([g.mel(window, cfg['frontend']) for _, window in g.audio_windows(y[clip_start:clip_end])])
                            tail_x = g.features(tail, cfg)
                            clip_probabilities.append(model['net'](tail_x).flatten().max().sigmoid().item())
                    prob = max(clip_probabilities)
                    model['clips'].append({**base, 'start_seconds': clip_start / g.SR,
                                           'end_seconds': clip_end / g.SR, 'duration_seconds': (clip_end - clip_start) / g.SR,
                                           'gunshot_probability': prob, 'threshold': threshold, 'flagged': int(prob > threshold)})
            if (i + 1) % 5 == 0:
                print(f'Tested {i + 1}/{len(inputs)} background recordings with {len(models)} frozen models', flush=True)
    if not evaluated:
        raise ValueError('No independent presumed-negative recordings remain.')
    reports = []
    for model in models:
        after = hashlib.sha256(model['path'].read_bytes()).hexdigest()
        if after != model['sha256']:
            raise RuntimeError('Checkpoint changed during test.')
        folder = a.out / model['name']
        folder.mkdir()
        for name, key in [('clip_predictions.csv', 'clips'), ('recording_predictions.csv', 'recordings'), ('window_predictions.csv', 'windows')]:
            write_csv(folder / name, model[key])
        if model['alerts']:
            write_csv(folder / 'candidate_alert_intervals.csv', model['alerts'])
        report = {'model': model['name'], 'threshold': model['threshold'],
                  'checkpoint_sha256_before': model['sha256'], 'checkpoint_sha256_after': after,
                  'checkpoint_unchanged': True, **negative_summary(model['clips'], model['recordings'])}
        report['by_dataset'] = {d: negative_summary([r for r in model['clips'] if r['dataset'] == d],
                                                   [r for r in model['recordings'] if r['dataset'] == d])
                                for d in sorted({r['dataset'] for r in evaluated})}
        report['by_category'] = {d + '/' + c: negative_summary(
            [r for r in model['clips'] if r['dataset'] == d and r['category'] == c],
            [r for r in model['recordings'] if r['dataset'] == d and r['category'] == c])
            for d, c in sorted({(r['dataset'], r['category']) for r in evaluated})}
        if development_splits:
            report['by_development_split'] = {s: negative_summary(
                [r for r in model['clips'] if r['development_split'] == s],
                [r for r in model['recordings'] if r['development_split'] == s])
                for s in sorted(set(development_splits.values()))}
        reports.append(report)
    result = {**protocol, 'evaluated_source_recordings': len(evaluated),
              'exact_overlap_exclusions': exclusions, 'results': reports,
              'unavailable_metrics': ['gunshot recall', 'gunshot precision', 'gunshot F1', 'ROC AUC', 'two-class accuracy'],
              'unavailable_reason': 'No confirmed positive class; presumed-negative-only test.'}
    g.write_json(a.out / 'test_metrics.json', result)
    print(json.dumps([{k: r[k] for k in ['model', 'threshold', 'source_recordings', 'duration_seconds',
          'ten_second_clips_including_short_tails', 'flagged_clips_candidate_false_alarms',
          'negative_clip_rejection_rate', 'candidate_alert_events_per_hour']} for r in reports], indent=2), flush=True)


if __name__ == '__main__':
    main()
