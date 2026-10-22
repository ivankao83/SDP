"""Evaluate the frozen current checkpoint on reserved external test recordings.

No optimizer, validation, threshold search, or checkpoint selection is performed.
Features are computed in memory to avoid keeping another spectrogram cache.
"""
import csv
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import gunshot as g
from continual_cycles import feature, fingerprint, metric


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--out', type=Path, default=Path('reports/run9_external_test'))
    args = parser.parse_args()
    latest = json.loads(Path('reports/latest_model.json').read_text())
    bundle_path = args.bundle or (Path(latest['bundle']) if args.checkpoint is None and latest.get('model_type')=='cnn_with_impulse_refiner' else None)
    hybrid = None
    if bundle_path:
        if args.checkpoint:
            raise ValueError('Choose either a CNN checkpoint or a full model bundle.')
        from predict_impulse_refined import load_bundle
        hybrid = load_bundle(bundle_path)
    checkpoint = Path(hybrid[0]['cnn_checkpoint']) if hybrid else args.checkpoint or Path(latest['checkpoint'])
    checksum = fingerprint(checkpoint)
    if args.checkpoint is None and args.bundle is None:
        assert checksum == latest['sha256']
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    threshold = float(hybrid[0]['threshold'] if hybrid else ckpt['config']['threshold'])
    if args.checkpoint is None and args.bundle is None:
        assert threshold == latest['threshold']
    manifest_path = Path('dataset/test_only_reserve/manifest.json')
    manifest_checksum = fingerprint(manifest_path)
    rows = json.loads(manifest_path.read_text())
    assert len(rows) == 800
    assert all(r['role'] == 'test_only' for r in rows)
    assert len({r['decoded_sha256'] for r in rows}) == len(rows)
    assert not set(hybrid[0]['seen_hashes'] if hybrid else ckpt['seen_hashes']) & {r['decoded_sha256'] for r in rows}
    output = args.out
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        'checkpoint': str(checkpoint), 'checkpoint_sha256': checksum,
        'manifest_sha256': manifest_checksum, 'threshold': threshold,
        'recordings': len(rows), 'training_performed': False,
        'validation_performed': False, 'threshold_tuning_performed': False,
        'checkpoint_selection_performed': False,
        'prediction_rule': 'sigmoid(max window logit) > frozen threshold',
        'limitations': [
            'FSD50K sample excludes known ESC-50/UrbanSound8K IDs and uploaders; original SESA IDs unavailable.',
            'Gunshot Triangulation contains 88 microphone clips of 22 firing events.',
            'Plastic-bag microphone/event grouping is unavailable.',
        ],
    }
    if hybrid:
        protocol.update(model_type='cnn_with_impulse_refiner',bundle=str(bundle_path),
                        refiner_sha256=fingerprint(Path(hybrid[0]['refiner'])),
                        prediction_rule='learned CNN/impulse-feature classifier probability > frozen threshold')
    if hybrid and hybrid[0].get('adapted_cnn_features'):
        protocol.update(adapted_cnn_checkpoint=hybrid[0]['adapted_cnn_checkpoint'],adapted_cnn_sha256=hybrid[0]['adapted_cnn_sha256'])
    g.write_json(output / 'protocol.json', protocol)
    torch.set_num_threads(2)
    net = hybrid[2] if hybrid else g.model_for(ckpt['config'])
    if not hybrid:
        net.load_state_dict(ckpt['state_dict'])
    net.eval()
    labels, probabilities = [], []
    with (output / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'dataset', 'path', 'source_group', 'true_label', 'probability', 'predicted_label'])
        writer.writeheader()
        with torch.inference_mode():
            for i, row in enumerate(rows):
                path = Path(row['path'])
                assert fingerprint(path) == row['sha256'], f'Audio file changed: {path}'
                audio = g.read_clip(path)
                assert hashlib.sha256(audio.tobytes()).hexdigest() == row['decoded_sha256']
                x = torch.from_numpy(feature(audio, ckpt['config']).astype(np.float32))
                probability = float(g.recording_logits(net, [x], torch.device('cpu')).sigmoid().item())
                if hybrid:
                    from impulse_refiner import waveform_features
                    inputs = np.r_[probability,waveform_features(audio)][None].astype(np.float32)
                    if hybrid[0].get('spectrogram_summary'):
                        from impulse_refiner import spectrogram_summary
                        inputs=np.column_stack([inputs,spectrogram_summary(x.numpy())[None]])
                    if hybrid[0].get('cnn_embedding'):
                        from impulse_refiner import cnn_embedding
                        inputs=np.column_stack([inputs,cnn_embedding(net,x.numpy())[None]])
                    if hybrid[0].get('adapted_cnn_features'):
                        from impulse_refiner import adapted_cnn_features
                        inputs=np.column_stack([inputs,adapted_cnn_features(hybrid[0],x.numpy())[None]])
                    probability = float(hybrid[3].predict_proba(inputs)[0,1])
                labels.append(int(row['label']))
                probabilities.append(probability)
                writer.writerow({
                    'dataset': row['dataset'], 'path': row['path'],
                    'source_group': row['source_group'], 'true_label': int(row['label']),
                    'probability': probability, 'predicted_label': int(probability > threshold),
                })
                if (i + 1) % 50 == 0:
                    stream.flush()
                    print('Tested', i + 1, '/', len(rows), flush=True)
    y, p = np.array(labels), np.array(probabilities)
    by_dataset = {}
    for name in dict.fromkeys(r['dataset'] for r in rows):
        ix = np.array([r['dataset'] == name for r in rows])
        by_dataset[name] = metric(y[ix], p[ix], threshold)
    assert fingerprint(checkpoint) == checksum
    if hybrid:
        assert fingerprint(Path(hybrid[0]['refiner'])) == protocol['refiner_sha256']
    if hybrid and hybrid[0].get('adapted_cnn_features'):
        assert fingerprint(Path(hybrid[0]['adapted_cnn_checkpoint']))==protocol['adapted_cnn_sha256']
    assert fingerprint(manifest_path) == manifest_checksum
    result = {
        **protocol, 'checkpoint_unchanged': True,
        'combined': metric(y, p, threshold), 'by_dataset': by_dataset,
    }
    g.write_json(output / 'results.json', result)
    for registry_path in [Path('dataset_roles.json'), Path('reports/run9_joint_four/test_only_reserves.json')]:
        registry = json.loads(registry_path.read_text())
        registry.update(predictions_generated=True, evaluation_report=str(output / 'results.json'))
        g.write_json(registry_path, registry)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
