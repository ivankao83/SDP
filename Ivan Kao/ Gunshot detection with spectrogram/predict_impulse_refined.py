"""Predict with a saved frozen-CNN + impulse-refiner model bundle."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch

import gunshot as g
from continual_cycles import feature,fingerprint
from impulse_refiner import waveform_features,spectrogram_summary,cnn_embedding,adapted_cnn_features,load_adapted_network


def load_bundle(path):
    config=json.loads(Path(path).read_text())
    assert config['type']=='cnn_with_impulse_refiner'
    assert fingerprint(Path(config['cnn_checkpoint']))==config['cnn_sha256']
    if 'refiner_sha256' in config:
        assert fingerprint(Path(config['refiner']))==config['refiner_sha256']
    ckpt=torch.load(config['cnn_checkpoint'],map_location='cpu',weights_only=True)
    net=g.model_for(ckpt['config']);net.load_state_dict(ckpt['state_dict']);net.eval()
    if config.get('adapted_cnn_features'):
        adapted,_=load_adapted_network(config['adapted_cnn_checkpoint'],config['adapted_cnn_sha256'])
        assert set(adapted['seen_hashes'])<=set(config['seen_hashes'])
        assert all(adapted['config'][k]==ckpt['config'][k] for k in ['model','experiment','bands','frontend'])
    return config,ckpt['config'],net,joblib.load(config['refiner'])


def predict(path,bundle):
    config,cfg,net,refiner=bundle
    audio=g.read_clip(path)
    x=torch.from_numpy(feature(audio,cfg).astype(np.float32))
    with torch.inference_mode():cnn=float(g.recording_logits(net,[x],torch.device('cpu')).sigmoid().item())
    inputs=np.r_[cnn,waveform_features(audio)][None].astype(np.float32)
    if config.get('spectrogram_summary'):inputs=np.column_stack([inputs,spectrogram_summary(x.numpy())[None]])
    if config.get('cnn_embedding'):inputs=np.column_stack([inputs,cnn_embedding(net,x.numpy())[None]])
    if config.get('adapted_cnn_features'):inputs=np.column_stack([inputs,adapted_cnn_features(config,x.numpy())[None]])
    probability=float(refiner.predict_proba(inputs)[0,1])
    return {'path':str(Path(path).resolve()),'gunshot_probability':probability,
            'threshold':config['threshold'],'gunshot_detected':probability>config['threshold']}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('audio',type=Path)
    parser.add_argument('--bundle',type=Path)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.bundle is None:
        latest=json.loads(Path('reports/latest_model.json').read_text())
        if not latest.get('bundle'):parser.error('The current model has no hybrid bundle; specify --bundle explicitly.')
        args.bundle=Path(latest['bundle'])
    print(json.dumps(predict(args.audio,load_bundle(args.bundle)),indent=2))


if __name__=='__main__':main()
