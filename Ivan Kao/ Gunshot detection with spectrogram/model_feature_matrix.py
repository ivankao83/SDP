"""Build the same cached feature extensions used by production prediction."""
from pathlib import Path
import numpy as np
from run_large_negative_cycles import vectors
from impulse_refiner import spectrogram_summary,cnn_embedding,adapted_cnn_features


def matrix(net,rows,cfg,config):
    x,y=vectors(net,rows,cfg)
    if config.get('spectrogram_summary'):
        x=np.column_stack([x,np.asarray([spectrogram_summary(np.load(r['feature_path'])) for r in rows])])
    if config.get('cnn_embedding'):
        root=Path('dataset/cnn_embedding_cache')/config['cnn_sha256'];root.mkdir(parents=True,exist_ok=True)
        values=[]
        for r in rows:
            path=root/(r['decoded_sha256']+'.npy')
            if not path.exists():np.save(path,cnn_embedding(net,np.load(r['feature_path'])))
            values.append(np.load(path))
        x=np.column_stack([x,np.asarray(values)])
    if config.get('adapted_cnn_features'):
        root=Path('dataset/adapted_cnn_cache')/config['adapted_cnn_sha256'];root.mkdir(parents=True,exist_ok=True)
        values=[]
        for r in rows:
            path=root/(r['decoded_sha256']+'.npy')
            if not path.exists():np.save(path,adapted_cnn_features(config,np.load(r['feature_path'])))
            values.append(np.load(path))
        x=np.column_stack([x,np.asarray(values)])
    if not np.isfinite(x).all():raise ValueError('Nonfinite model input')
    return x,y
