"""Synthetic checks only: these do not measure gunshot detection accuracy."""
import gc
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import gunshot as g


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(4)
    audio = rng.normal(0, .05, g.SAMPLES).astype(np.float32)
    for frontend, shape in [('notebook', (128, 161)), ('paper159', (128, 159))]:
        s = g.mel(audio, frontend)
        assert s.shape == shape
        for experiment in range(1, 7):
            cfg = dict(experiment=experiment, bands=60, median_reference=float(np.median(s)))
            x = g.features([s, np.zeros_like(s)], cfg)
            assert x.shape == (2, 1, 60, shape[1]) and torch.isfinite(x).all()
            if g.EXPERIMENTS[experiment][1]:
                assert x.min() >= 0 and x.max() <= 1
    print('Feature dimensions, all six modes, and silence: PASS')
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)/'audio.wav'
        sf.write(path, audio, g.SR, subtype='FLOAT')
        assert g.read_clip(path).shape == (g.SAMPLES,)
        sf.write(path, audio[:g.SR], g.SR, subtype='FLOAT')
        assert g.read_clip(path).shape == (g.SR,)
        _, padded = next(g.audio_windows(g.read_clip(path)))
        assert padded.shape == (g.SAMPLES,)
    print('Audio loading and short-clip padding: PASS')
    for name in ['vgg16', 'resnet34', 'mobilenet_v2']:
        cfg = dict(model=name, experiment=6)
        net = g.model_for(cfg).eval()
        with torch.inference_mode():
            out = net(torch.zeros(2, 1, 60, 161))
        assert out.shape == (2, 1) and torch.isfinite(out).all()
        assert not any(isinstance(m, (torch.nn.ReLU, torch.nn.ReLU6)) for m in net.modules())
        print(name, 'forward pass and activation replacement: PASS')
        del net
        gc.collect()
    net = g.model_for(dict(model='mobilenet_v2', experiment=3))
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss, labels, probabilities = g.run_epoch(net, g.loader(torch.randn(2, 1, 60, 161), [0, 1], 2),
                                             torch.device('cpu'), opt)
    assert np.isfinite(loss) and np.isfinite(probabilities).all()
    report = g.metrics(np.array([0, 1]), np.array([.1, .9]))
    assert report['gunshot']['f1-score'] == 1
    print('Training backward step and metrics: PASS')


if __name__ == '__main__':
    main()

