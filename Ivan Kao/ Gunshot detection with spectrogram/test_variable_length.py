"""Checks file labels, window coverage, variable-length batches and checkpoint use."""
import argparse
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import gunshot as g


class VariableLengthTests(unittest.TestCase):
    def test_coverage(self):
        for length in [1, 2400, 48000, 72000, 72001, 96000, 173123]:
            y = np.arange(length, dtype=np.float32)
            coverage = np.zeros(length, dtype=bool)
            for start, window in g.audio_windows(y):
                self.assertEqual(window.shape, (72000,))
                valid = min(72000, length-start)
                np.testing.assert_array_equal(window[:valid], y[start:start+valid])
                self.assertTrue(np.all(window[valid:] == 0))
                coverage[start:start+valid] = True
            self.assertTrue(coverage.all())
        with self.assertRaises(ValueError):
            g.window_starts(0)

    def test_recording_label_pooling(self):
        class MeanNet(torch.nn.Module):
            def forward(self, x):
                return x.mean((1, 2, 3)).unsqueeze(1)
        bags = [torch.tensor([-4., 3.])[:, None, None, None],
                torch.tensor([-2.])[:, None, None, None]]
        out = g.recording_logits(MeanNet(), bags, torch.device('cpu'))
        torch.testing.assert_close(out, torch.tensor([3., -2.]))
        self.assertEqual(len(out), 2)  # Two recording labels, not three window labels.

    def test_variable_length_training_and_restore(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(77)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ['train', 'val', 'test']:
                for cls, seconds in [('gunshot', 1.7), ('non_gunshot', 4.25)]:
                    folder = root/'data'/split/cls
                    folder.mkdir(parents=True)
                    sf.write(folder/'sample.wav', rng.normal(0, .03, (int(seconds*32000), 2)), 32000)
            loaded = g.load_split(root/'data', 'train', 'notebook')
            self.assertEqual([len(x) for x in loaded[0]], [3, 1])
            self.assertEqual(len(loaded[1]), 2)
            a = argparse.Namespace(data=str(root/'data'), out=str(root/'run'), seed=42,
                model='mobilenet_v2', experiment=6, bands=60, frontend='notebook', epochs=1,
                batch_size=2, device='cpu')
            g.train(a)
            a.checkpoint = str(root/'run'/'best.pt')
            a.out = str(root/'results')
            g.evaluate(a)
            with (root/'results'/'test_predictions.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 2)
            for cls in ['gunshot', 'non_gunshot']:
                a.audio = str(root/'data'/'test'/cls/'sample.wav')
                a.hop = 1.
                a.output = str(root/f'{cls}.csv')
                g.predict(a)
                with Path(a.output).open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 1 if cls == 'gunshot' else 3)


if __name__ == '__main__':
    unittest.main()
