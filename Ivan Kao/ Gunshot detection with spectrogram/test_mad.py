import unittest
import numpy as np
import torch
import gunshot as g
from train_mad import source, select_threshold


class MadRegressionTests(unittest.TestCase):
    def test_same_video_different_timestamps_is_one_group(self):
        self.assertEqual(source({'youtube url': 'https://www.youtube.com/watch?v=abc&t=1s'}),
                         source({'youtube url': 'https://www.youtube.com/watch?v=abc&t=90s'}))

    def test_threshold_changes_decisions(self):
        y, p = np.array([0, 1]), np.array([.1, .4])
        self.assertEqual(g.metrics(y, p)['gunshot']['recall'], 0.)
        self.assertEqual(g.metrics(y, p, .3)['gunshot']['recall'], 1.)
        t, report = select_threshold(y, p)
        self.assertLess(t, .4)
        self.assertEqual(report['gunshot']['f1-score'], 1.)

    def test_full_spectrum_recording_pooling_and_backward(self):
        net = g.model_for({'model': 'compact_cnn'})
        bags = [torch.rand(1, 1, 128, 161), torch.rand(3, 1, 128, 161)]
        net.eval()
        logits = g.recording_logits(net, bags, torch.device('cpu'))
        torch.testing.assert_close(logits[1], net(bags[1]).max())
        self.assertEqual(tuple(logits.shape), (2,))
        logits.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_threshold_does_not_sacrifice_weaker_domain(self):
        # A high cutoff would completely miss the quieter domain's positive.
        y = np.array([0, 1, 0, 1])
        probabilities = np.array([.1, .2, .7, .9])
        domains = np.array(['quiet', 'quiet', 'loud', 'loud'])
        threshold, _ = select_threshold(y, probabilities, domains)
        self.assertLess(threshold, .2)
        self.assertEqual(g.metrics(y[:2], probabilities[:2], threshold)['gunshot']['recall'], 1.)


if __name__ == '__main__':
    unittest.main()
