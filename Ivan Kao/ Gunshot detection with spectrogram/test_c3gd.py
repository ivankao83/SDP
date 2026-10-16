import unittest
import numpy as np
import gunshot as g
from train_c3gd import binary_label, event_split, make_pair


class C3GDTests(unittest.TestCase):
    def test_mad_multiclass_labels_map_to_binary_before_loss_and_weighting(self):
        self.assertEqual([binary_label({'domain': 'mad', 'label': str(i)}) for i in range(7)],
                         [0, 1, 0, 0, 0, 0, 0])
        self.assertEqual(binary_label({'domain': 'c3gd_augmented', 'label': '1'}), 1)
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            binary_label({'domain': 'original', 'label': '6'})

    def test_microphones_from_same_event_stay_together(self):
        a = {'event_id': 'ny_gravel_pit', 'mic_id': 'ALICE', 'file_id': 'a', 'clip_id': '0'}
        b = {**a, 'mic_id': 'BOB', 'file_id': 'b'}
        self.assertEqual(event_split(a), 'val')
        self.assertEqual(event_split(a), event_split(b))
        self.assertEqual(event_split({'event_id': 'nj_farm_2'}), 'test')
        with self.assertRaises(KeyError):
            event_split({'event_id': 'unknown'})

    def test_short_and_long_pairs_have_matched_duration_and_finite_audio(self):
        rng = np.random.default_rng(12)
        shot = np.zeros(g.SR, dtype=np.float32)
        shot[3000:3200] = .5
        noise = rng.normal(0, .01, g.SAMPLES).astype(np.float32)
        for length in [g.SR, g.SAMPLES]:
            positive, negative = make_pair(shot, noise, rng, length, 5., 12000)
            self.assertEqual(len(positive), len(negative))
            self.assertEqual(len(positive), length)
            self.assertTrue(np.isfinite(positive).all())
            self.assertLessEqual(np.abs(positive).max(), 1.)
            self.assertGreater(np.linalg.norm(positive - negative), .01)

    def test_silent_positive_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'silent'):
            make_pair(np.zeros(g.SR), np.ones(g.SAMPLES) * .01,
                      np.random.default_rng(1), g.SAMPLES, 0., g.SR)


if __name__ == '__main__':
    unittest.main()
