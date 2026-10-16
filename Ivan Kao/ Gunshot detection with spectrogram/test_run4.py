"""Checks for the saved run4 experiment boundaries and initialization."""
import hashlib
import json
import unittest
from pathlib import Path
import torch
from train_c3gd import binary_label


class Run4ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path('runs/run4_background_revision')
        cls.splits = json.loads((cls.root / 'split_manifest.json').read_text())

    def test_whole_background_categories_and_files_stay_in_one_split(self):
        groups = {}
        files = {}
        for split, rows in self.splits.items():
            for row in rows:
                if row['domain'] != 'new_background':
                    continue
                key = (row['dataset'], row['category'])
                self.assertIn(groups.setdefault(key, split), [split])
                self.assertIn(files.setdefault(row['path'], split), [split])
        self.assertEqual(len(files), 57)

    def test_replay_preserves_previous_domain_and_class_membership(self):
        previous = json.loads(Path('runs/c3gd_balanced/split_manifest.json').read_text())
        for split, rows in self.splits.items():
            allowed = {(r['path'], r['domain'], binary_label(r)) for r in previous[split]}
            for row in rows:
                self.assertNotIn('shotspotter', row['path'].lower())
                if row['domain'] in ['original', 'mad']:
                    self.assertIn((row['path'], row['domain'], binary_label(row)), allowed)
                if row['domain'] == 'c3gd_augmented':
                    self.assertIn(row['event_id'], ['oh_farm', 'nj_farm'] if split == 'train' else ['ny_gravel_pit'])

    def test_previous_checkpoint_unchanged_and_new_model_is_finetuned(self):
        protocol = json.loads((self.root / 'protocol.json').read_text())
        old = Path(protocol['initialization'])
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(), protocol['initialization_sha256'])
        new = torch.load(self.root / 'best.pt', map_location='cpu', weights_only=True)
        self.assertEqual(new['config']['initialization'], 'run3_checkpoint')
        self.assertGreater(new['epoch'], 0)
        self.assertFalse(protocol['shotspotter_used_for_training_or_selection'])


if __name__ == '__main__':
    unittest.main()
