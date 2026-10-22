import json
import unittest
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from train_joint_four import load_rows,WeightedFeatures,choose


class JointTests(unittest.TestCase):
    def test_all_four_splits_preserved(self):
        rows=load_rows()
        self.assertEqual(sum(r['split']=='train' for r in rows),2742)
        self.assertEqual(sum(r['split']=='val' for r in rows),675)
        self.assertEqual(sum(r['split']=='test' for r in rows),804)

    def test_each_domain_class_has_equal_total_weight(self):
        rows=[r for r in load_rows() if r['split']=='train']
        data=WeightedFeatures(rows)
        totals=defaultdict(float)
        for row,weight in zip(rows,data.weights):totals[(row['domain'],row['label'])]+=weight
        np.testing.assert_allclose(list(totals.values()),len(rows)/len(totals),rtol=1e-10)

    def test_checkpoint_adopts_trained_weights_and_excludes_test(self):
        root=Path('runs/run9_joint_four')
        result=json.loads((root/'results.json').read_text())
        checkpoint=torch.load(root/'best.pt',map_location='cpu',weights_only=True)
        self.assertGreater(checkpoint['epoch'],0)
        self.assertTrue(result['trained_weight_changes_adopted'])
        self.assertTrue(result['previous_checkpoint_unchanged'])
        hashes={r['decoded_sha256'] for r in load_rows() if r['split']=='test'}
        self.assertFalse(hashes&set(checkpoint['seen_hashes']))
        protocol=json.loads((root/'protocol.json').read_text())
        self.assertFalse(protocol['external_test_only_datasets_loaded'])
        self.assertFalse(protocol['shotspotter_used_for_training_or_selection'])


if __name__=='__main__':unittest.main()
