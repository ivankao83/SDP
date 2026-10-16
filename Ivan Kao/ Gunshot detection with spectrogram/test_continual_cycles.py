import json
import unittest
from pathlib import Path
from continual_cycles import metric, dataset_rows


class CycleTests(unittest.TestCase):
    def test_metrics_keep_positive_and_negative_denominators_separate(self):
        r=metric([0,0,0,1],[.1,.6,.2,.9],.5)
        self.assertEqual(r['false_positive_rate'],1/3)
        self.assertEqual(r['accuracy'],.75)
        self.assertEqual(r['gunshots_detected'],1)
        self.assertEqual(r['correct_non_gunshots'],2)

    def test_single_class_metrics_are_unavailable(self):
        r=metric([0,0],[.1,.9],.5)
        self.assertIsNone(r['accuracy']);self.assertIsNone(r['gunshot_recall'])
        self.assertEqual(r['negative_rejection_rate'],.5)
        r=metric([1,1],[.1,.9],.5)
        self.assertIsNone(r['false_positive_rate']);self.assertIsNone(r['roc_auc'])
        self.assertEqual(r['gunshot_recall'],.5)

    def test_fsdd_speakers_are_disjoint(self):
        rows=dataset_rows('FSDD')
        groups={s:{r['source_group'] for r in rows if r['split']==s} for s in ['train','val','test']}
        self.assertEqual(len(rows),3000)
        self.assertFalse(groups['train'] & (groups['val']|groups['test']))
        self.assertFalse(groups['val'] & groups['test'])

    def test_sesa_reserves_named_group_for_test(self):
        rows=dataset_rows('SESA')
        self.assertEqual([r['split'] for r in rows if r['source_group']=='gunshot_000'],['test'])
        self.assertEqual(sum(r['split']=='test' for r in rows),105)

    def test_saved_features_match_their_audio_paths(self):
        import numpy as np
        import torch
        from continual_cycles import feature
        import gunshot as g
        for folder in ['run5_sesa_valid','run6_fsdd']:
            root=Path('runs')/folder
            if not (root/'best.pt').exists(): continue
            cfg=torch.load(root/'best.pt',map_location='cpu',weights_only=True)['config']
            rows=json.loads((root/'new_data_manifest.json').read_text())
            for row in [rows[0],rows[len(rows)//2],rows[-1]]:
                np.testing.assert_array_equal(np.load(row['feature_path']),feature(g.read_clip(row['path']),cfg))

    def test_completed_cycles_never_optimize_test_rows(self):
        for path in Path('runs').glob('run*_*/training_manifest.json'):
            splits=json.loads(path.read_text())
            for s,rows in splits.items():
                self.assertTrue(all(r['split']==s for r in rows),str(path))
                self.assertFalse(any('shotspotter' in r['path'].lower() for r in rows))


if __name__=='__main__':unittest.main()
