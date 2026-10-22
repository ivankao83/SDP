import unittest
from collections import defaultdict

from run_five_new_cycles import CycleFeatures, check_splits


def row(domain, label, split, key):
    return {'path':f'dataset/examples/{key}.wav','domain':domain,'label':label,
            'split':split,'source_group':key,'decoded_sha256':key}


class FiveCycleTests(unittest.TestCase):
    def test_half_weight_new_half_replay_balanced_by_class(self):
        rows=[row('old',0,'train',f'o{i}') for i in range(8)]
        rows += [row('old',1,'train','op')]
        rows += [row('new',0,'train',f'n{i}') for i in range(3)]
        rows += [row('new',1,'train','np')]
        data=CycleFeatures(rows,'new');totals=defaultdict(float)
        for r,w in zip(rows,data.weights):totals[(r['domain'],r['label'])]+=w
        for value in totals.values():self.assertAlmostEqual(value,len(rows)/4)

    def test_optimizer_rejects_test_split(self):
        with self.assertRaises(AssertionError):
            CycleFeatures([row('new',0,'test','one')],'new')

    def test_source_group_cannot_cross_splits(self):
        a=row('new',0,'train','one');b=row('new',0,'test','two')
        b['source_group']=a['source_group']
        with self.assertRaises(AssertionError):check_splits([a,b])

    def test_hash_cannot_cross_splits(self):
        a=row('new',0,'train','one');b=row('new',0,'val','two')
        b['decoded_sha256']=a['decoded_sha256']
        with self.assertRaises(AssertionError):check_splits([a,b])


if __name__=='__main__':unittest.main()
