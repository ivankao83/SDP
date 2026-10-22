import unittest
import numpy as np
from train_joint_impulses import JointFeatures, assess


def row(domain,label,index,split='train',**extra):
    return {'path':f'dataset/examples/{domain}_{index}.wav','domain':domain,
            'label':label,'split':split,**extra}


class JointImpulseTests(unittest.TestCase):
    def test_loss_mass_new_two_thirds_replay_one_third(self):
        rows=[row('old',0,i) for i in range(4)]+[row('old',1,4)]
        rows += [row('new',0,i) for i in range(7)]+[row('new',1,7)]
        data=JointFeatures(rows,{'new'})
        self.assertAlmostEqual(sum(data.weights[:5]),len(rows)/3)
        self.assertAlmostEqual(sum(data.weights[5:]),len(rows)*2/3)

    def test_weak_labels_receive_less_weight(self):
        rows=[row('old',0,0),row('new',1,0),row('new',1,1,training_weight_multiplier=.25)]
        data=JointFeatures(rows,{'new'})
        self.assertAlmostEqual(data.weights[2],data.weights[1]/4)

    def test_many_negative_domains_do_not_outweigh_positive_class(self):
        rows=[row('old',0,0),row('old',1,1)]
        rows += [row('new_a',0,0),row('new_b',0,0),row('new_c',0,0),row('new_d',1,0)]
        data=JointFeatures(rows,{'new_a','new_b','new_c','new_d'},class_balance=True)
        positive=sum(w for r,w in zip(rows,data.weights) if r['label']==1)
        negative=sum(w for r,w in zip(rows,data.weights) if r['label']==0)
        self.assertAlmostEqual(positive,negative)

    def test_negative_only_new_domain_keeps_half_positive_replay_mass(self):
        rows=[row('old',0,0),row('old',1,1),row('IndependentBagPops',0,0)]
        data=JointFeatures(rows,{'IndependentBagPops'},bag_focus=True)
        self.assertAlmostEqual(data.weights[1],len(rows)/2)
        self.assertAlmostEqual(data.weights[0]+data.weights[2],len(rows)/2)

    def test_negative_only_validation_counts_new_rejections(self):
        rows=[row('old',0,0),row('new',0,1)]
        _,quality=assess(np.array([0,0]),np.array([.1,.9]),rows,{'old'})
        self.assertEqual(quality,.5)

    def test_validation_and_reserved_audio_rejected_by_optimizer(self):
        with self.assertRaises(AssertionError):JointFeatures([row('new',0,0,'test')],{'new'})
        with self.assertRaises(ValueError):JointFeatures([row('new',0,0,role='test_only')],{'new'})

    def test_selection_is_macro_averaged_across_domain_classes(self):
        rows=[row('old',0,0),row('new',0,0),row('new',1,1)]
        strata,quality=assess(np.array([0,0,1]),np.array([.1,.9,.9]),rows,{'old'})
        self.assertEqual(strata['new:0'],0.)
        self.assertEqual(quality,.75)


if __name__=='__main__':unittest.main()
