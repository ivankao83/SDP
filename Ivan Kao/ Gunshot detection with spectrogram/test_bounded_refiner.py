"""Regression checks for unstable Newton steps from saturated CNN probabilities."""
import unittest
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from impulse_refiner import BoundedGradientBoostingClassifier,CNNPrior,RefinerPrior,spectrogram_summary,ProbabilityBlend,cnn_embedding


class BoundedRefinerTests(unittest.TestCase):
    def test_frozen_cnn_embedding_pools_windows_without_changing_predictions(self):
        import torch
        net=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Dropout(.5),torch.nn.Linear(8,1)).eval()
        spectra=np.arange(24,dtype=np.float32).reshape(3,1,2,4)
        with torch.inference_mode():before=net(torch.from_numpy(spectra)).numpy().copy()
        embedding=cnn_embedding(net,spectra)
        np.testing.assert_array_equal(embedding,spectra.reshape(3,8).max(axis=0))
        with torch.inference_mode():np.testing.assert_array_equal(net(torch.from_numpy(spectra)).numpy(),before)
        net.train()
        with self.assertRaises(ValueError):cnn_embedding(net,spectra)

    def test_blend_projects_legacy_feature_dimensions_and_survives_serialization(self):
        import io,joblib
        x=np.random.default_rng(40).normal(size=(40,4)).astype(np.float32);y=np.arange(40)%2
        old=GradientBoostingClassifier(n_estimators=3,random_state=40).fit(x[:,:2],y)
        new=GradientBoostingClassifier(n_estimators=3,random_state=41).fit(x,y)
        model=ProbabilityBlend(old,new,.25)
        expected=.75*old.predict_proba(x[:,:2])+.25*new.predict_proba(x)
        stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0)
        restored=joblib.load(stream)
        self.assertEqual(restored.n_features_in_,4)
        np.testing.assert_allclose(restored.predict_proba(x),expected)

    def test_spectrogram_summary_is_peak_aligned_and_finite_on_silence(self):
        pattern=np.random.default_rng(32).uniform(size=(128,12)).astype(np.float32)
        x=np.zeros((1,1,128,161),dtype=np.float32);y=x.copy()
        x[0,0,:,40:52]=pattern;y[0,0,:,80:92]=3.7*pattern
        np.testing.assert_allclose(spectrogram_summary(x),spectrogram_summary(y),atol=1e-6)
        silence=spectrogram_summary(np.zeros_like(x));self.assertEqual(len(silence),320);self.assertTrue(np.isfinite(silence).all())

    def test_prior_can_project_enriched_features_to_its_frozen_expert(self):
        x=np.random.default_rng(32).normal(size=(40,3)).astype(np.float32);y=np.arange(40)%2
        expert=GradientBoostingClassifier(n_estimators=3,random_state=32).fit(x,y)
        enriched=np.column_stack([x,np.ones((40,320))])
        wrapper=RefinerPrior(expert,feature_count=3).fit(enriched,y)
        np.testing.assert_allclose(wrapper.predict_proba(enriched),expert.predict_proba(x))
        legacy=RefinerPrior(expert).fit(x,y);del legacy.feature_count
        np.testing.assert_allclose(legacy.predict_proba(x),expert.predict_proba(x))
    def test_soft_prior_reuses_frozen_predictions_without_refitting_the_expert(self):
        x=np.column_stack([np.linspace(.001,.999,40),np.arange(40)]).astype(np.float32);y=np.arange(40)%2
        expert=GradientBoostingClassifier(n_estimators=3,random_state=29).fit(x,y)
        saved=[t.tree_.value.copy() for t in expert.estimators_[:,0]]
        wrapper=RefinerPrior(expert,probability_floor=.1).fit(x,1-y)
        np.testing.assert_allclose(wrapper.predict_proba(x)[:,1],np.clip(expert.predict_proba(x)[:,1],.1,.9))
        for values,tree in zip(saved,expert.estimators_[:,0]):np.testing.assert_array_equal(values,tree.tree_.value)
    def test_saturated_confident_errors_have_bounded_logit_updates(self):
        x=np.column_stack([np.r_[np.full(30,.000001),np.full(30,.999999)],np.zeros(60)]).astype(np.float32)
        y=np.r_[np.ones(30,dtype=int),np.zeros(30,dtype=int)]
        model=BoundedGradientBoostingClassifier(init=CNNPrior(),n_estimators=1,learning_rate=.03,max_depth=1,min_samples_leaf=2,random_state=27)
        model.max_leaf_update=3.;model.fit(x,y)
        # Use the estimator's actual float32 prior, including its clipping.
        old_logit=model._raw_predict_init(x).ravel()
        self.assertLessEqual(np.max(np.abs(model.decision_function(x)-old_logit)),.090001)
        self.assertLessEqual(np.abs(model.estimators_[0,0].tree_.value).max(),3.)

    def test_continuation_preserves_old_trees_and_bound_applies_to_next_stage(self):
        rng=np.random.default_rng(27);x=rng.normal(size=(80,3)).astype(np.float32);x[:,0]=rng.uniform(.01,.99,80);y=(x[:,1]>0).astype(int)
        model=GradientBoostingClassifier(init=CNNPrior(),n_estimators=3,learning_rate=.03,random_state=27).fit(x,y)
        old_values=[t.tree_.value.copy() for t in model.estimators_[:,0]];old_logit=model.decision_function(x).copy()
        model.__class__=BoundedGradientBoostingClassifier;model.max_leaf_update=3.
        model.set_params(warm_start=True,n_estimators=4);model.fit(x,1-y)
        for expected,tree in zip(old_values,model.estimators_[:,0]):np.testing.assert_array_equal(expected,tree.tree_.value)
        self.assertLessEqual(np.max(np.abs(model.decision_function(x)-old_logit)),.090001)


if __name__=='__main__':unittest.main()
