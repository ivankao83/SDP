import unittest
import numpy as np
from impulse_refiner import CNNPrior,waveform_features


class ImpulseRefinerTests(unittest.TestCase):
    def test_prior_reproduces_cnn_probability(self):
        x=np.array([[.1,4],[.9,8]],dtype=np.float32)
        p=CNNPrior().fit(x,np.array([0,1])).predict_proba(x)
        np.testing.assert_allclose(p[:,1],x[:,0])
        np.testing.assert_allclose(p.sum(axis=1),1.)

    def test_waveform_features_ignore_gain_and_peak_position(self):
        burst=np.sin(np.arange(1000)*.4)*np.exp(-np.arange(1000)/200.)
        a=np.r_[np.zeros(3000),burst,np.zeros(5000)]
        b=np.r_[np.zeros(6000),burst*.2,np.zeros(2000)]
        np.testing.assert_allclose(waveform_features(a),waveform_features(b),rtol=1e-4,atol=1e-5)

    def test_silence_produces_finite_features(self):
        x=waveform_features(np.zeros(1000))
        self.assertTrue(np.isfinite(x).all())
        self.assertGreater(len(x),50)


if __name__=='__main__':unittest.main()
