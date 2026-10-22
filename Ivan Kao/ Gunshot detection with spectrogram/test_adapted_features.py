"""Regression checks for old/new network feature order, cached inputs and checkpoint integrity."""
import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import gunshot as g
from continual_cycles import fingerprint
from model_feature_matrix import matrix
from impulse_refiner import load_adapted_network,adapted_cnn_features


class AdaptedFeaturesTests(unittest.TestCase):
    def test_extensions_keep_original_inputs_and_separate_network_caches(self):
        torch.set_num_threads(2);torch.manual_seed(4444)
        cfg={'model':'compact_cnn','experiment':2};net=g.model_for(cfg).eval()
        with tempfile.TemporaryDirectory() as directory:
            original=Path.cwd()
            try:
                os.chdir(directory);root=Path(directory);spectra=np.random.default_rng(44).uniform(size=(2,1,128,161)).astype(np.float16)
                np.save(root/'feature.npy',spectra);checkpoint=root/'adapted.pt'
                torch.save({'config':cfg,'state_dict':net.state_dict(),'seen_hashes':[]},checkpoint)
                config={'cnn_sha256':'test-frozen-cnn','spectrogram_summary':True,'cnn_embedding':True}
                rows=[{'feature_path':str(root/'feature.npy'),'decoded_sha256':'test-waveform'}]
                base=np.arange(91,dtype=np.float32)[None];label=np.array([1])
                with patch('model_feature_matrix.vectors',return_value=(base,label)):
                    old,_=matrix(net,rows,cfg,config)
                    augmented={**config,'adapted_cnn_features':True,'adapted_cnn_checkpoint':str(checkpoint),'adapted_cnn_sha256':fingerprint(checkpoint)}
                    new,_=matrix(net,rows,cfg,augmented);repeated,_=matrix(net,rows,cfg,augmented)
                self.assertEqual(old.shape,(1,475));self.assertEqual(new.shape,(1,540))
                np.testing.assert_array_equal(new[:,:475],old)
                np.testing.assert_array_equal(new,repeated)
                np.testing.assert_allclose(new[:,475:],adapted_cnn_features(augmented,spectra)[None])
                with self.assertRaises(ValueError):load_adapted_network(str(checkpoint),'0'*64)
            finally:os.chdir(original);load_adapted_network.cache_clear()


if __name__=='__main__':unittest.main()
