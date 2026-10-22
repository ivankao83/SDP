import unittest
from training_data_policy import assert_development_allowed


class TrainingPolicyTests(unittest.TestCase):
    def test_reserved_directory_cannot_enter_development(self):
        with self.assertRaises(ValueError):
            assert_development_allowed([{'path':'dataset/test_only_reserve/a.wav'}])

    def test_role_blocks_audio_outside_reserved_directory(self):
        with self.assertRaises(ValueError):
            assert_development_allowed([{'path':'dataset/copied.wav','role':'test_only'}])

    def test_shotspotter_remains_test_only(self):
        with self.assertRaises(ValueError):
            assert_development_allowed([{'path':'dataset/ShotSpotter/shot.wav'}])

    def test_existing_training_audio_is_allowed(self):
        assert_development_allowed([{'path':'dataset/continual_cycles/example.wav','split':'train'}])


if __name__ == '__main__':
    unittest.main()
