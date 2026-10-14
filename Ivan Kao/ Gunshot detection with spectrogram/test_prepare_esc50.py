import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
from prepare_esc50 import prepare


class PreparationTests(unittest.TestCase):
    def test_windows_and_source_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source'
            (source/'audio').mkdir(parents=True)
            (source/'meta').mkdir()
            rows = []
            for fold in range(1, 6):
                name = f'{fold}-source{fold}-A-0.wav'
                y = np.linspace(-.1, .1, 5*44100, dtype=np.float32)
                sf.write(source/'audio'/name, y, 44100)
                rows.append(dict(filename=name, fold=fold, category='dog', src_file=f'source{fold}'))
            def write_meta():
                with (source/'meta'/'esc50.csv').open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
            write_meta()
            summary = prepare(source, root/'out')
            self.assertEqual(summary['output_windows'], {'train':9, 'val':3, 'test':3})
            for path in (root/'out').rglob('*.wav'):
                info = sf.info(path)
                self.assertEqual((info.frames, info.samplerate, info.channels), (72000, 24000, 1))
            with (root/'out'/'esc50_manifest.csv').open() as f:
                windows = list(csv.DictReader(f))
            for source_id in {x['source_id'] for x in windows}:
                self.assertEqual(len({x['split'] for x in windows if x['source_id']==source_id}), 1)
            with self.assertRaises(ValueError):
                prepare(source, root/'out')
            rows[-1]['src_file'] = rows[0]['src_file']
            write_meta()
            with self.assertRaisesRegex(ValueError, 'spans folds'):
                prepare(source, root/'leaky')
            self.assertFalse((root/'leaky').exists())


if __name__ == '__main__':
    unittest.main()
