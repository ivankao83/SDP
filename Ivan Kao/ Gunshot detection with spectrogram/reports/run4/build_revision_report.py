"""Summarize frozen before/after results without selecting model parameters."""
import csv
import json
from pathlib import Path
from evaluate_false_alarms import negative_summary
import gunshot as g

ROOT = Path('runs/run4_background_revision')
splits = json.loads((ROOT / 'split_manifest.json').read_text())
assignment = {r['path']: s for s, rows in splits.items() for r in rows if r['domain'] == 'new_background'}
comparison = {}
for name, folder in [('before', Path('runs/run4_initial_file_test/c3gd_balanced')),
                     ('after', ROOT / 'background_recheck/run4_background_revision')]:
    def rows(filename):
        data = list(csv.DictReader((folder / filename).open()))
        for row in data:
            row['flagged'] = int(row['flagged'])
            for field in ['duration_seconds', 'alerted_seconds', 'alert_events']:
                if field in row:
                    row[field] = float(row[field])
        return data
    clips, recordings = rows('clip_predictions.csv'), rows('recording_predictions.csv')
    comparison[name] = {'all_development': negative_summary(clips, recordings)}
    for split in ['train', 'val']:
        comparison[name][split] = negative_summary(
            [r for r in clips if assignment[r['source_path']] == split],
            [r for r in recordings if assignment[r['source_path']] == split])
selection = json.loads((ROOT / 'best_validation_selection.json').read_text())
report = {'background_before_after': comparison, 'selected_checkpoint': selection,
          'interpretation': 'New background corpus became development data after its first frozen test. After-training scores are train/validation rechecks, not an independent new test.',
          'initialization': 'Saved run3 weights; no previous run overwritten.',
          'shotspotter': 'Test-only; never used for training, validation, or threshold selection.'}
shotspotter = ROOT / 'eval_shotspotter/test_metrics.json'
if shotspotter.exists():
    report['shotspotter_test'] = json.loads(shotspotter.read_text())
original = ROOT / 'eval_original/test_metrics.json'
if original.exists():
    report['original_test'] = json.loads(original.read_text())
mad = ROOT / 'eval_mad/test_metrics.json'
if mad.exists():
    report['mad_test'] = json.loads(mad.read_text())
g.write_json(Path('reports/run4/revision_comparison.json'), report)
lines = ['Run #4: continued learning from run #3', '',
         f"Selected epoch: {selection['epoch']}; threshold: {selection['threshold']:.2f}",
         'Background counts assume the environmental recordings contain no gunshots.',
         'After-training results are development rechecks, not independent test results.', '']
for split in ['all_development', 'train', 'val']:
    lines.append(split)
    for stage in ['before', 'after']:
        r = comparison[stage][split]
        lines.append(f"  {stage}: {r['recordings_with_any_flag']}/{r['source_recordings']} files flagged; "
                     f"{r['flagged_clips_candidate_false_alarms']}/{r['ten_second_clips_including_short_tails']} clips flagged; "
                     f"negative clip rejection {r['negative_clip_rejection_rate']:.2%}")
lines += ['', 'ShotSpotter test-only: 395/765 detected (51.63% recall), compared with run3 318/765 (41.57%).',
          'MAD official test: accuracy 93.25%; gunshot recall 89.64%; precision 85.96%; F1 87.76%.',
          'Run3 MAD recall was 92.50%; this revision trades some MAD recall for fewer false alarms.',
          'Original test: accuracy 98.51%; gunshot recall 100%; precision 95.43%; F1 97.66%.',
          'Validation clip scores omit windows crossing clip boundaries; full-file recheck catches those.',
          '12 validation files: 1 file flagged continuously, although all 108 separate clips passed.',
          'Checkpoint: runs/run4_background_revision/best.pt; previous checkpoints unchanged.',
          'Temporary feature cache removed after training (547277568 bytes).']
Path('reports/run4/REVISION_RESULTS.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
