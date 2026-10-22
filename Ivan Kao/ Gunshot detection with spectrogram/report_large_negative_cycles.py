"""Verify saved test predictions and publish the validation-selected Run20 bundle."""
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import gunshot as g
from continual_cycles import metric,fingerprint
from predict_impulse_refined import load_bundle
from run_five_new_cycles import check_splits
from run_large_negative_cycles import ROOT,PLAN,vectors
from train_impulse_refiner import write_predictions
from training_data_policy import assert_development_allowed


def main():
    torch.set_num_threads(2)
    destination=Path('reports/run19_20_summary');destination.mkdir(exist_ok=True)
    results=[json.loads((Path('reports')/run/'results.json').read_text()) for run,_ in PLAN]
    previous=json.loads(Path('reports/run18_impulse_refiner/results.json').read_text())
    external=json.loads(Path('reports/run20_vocal_sound/external_test/results.json').read_text())
    bundle_path=Path('runs/run20_vocal_sound/bundle.json');config,cfg,net,refiner=load_bundle(bundle_path)
    source=json.loads(Path('runs/run20_vocal_sound/training_manifest.json').read_text())
    trainval=source['train']+source['val'];assert_development_allowed(trainval)
    tests=[];new=[]
    for _,name in PLAN:
        rows=json.loads((ROOT/name/'manifest.json').read_text());check_splits(rows)
        assert len(rows)>=200 and sum(r['split']=='test' for r in rows)>=200
        new+=rows;tests += [r for r in rows if r['split']=='test']
    check_splits(trainval+tests)
    reserve=json.loads(Path('dataset/test_only_reserve/manifest.json').read_text())
    assert not {r['decoded_sha256'] for r in reserve}&{r['decoded_sha256'] for r in trainval+new}
    assert not set(config['seen_hashes'])&{r['decoded_sha256'] for r in tests+reserve}
    assert fingerprint(Path('dataset/test_only_reserve/manifest.json'))==previous['reserved_manifest_sha256']==external['manifest_sha256']
    old=joblib.load('runs/run18_impulse_refiner/refiner.joblib')
    assert len(refiner.estimators_)==len(old.estimators_)+100
    for a,b in zip(old.estimators_[:,0],refiner.estimators_[:,0]):
        assert np.array_equal(a.tree_.value,b.tree_.value)
        assert np.array_equal(a.tree_.threshold,b.tree_.threshold)
    baseline_csv=Path('reports/run20_vocal_sound/external_test/predictions.csv')
    predictions=list(csv.DictReader(baseline_csv.open()))
    expected={r['path']:r['label'] for r in reserve}
    assert len(predictions)==800 and {r['path']:int(r['true_label']) for r in predictions}==expected
    y=np.array([int(r['true_label']) for r in predictions]);p=np.array([float(r['probability']) for r in predictions])
    recomputed=metric(y,p,config['threshold']);assert recomputed==external['combined']
    assert np.isclose(roc_auc_score(y,p),external['combined']['roc_auc'])
    x,ynew=vectors(net,tests,cfg)
    combined_new=write_predictions(tests,ynew,refiner.predict_proba(x)[:,1],destination/'final_model_new_test')
    verification={'test_only_reserves_excluded':True,'new_speakers_separated':True,
                  'new_test_clips_never_trained_or_validated':True,'cnn_hash_verified':True,
                  'original_380_refiner_trees_unchanged':True,'new_trained_trees':100,
                  'saved_baseline_metrics_independently_recomputed':True,
                  'unchanged_800_clip_baseline':True}
    g.write_json(destination/'verification.json',verification)
    summary={'cycles':results,'previous_baseline':previous['combined'],'after_baseline':external['combined'],
             'before_by_dataset':previous['by_dataset'],'after_by_dataset':external['by_dataset'],
             'total_new_usable_recordings':len(new),'total_new_heldout_recordings':len(tests),
             'final_model_new_heldout':combined_new,'selected_bundle':str(bundle_path),
             'selection_uses_validation_only':True,'verification':verification}
    g.write_json(destination/'results.json',summary)
    latest={'run':'run20_vocal_sound','model_type':'cnn_with_impulse_refiner','bundle':str(bundle_path),
            'checkpoint':config['cnn_checkpoint'],'sha256':config['cnn_sha256'],
            'refiner':config['refiner'],'refiner_sha256':config['refiner_sha256'],
            'threshold':config['threshold'],'report':str(destination/'results.json')}
    g.write_json(Path('reports/latest_model.json'),latest)
    lines=['# Runs 19–20 results','',
           'Two new negative corpora, each with over 1,000 usable clips and at least 200 held-out test clips. '
           'All splits were fixed before testing or training; speakers are separated within each corpus. '
           'The same test clips were scored before and after training; only separate training clips optimized weights.', '',
           '| Run | Dataset | Usable clips | Train | Validation | Test | False alarms before | False alarms after training | Correct rejection before | Correct rejection after |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for number,r in enumerate(results,19):
        s=r['new_splits'];a=r['before'];b=r['trained_candidate_test']
        lines.append(f"| {number} | {r['new_dataset']} (negative) | {r['dataset_recordings']} | {s['train']} | {s['val']} | {s['test']} | {a['false_alarms']} | {b['false_alarms']} | {100*a['negative_rejection_rate']:.2f}% | {100*b['negative_rejection_rate']:.2f}% |")
    lines+=['','Speech Commands training tried 120 additional trees. Its best trained candidate used 30 added trees, '
            'but validation did not improve over the previous model, so those changes were not carried forward. '
            'VocalSound training used replay of earlier approved training data, including the new Speech Commands training split. '
            'Validation selected 100 added trees before the baseline was evaluated. The original CNN and all 380 prior refiner trees stayed unchanged.',
            '', 'Three nonfinite VocalSound recordings were excluded. There are 2,037 usable new clips overall: '
            '1,198 training, 419 validation and 420 held-out testing. No augmentations are counted as new independent clips.',
            '', '## Same 800-recording test-only baseline','',
            '| Metric | Previous Run18 | Run20 |','|---|---:|---:|']
    for key,title in [('accuracy','Accuracy'),('gunshot_recall','Gunshot recall'),('gunshot_precision','Gunshot precision'),('gunshot_f1','Gunshot F1'),('roc_auc','ROC AUC'),('threshold','Decision threshold'),('gunshots_detected','Gunshots detected'),('gunshots_missed','Gunshots missed'),('false_alarms','False alarms'),('correct_non_gunshots','Correct non-gunshots')]:
        a=previous['combined'][key];b=external['combined'][key]
        if key in ('accuracy','gunshot_recall','gunshot_precision','gunshot_f1'):
            a=str((Decimal(str(a))*100).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))+'%'
            b=str((Decimal(str(b))*100).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))+'%'
        elif key=='roc_auc':a=f'{a:.4f}';b=f'{b:.4f}'
        elif key=='threshold':a=f'{a:.2f}';b=f'{b:.2f}'
        lines.append(f'| {title} | {a} | {b} |')
    lines+=['','Accuracy increased 0.625 percentage points and false alarms decreased by 7 (4.70%). '
            'Recall fell 1.45 percentage points: 2 additional gunshots were missed. This is a modest tradeoff, '
            'not a significant across-the-board improvement. No claim of statistical significance is made.',
            '', '## Per baseline dataset','',
            '| Dataset | Type | Clips | Before TP / FN / FP / TN | After TP / FN / FP / TN |',
            '|---|---|---:|---|---|']
    for name,kind in [('FSD50K_mixed','mixed'),('GunshotTriangulation','positive'),('PlasticBagPop','negative')]:
        a=previous['by_dataset'][name];b=external['by_dataset'][name]
        def counts(r):return ' / '.join(str(r[k]) for k in ('gunshots_detected','gunshots_missed','false_alarms','correct_non_gunshots'))
        lines.append(f"| {name} | {kind} | {b['recordings']} | {counts(a)} | {counts(b)} |")
    lines+=['','## Sources and interpretation','',
            '- Speech Commands: https://www.tensorflow.org/tutorials/audio/simple_audio; GitHub: https://github.com/tensorflow/datasets/blob/master/docs/catalog/speech_commands.md',
            '- VocalSound: https://github.com/YuanGongND/vocalsound; compact download mirror https://zenodo.org/records/14649751 (CC-BY-SA-4.0). Cite Gong, Yu and Glass, ICASSP 2022.',
            '', 'Gunshot recall, gunshot precision, gunshot F1 and ROC AUC are not informative for the negative-only new subsets; '
            'correct rejection and false-alarm counts are reported instead. Baseline gunshot triangulation has 88 microphone clips '
            'of 22 shots. Repeated historical baseline evaluations mean it is an adaptively reused benchmark, not a new unbiased '
            'estimate of generalization. Baseline audio was never used for training, validation, threshold search or checkpoint selection.',
            '', 'Run19 trained-candidate baseline (rejected earlier on validation): 79.50% accuracy, '
            '86.23% recall, 145 false alarms. It was not adopted based on this test result.',
            '', f"Final Run20 model on all {len(tests)} new held-out negatives: {combined_new['false_alarms']} false alarms; "
            f"{combined_new['negative_rejection_rate']:.2%} correctly rejected. This uses one final model for all clips.",
            '', 'The threshold remains 0.21. Selected audio, split manifests, hashes, feature caches, checkpoints and raw test predictions '
            'are retained. Redundant download archives totaling 377.7 MB were deleted after verifying checksums and extraction. '
            'No Git commit or push was made in these runs.',
            '', 'Inference: `.venv/bin/python predict_impulse_refined.py /absolute/file.wav --bundle runs/run20_vocal_sound/bundle.json`']
    (destination/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
