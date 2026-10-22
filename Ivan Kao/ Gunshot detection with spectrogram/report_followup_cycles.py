"""Audit the current model and summarize all follow-up trials without changing selection."""
import csv
import json
import re
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix,roc_auc_score

import gunshot as g
from continual_cycles import fingerprint,metric
from predict_impulse_refined import load_bundle,predict
from run_five_new_cycles import check_splits
from run_large_negative_cycles import vectors
from impulse_refiner import spectrogram_summary
from training_data_policy import assert_development_allowed

OUT=Path('reports/followup_cycles_20261021')
SOURCES=[('VietnamForest','mixed',Path('dataset/run21_vietnam/VietnamForest/manifest.json')),
         ('RealImpactMeasured','negative',Path('dataset/run22_realimpact/RealImpactMeasured/manifest.json')),
         ('MIMIIValve','negative',Path('dataset/run23_mimii/MIMIIValve/manifest.json')),
         ('ReaLISEDIndoor','negative',Path('dataset/run24_realised/ReaLISEDIndoor/manifest.json')),
         ('AlbuferaField','mixed; training only after pretest',Path('dataset/run28_albufera/AlbuferaField/manifest.json')),
         ('IndependentPops','negative',Path('dataset/run38_independent_pops/IndependentPops/manifest.json'))]


def pct(x):return str((Decimal(str(x))*100).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))+'%'


def ppdelta(a,b):
    return format(((Decimal(str(a))-Decimal(str(b)))*100).quantize(Decimal('.01'),rounding=ROUND_HALF_UP),'+.2f')


def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    latest=json.loads(Path('reports/latest_model.json').read_text());run=latest['run']
    report=Path('reports')/run;external=json.loads((report/'external_test/results.json').read_text())
    initial=json.loads(Path('reports/run20_vocal_sound/external_test/results.json').read_text())
    config,cfg,net,model=load_bundle(latest['bundle'])
    manifest=Path('dataset/test_only_reserve/manifest.json');reserved=json.loads(manifest.read_text())
    assert fingerprint(manifest)==initial['manifest_sha256']==external['manifest_sha256']
    assert external['refiner_sha256']==fingerprint(Path(config['refiner']))
    assert external['checkpoint_sha256']==config['cnn_sha256']
    development=json.loads((Path('runs')/run/'training_manifest.json').read_text())
    trainval=development['train']+development['val'];assert_development_allowed(trainval)
    assert not {r['decoded_sha256'] for r in reserved}&{r['decoded_sha256'] for r in trainval}
    assert not set(config['seen_hashes'])&{r['decoded_sha256'] for r in reserved}
    rows=list(csv.DictReader((report/'external_test/predictions.csv').open()))
    assert len(rows)==800
    assert {r['path']:int(r['true_label']) for r in rows}=={r['path']:r['label'] for r in reserved}
    y=np.array([int(r['true_label']) for r in rows]);p=np.array([float(r['probability']) for r in rows])
    recomputed=metric(y,p,config['threshold']);assert recomputed==external['combined']
    cm=confusion_matrix(y,p>config['threshold']).tolist();assert np.isclose(roc_auc_score(y,p),recomputed['roc_auc'])
    for name in external['by_dataset']:
        row=next(r for r in rows if r['dataset']==name)
        actual=predict(row['path'],(config,cfg,net,model))
        assert np.isclose(actual['gunshot_probability'],float(row['probability']),atol=1e-6)
    source_summaries=[];new_tests=[]
    for name,kind,path in SOURCES:
        data=json.loads(path.read_text());assert len(data)>=200
        check_splits(data);new_tests.extend(r for r in data if r['split']=='test')
        source_summaries.append({'dataset':name,'type':kind,'recordings':len(data),
            'positive':sum(r['label']==1 for r in data),'negative':sum(r['label']==0 for r in data),
            'splits':{s:sum(r['split']==s for r in data) for s in ['train','val','test']},'manifest':str(path)})
    check_splits(trainval+new_tests)
    assert not {r['decoded_sha256'] for r in new_tests}&{r['decoded_sha256'] for r in trainval}
    # Albufera was explicitly training-only; its post-training metric is seen.
    albufera=json.loads(next(path for name,kind,path in SOURCES if name=='AlbuferaField').read_text());x,ya=vectors(net,albufera,cfg)
    if config.get('spectrogram_summary'):
        x=np.column_stack([x,np.asarray([spectrogram_summary(np.load(r['feature_path'])) for r in albufera])])
    seen=metric(ya,model.predict_proba(x)[:,1],config['threshold'])
    g.write_json(OUT/'albufera_seen_retest.json',{'evaluation_is_in_sample':True,'metrics':seen})
    trials=[]
    for path in Path('reports').glob('run*/external_test/results.json'):
        match=re.match(r'run(\d+)_',path.parent.parent.name)
        if not match or int(match.group(1))<21:continue
        data=json.loads(path.read_text());rpath=path.parent.parent/'results.json'
        data_train=json.loads(rpath.read_text()) if rpath.exists() else {}
        selection=path.parent.parent/'selection.json'
        selected=json.loads(selection.read_text()) if selection.exists() else {}
        trials.append({'run':path.parent.parent.name,'number':int(match.group(1)),
                       'baseline':data['combined'],'selected_added_trees':selected.get('added_trees',selected.get('stage')),
                       'model_or_selection_revision':int(match.group(1))>=25,
                       'model_report':str(rpath)})
    trials.sort(key=lambda r:r['number'])
    before=initial['combined'];after=external['combined']
    success=after['accuracy']>before['accuracy'] and after['gunshot_recall']>=before['gunshot_recall'] and after['false_alarms']<before['false_alarms']
    verification={'same_800_paths_and_labels':True,'same_reserved_manifest_sha256':fingerprint(manifest),
        'reserved_training_validation_overlap':0,'new_heldout_training_validation_overlap':0,
        'confusion_matrix_independently_recomputed':cm,'roc_auc_independently_recomputed':True,
        'live_inference_matches_saved_predictions_for_each_baseline_dataset':True,
        'cnn_sha256_verified':config['cnn_sha256'],'refiner_sha256_verified':fingerprint(Path(config['refiner'])),
        'model_frozen_before_baseline_prediction':(report/'selection.json').stat().st_mtime_ns<=(report/'external_test/protocol.json').stat().st_mtime_ns}
    summary={'date':'2026-10-21','current_run':run,'current_bundle':latest['bundle'],
             'starting_baseline_run':'run20_vocal_sound','starting_baseline':before,'current_baseline':after,
             'accuracy_improved':after['accuracy']>before['accuracy'],'recall_preserved':after['gunshot_recall']>=before['gunshot_recall'],
             'balanced_improvement_target_met':success,'requested_accuracy_increase_met':after['accuracy']>before['accuracy'],'by_dataset':external['by_dataset'],
             'starting_by_dataset':initial['by_dataset'],'new_sources':source_summaries,
             'total_new_clips':sum(s['recordings'] for s in source_summaries),
             'new_heldout_clips':len(new_tests),'albufera_after_training_seen':seen,'trials':trials,
             'verification':verification,'benchmark_adaptively_reused':True}
    g.write_json(OUT/'results.json',summary);g.write_json(OUT/'verification.json',verification)
    lines=['# Follow-up training and baseline results — 2026-10-21','',
           f"Current model: **{run}**. Starting comparison: Run20 on exactly the same 800 reserved clips.",
           '', '| Metric | Starting Run20 | Current |','|---|---:|---:|']
    for key,title in [('recordings','Test recordings'),('accuracy','Accuracy'),('gunshot_recall','Gunshot recall'),
        ('gunshot_precision','Gunshot precision'),('gunshot_f1','Gunshot F1'),('roc_auc','ROC AUC'),
        ('threshold','Decision threshold'),('gunshots_detected','Gunshots detected'),('gunshots_missed','Gunshots missed'),
        ('false_alarms','False alarms'),('correct_non_gunshots','Correct non-gunshots')]:
        a,b=before[key],after[key]
        if key in ['accuracy','gunshot_recall','gunshot_precision','gunshot_f1']:a,b=pct(a),pct(b)
        elif key=='roc_auc':a,b=f'{a:.4f}',f'{b:.4f}'
        elif key=='threshold':a,b=f'{a:.3f}',f'{b:.3f}'
        lines.append(f'| {title} | {a} | {b} |')
    lines+=['', f"Accuracy improved by {ppdelta(after['accuracy'],before['accuracy'])} percentage points. Gunshot recall changed by {ppdelta(after['gunshot_recall'],before['gunshot_recall'])} percentage points. The stronger target of improved accuracy and fewer false alarms without a recall loss was {'met' if success else 'not met'}.", '', '## New data','', '| Dataset | Type | Clips | Gunshot / non-gunshot | Train / validation / held-out test |',
            '|---|---|---:|---:|---|']
    for s in source_summaries:
        split=s['splits'];lines.append(f"| {s['dataset']} | {s['type']} | {s['recordings']} | {s['positive']} / {s['negative']} | {split['train']} / {split['val']} / {split['test']} |")
    lines+=['',f"There are {sum(s['recordings'] for s in source_summaries):,} new original or measured microphone clips across {len(source_summaries)} corpora. "
            'No synthetic variants were generated to meet the 200-clip minimum. RealImpact channels share hammer strikes '
            'and are separated by physical object; they are not 600 independent impacts. Albufera clips all come from one '
            'continuous recording, so the entire corpus was used only for training after its initial pretest. '
            'Its after-training retest is explicitly seen and must not be presented as unseen accuracy.',
            '', f"Albufera initially: 2/55 gunshots detected, 75.66% accuracy. Current seen retest: "
            f"{seen['gunshots_detected']}/55 detected, {pct(seen['accuracy'])} accuracy. "
            f'Comparison uses the reserved baseline and {len(new_tests):,} never-trained held-out clips from the source-separated new corpora.',
            '', '## Per baseline dataset','',
            '| Dataset | Clips | Starting TP / FN / FP / TN | Current TP / FN / FP / TN |',
            '|---|---:|---|---|']
    for name,b in external['by_dataset'].items():
        a=initial['by_dataset'][name]
        def counts(r):return ' / '.join(str(r[k]) for k in ['gunshots_detected','gunshots_missed','false_alarms','correct_non_gunshots'])
        lines.append(f"| {name} | {b['recordings']} | {counts(a)} | {counts(b)} |")
    lines+=['','## Complete trial history','',
            'Runs21–24 are new-corpus cycles. Runs25 onward are pooled training, calibration or model-selection revisions '
            'using those corpora; Albufera was added in Run28 and IndependentPops in Run38. Model revisions are not additional independent datasets.',
            '', '| Run | Baseline accuracy | Gunshot recall | False alarms | Retained trees / head stage |',
            '|---|---:|---:|---:|---:|']
    for t in trials:
        m=t['baseline'];lines.append(f"| {t['run']} | {pct(m['accuracy'])} | {pct(m['gunshot_recall'])} | {m['false_alarms']} | {t['selected_added_trees']} |")
    lines+=['','## Revisions','',
            '- Corrected unstable continued boosting: tiny Hessians from extreme probabilities could make one tree flip probabilities almost completely. New Newton leaf values are bounded; regression tests verify the per-stage logit bound and unchanged earlier trees.',
            '- Retained earlier knowledge in frozen classifier priors; clipped extreme prior confidence to permit gradual adaptation.',
            '- Tested finer stage selection and validation-only threshold calibration. Pooled validation recall and macro-recall guards protect earlier positive examples; negative rejection is the selection objective.',
            '- Adaptation revisions allow explicitly recorded small validation-score tolerances and use new training-domain fit as a secondary objective. These revisions do not claim independent validation of that training-only domain.',
            '- Added optional peak-aligned spectrogram mean/max summaries alongside CNN and waveform features. Old bundles remain loadable; the prediction CLI now defaults to the current saved bundle.',
            '- Runs39 onward calibrate sensitivity and optionally blend the learned system with the frozen Run20 expert; blend weight and threshold are selected on development validation only. These are calibration revisions, not new dataset cycles.',
            '- IndependentPops uses author-labelled public Freesound previews, excludes all known FSD50K source IDs and reserved uploaders, and separates whole uploaders across train, validation and test. Labels use original author descriptions, not generated WavCaps captions.',
            '', '## Evaluation limits','',
            'None of the 800 reserved recordings entered training, validation or within-run threshold/model selection. '
            'Repeated baseline evaluations did guide stopping and strategy changes. This makes the baseline an adaptively reused '
            'development benchmark; it is not an unbiased new final test. No statistical-significance claim is made. '
            'Gunshot Triangulation contains 88 microphone clips of 22 firing events. '
            'A higher aggregate accuracy must be read alongside recall, precision and each dataset result.',
            '', '## Sources','',
            '- Vietnam: https://zenodo.org/records/13893977 — Vietnam-only subset; Belize audio excluded.',
            '- RealImpact: https://github.com/samuel-clarke/RealImpact — Clarke et al., CVPR 2023; measured, hammer-deconvolved object impacts.',
            '- MIMII: https://zenodo.org/records/3384388 — one SNR version, distinct machine IDs across splits.',
            '- ReaLISED: https://zenodo.org/records/6488321 — Mohino-Herranz et al., 2022; television excluded; classes separated; original room/session IDs unavailable.',
            '- Albufera: https://github.com/gpinyero/EUSIPCO2023 — author-labelled shooting/background clips, training-only after pretest.',
            '- IndependentPops: original Freesound URLs and licences in dataset/run38_independent_pops/ATTRIBUTIONS.md; discovery metadata from https://huggingface.co/datasets/cvssp/WavCaps. One original source ID per clip.',
            '', 'The 56-WAV prepared firearm library was excluded for failing the 200-clip minimum. '
            'Redundant archives were removed after extraction/checksum records; selected audio, manifests, models and predictions remain local. '
            'No commit or push was performed.',
            '', f"Current bundle: `{latest['bundle']}`. Prediction: `.venv/bin/python predict_impulse_refined.py /absolute/audio.wav`."]
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    latest['report']=str(OUT/'results.json');g.write_json(Path('reports/latest_model.json'),latest)
    print(json.dumps({'run':run,'before':before,'after':after,'target_met':success,'report':str(OUT/'RESULTS.md')},indent=2))


if __name__=='__main__':main()
