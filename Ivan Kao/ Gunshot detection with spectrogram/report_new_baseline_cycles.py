"""Verify the final saved model and report the new-source cycles against fixed Run41."""
import csv,json,re
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import gunshot as g
from continual_cycles import fingerprint,metric
from predict_impulse_refined import load_bundle,predict
from model_feature_matrix import matrix
from run_five_new_cycles import check_splits
from training_data_policy import assert_development_allowed
OUT=Path('reports/followup_cycles_20261022')


def percent(v):return 'N/A' if v is None else str((Decimal(str(v))*100).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))+'%'


def read_predictions(path):return list(csv.DictReader(path.open()))


def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    latest=json.loads(Path('reports/latest_model.json').read_text());run=latest['run'];report=Path('reports')/run
    assert int(re.match(r'run(\d+)',run).group(1))>=42
    initial=json.loads(Path('reports/run41_balanced_validation/external_test/results.json').read_text());final=json.loads((report/'external_test/results.json').read_text())
    loaded=load_bundle(latest['bundle']);config,cfg,net,model=loaded
    reserved=json.loads(Path('dataset/test_only_reserve/manifest.json').read_text());reserved_sha=fingerprint(Path('dataset/test_only_reserve/manifest.json'))
    assert initial['manifest_sha256']==final['manifest_sha256']==reserved_sha
    development=json.loads((Path('runs')/run/'training_manifest.json').read_text());trainval=development['train']+development['val'];assert_development_allowed(trainval)
    assert not {r['decoded_sha256'] for r in reserved}&{r['decoded_sha256'] for r in trainval}
    assert not set(config['seen_hashes'])&{r['decoded_sha256'] for r in reserved}
    old=read_predictions(Path('reports/run41_balanced_validation/external_test/predictions.csv'));new=read_predictions(report/'external_test/predictions.csv')
    oldmap={r['path']:r for r in old};newmap={r['path']:r for r in new};assert set(oldmap)==set(newmap)=={r['path'] for r in reserved}
    for r in reserved:assert int(oldmap[r['path']]['true_label'])==int(newmap[r['path']]['true_label'])==r['label']
    y=np.array([int(r['true_label']) for r in new]);p=np.array([float(r['probability']) for r in new]);assert metric(y,p,config['threshold'])==final['combined']
    for name in final['by_dataset']:
        r=next(r for r in new if r['dataset']==name)
        assert np.isclose(predict(r['path'],loaded)['gunshot_probability'],float(r['probability']),atol=1e-6)
    cycles=[(Path('dataset/run42_new_sources/manifest.json'),'run42_new_sources','run41_balanced_validation')]
    if int(re.match(r'run(\d+)',run).group(1))>=46:
        cycles.append((Path('dataset/run46_claprir/ClapRIRPhone/manifest.json'),'run46_room_handclaps','run45_adapted_system'))
    sources=[r for path,cycle,prior_run in cycles for r in json.loads(path.read_text())];assert len(sources)>=400
    check_splits(trainval+[r for r in sources if r['split']=='test'])
    assert not {r['decoded_sha256'] for r in sources if r['split']=='test'}&{r['decoded_sha256'] for r in trainval}
    x,ys=matrix(net,sources,cfg,config);prob=model.predict_proba(x)[:,1]
    before_map={};before_threshold={};before_run={}
    for path,cycle,prior_run in cycles:
        prior_config=json.loads((Path('runs')/prior_run/'bundle.json').read_text())
        rows=read_predictions(Path('reports')/cycle/'new_source_before_training.csv')
        for row in rows:
            before_map[row['path']]=row;before_threshold[row['path']]=prior_config['threshold'];before_run[row['path']]=prior_run
    assert set(before_map)=={r['path'] for r in sources}
    for r in sources:
        prior=before_map[r['path']]
        assert int(prior['true_label'])==r['label']
        assert int(prior['predicted_label'])==int(float(prior['probability'])>before_threshold[r['path']])
    def pooled_before(indices):
        y=ys[indices];pred=np.array([int(before_map[sources[i]['path']]['predicted_label']) for i in indices])
        thresholds={before_threshold[sources[i]['path']] for i in indices}
        if len(thresholds)==1:return metric(y,np.array([float(before_map[sources[i]['path']]['probability']) for i in indices]),next(iter(thresholds)))
        result=metric(y,np.where(pred,1.,0.),.5);result['roc_auc']=None;result['threshold']=None
        result['different_frozen_pretest_models']=True;return result
    source_results={}
    for name in sorted({r['domain'] for r in sources}):
        indices=[i for i,r in enumerate(sources) if r['domain']==name];held=[i for i in indices if sources[i]['split']=='test']
        prior=np.array([float(before_map[sources[i]['path']]['probability']) for i in indices]);target=ys[indices]
        source_results[name]={'recordings':len(indices),'positive':int(sum(target)),'negative':int(sum(target==0)),
            'splits':dict(Counter(sources[i]['split'] for i in indices)),
            'before_training_all':pooled_before(indices),'before_run':before_run[sources[indices[0]]['path']],'after_training_all_contains_seen':metric(target,prob[indices],config['threshold']),
            'heldout_before':pooled_before(held),
            'heldout_after':metric(ys[held],prob[held],config['threshold'])}
    all_before=pooled_before(list(range(len(sources))))
    all_after=metric(ys,prob,config['threshold']);held=np.array([r['split']=='test' for r in sources])
    held_before=pooled_before(np.flatnonzero(held).tolist())
    held_after=metric(ys[held],prob[held],config['threshold'])
    changes=Counter()
    for path,r in newmap.items():
        truth=int(r['true_label']);a=int(oldmap[path]['predicted_label'])==truth;b=int(r['predicted_label'])==truth
        changes['both_correct' if a and b else 'both_wrong' if not a and not b else 'improved' if b else 'regressed']+=1
    a,b=initial['combined'],final['combined'];objective=json.loads((OUT/'objective_before_trials.json').read_text())
    passed=b['accuracy']>=objective['minimum_accuracy']-1e-12 and b['gunshot_recall']>=objective['minimum_gunshot_recall']-1e-12 and b['false_alarms']<objective['maximum_false_alarms_exclusive']
    trials=[]
    for path in Path('reports').glob('run*/external_test/results.json'):
        match=re.match(r'run(\d+)',path.parent.parent.name)
        if match and int(match.group(1))>=42:
            value=json.loads(path.read_text());trials.append({'run':path.parent.parent.name,'number':int(match.group(1)),'baseline':value['combined']})
    trials.sort(key=lambda r:r['number'])
    verification={'same_800_paths_labels_and_manifest_sha256':reserved_sha,'reserved_training_validation_overlap':0,
        'new_heldout_training_validation_overlap':0,'metrics_independently_recomputed':True,
        'production_prediction_matches_saved_result_for_each_baseline_dataset':True,
        'cnn_sha256':config['cnn_sha256'],'refiner_sha256':fingerprint(Path(config['refiner'])),
        'adapted_cnn_sha256':config.get('adapted_cnn_sha256'),
        'adapted_checkpoint_hash_verified':not config.get('adapted_cnn_features') or fingerprint(Path(config['adapted_cnn_checkpoint']))==config['adapted_cnn_sha256'],
        'selected_before_baseline_prediction':(report/'selection.json').stat().st_mtime_ns<=(report/'external_test/protocol.json').stat().st_mtime_ns}
    result={'current_run':run,'current_bundle':latest['bundle'],'starting_baseline':a,'latest_baseline':b,'by_dataset_before':initial['by_dataset'],
        'by_dataset_after':final['by_dataset'],'target':objective,'practical_target_met':passed,'new_sources':source_results,
        'new_all_before':all_before,'new_all_after_contains_seen':all_after,'new_heldout_before':held_before,'new_heldout_after':held_after,
        'paired_changes':dict(changes),'trials':trials,'verification':verification,'benchmark_adaptively_reused':True}
    g.write_json(OUT/'results.json',result);g.write_json(OUT/'verification.json',verification)
    lines=['# New-source training and baseline comparison — 2026-10-22','',f'Current model: **{run}**. Comparison: Run41, same 800 reserved recordings.',
        '',f"Practical target (+5 percentage points in accuracy, no recall loss, fewer false alarms): **{'met' if passed else 'not met'}**. This is not a statistical-significance claim.",
        '', '| Metric | Starting Run41 | Current |','|---|---:|---:|']
    labels=[('recordings','Test recordings'),('accuracy','Accuracy'),('gunshot_recall','Gunshot recall'),('gunshot_precision','Gunshot precision'),('gunshot_f1','Gunshot F1'),('roc_auc','ROC AUC'),('threshold','Decision threshold'),('gunshots_detected','Gunshots detected'),('gunshots_missed','Gunshots missed'),('false_alarms','False alarms'),('correct_non_gunshots','Correct non-gunshots')]
    for key,title in labels:
        left,right=a[key],b[key]
        if key in ['accuracy','gunshot_recall','gunshot_precision','gunshot_f1']:left,right=percent(left),percent(right)
        elif key in ['roc_auc','threshold']:left,right=f'{left:.4f}',f'{right:.4f}'
        lines.append(f'| {title} | {left} | {right} |')
    lines+=['','## Per baseline dataset','','| Dataset | Clips | Before TP / FN / FP / TN | Current TP / FN / FP / TN |','|---|---:|---|---|']
    for name,right in final['by_dataset'].items():
        left=initial['by_dataset'][name]
        def counts(z):return ' / '.join(str(z[k]) for k in ['gunshots_detected','gunshots_missed','false_alarms','correct_non_gunshots'])
        lines.append(f"| {name} | {right['recordings']} | {counts(left)} | {counts(right)} |")
    lines+=['','## New sources and held-out tests','','| Source | Type | Unique clips | Train / validation / held-out | Held-out gunshots detected before → after | Held-out false alarms before → after |','|---|---|---:|---|---|---|']
    for name,r in source_results.items():
        s=r['splits'];before,after=r['heldout_before'],r['heldout_after'];kind='positive' if not r['negative'] else 'negative' if not r['positive'] else 'mixed'
        lines.append(f"| {name} | {kind} | {r['recordings']} | {s['train']} / {s['val']} / {s['test']} | {before['gunshots_detected']} → {after['gunshots_detected']} | {before['false_alarms']} → {after['false_alarms']} |")
    lines+=['',f"The {len(sources)}-clip pool before each source's own training cycle: {percent(all_before['accuracy'])} accuracy. Its current retest: {percent(all_after['accuracy'])}, **including recordings used for training and validation**. This is a seen-pool result.",
        '',f"Pretests use the frozen model available before each source cycle. Only the {int(held.sum())} held-out new-source clips provide the separate never-trained comparison: {percent(held_before['accuracy'])} → {percent(held_after['accuracy'])} accuracy.",
        '', 'IndependentFieldMixed is a newly curated 306-clip collection, not a separately published benchmark. Original Freesound author descriptions establish real firearm discharge or non-firearm material/action labels. Only CC0 or attribution-only previews were used; generated captions were not used as labels. Whole uploaders stay in one split; complete positive recordings were preserved to avoid cropping away quiet shots.',
        '', 'AydemirGunshots selects 40 filenames per source folder from the published corpus. The 360 selected files contained 40 exact decoded duplicates, leaving 320 unique clips. Entire firearm-model groups are separated; M4/M16 share one training family. Original video IDs are unavailable, so model grouping cannot establish all video or cross-corpus independence. The source licence field is Unknown; local audio is excluded from Git.',
        '', 'ClapRIRPhone uses 280 author-recorded phone handclaps, with paired inferred RIR outputs excluded. Whole rooms define 160 training, 40 validation and 80 test recordings. Seven rooms are available; performer identity is not provided. Git blob and author export hashes were checked before resampling. The author does not assert a separate asset dataset licence; audio remains local.',
        '', '## Training and model revisions','', '| Run | Baseline accuracy | Recall | False alarms |','|---|---:|---:|---:|']
    for t in trials:
        m=t['baseline'];lines.append(f"| {t['run']} | {percent(m['accuracy'])} | {percent(m['gunshot_recall'])} | {m['false_alarms']} |")
    lines+=['', 'Run42 is the new-source training cycle with earlier-data replay and frozen CNN pattern embeddings. Run43 tests alternative learned correction heads on those approved development splits. Run44 fine-tunes a copy of the prior CNN using balanced replay and the frozen full system as a teacher; its epoch is selected on validation only. The combined adapted system is evaluated in Run45. Run46 adds the new room-separated ClapRIRPhone collection, pretesting it with Run45 before training; later model revisions may use these same approved development splits. Runs47–49 are validation-only sensitivity calibration revisions with no new weight training. These model revisions are not additional datasets.',
        '', '## Verification and limits','',f"Paired clip changes: {changes['improved']} previously wrong predictions corrected; {changes['regressed']} previously correct predictions became wrong.",
        '', 'All 800 baseline recordings are excluded from training and within-run validation, threshold, epoch and correction-head selection. Repeated baseline results guide continuation and strategy changes, making this an adaptively reused development benchmark rather than an unbiased final test. Correlated microphone/event recordings also limit inferential claims. No statistical-significance claim is made.',
        '', 'Production inference was checked against saved predictions for each baseline dataset. The 800 paths, labels and manifest SHA256 are unchanged; confusion counts and ROC AUC were recomputed independently.',
        '', '## Sources','', '- IndependentFieldMixed original author links and licences: dataset/run42_independent_firearms/ATTRIBUTIONS.md.',
        '- Discovery index: https://huggingface.co/datasets/cvssp/WavCaps (original author descriptions, not generated captions).',
        '- ClapRIRPhone source: https://github.com/Akinesia112/ClapRIR; original phone recordings only, not inferred RIRs.',
        '- Published gunshot corpus: https://www.kaggle.com/datasets/emrahaydemr/gunshot-audio-dataset.',
        '- Citation: Tuncer, Dogan, Akbal and Aydemir (2021), An Automated Gunshot Audio Classification Method Based On Finger Pattern Feature Generator And Iterative Relieff Feature Selector.',
        '', 'Original audio and model files remain local. No Git commit or push was performed.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n');latest['report']=str(OUT/'results.json');g.write_json(Path('reports/latest_model.json'),latest)
    print(json.dumps({'run':run,'target_met':passed,'before':a,'after':b,'report':str(OUT/'RESULTS.md')},indent=2))


if __name__=='__main__':main()
