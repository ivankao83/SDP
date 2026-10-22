"""Continue the existing refiner on new >=200-clip corpora, using train/val only."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch

import gunshot as g
from continual_cycles import feature,fingerprint,infer,metric
from impulse_refiner import waveform_features
from predict_impulse_refined import load_bundle
from run_five_new_cycles import check_splits
from train_impulse_refiner import write_predictions
from training_data_policy import assert_development_allowed

ROOT=Path('dataset/run19_20_large_negatives')
THRESHOLD=.20999999999999996
PLAN=[('run19_speech_commands','SpeechCommands'),('run20_vocal_sound','VocalSound')]
SPLIT_POLICY='Speakers separated across train/validation/test; no augmented test recordings.'
POSITIVE_RETENTION_TOLERANCE=.03
NEGATIVE_RETENTION_TOLERANCE=.05
FINE_STAGES=False


def prepare(name,cfg,blocked):
    folder=ROOT/name;cache=folder/'features';cache.mkdir(exist_ok=True)
    rows=[];excluded=[]
    for raw in json.loads((folder/'raw_manifest.json').read_text()):
        try:audio=g.read_clip(raw['path'])
        except ValueError as error:
            excluded.append({**raw,'reason':str(error)});continue
        digest=hashlib.sha256(audio.tobytes()).hexdigest()
        if digest in blocked:
            excluded.append({**raw,'reason':'decoded duplicate of prior or reserved audio'});continue
        blocked.add(digest);path=cache/(digest+'.npy')
        if not path.exists():np.save(path,feature(audio,cfg))
        rows.append({**raw,'decoded_sha256':digest,'sha256':fingerprint(Path(raw['path'])),
                     'feature_path':str(path.resolve())})
    assert len(rows)>=200
    assert sum(r['split']=='test' for r in rows)>=200
    check_splits(rows)
    g.write_json(folder/'manifest.json',rows);g.write_json(folder/'duplicates_excluded.json',excluded)
    print('PREPARED',name,len(rows),dict(Counter(r['split'] for r in rows)),flush=True)
    return rows


def vectors(net,rows,cfg):
    cache=ROOT/'vectors';cache.mkdir(exist_ok=True)
    missing=[r for r in rows if not (cache/(r['decoded_sha256']+'.npy')).exists()]
    if missing:
        _,p=infer(net,missing)
        for i,(r,prob) in enumerate(zip(missing,p)):
            x=np.r_[prob,waveform_features(g.read_clip(r['path']))].astype(np.float32)
            np.save(cache/(r['decoded_sha256']+'.npy'),x)
            if (i+1)%500==0:print('Cached vectors',i+1,'/',len(missing),flush=True)
    return np.asarray([np.load(cache/(r['decoded_sha256']+'.npy')) for r in rows]),np.array([r['label'] for r in rows])


def strata(y,p,rows):
    return {f'{d}:{l}':float(((p[np.array([r['domain']==d and r['label']==l for r in rows])]>THRESHOLD)==l).mean())
            for d,l in sorted({(r['domain'],r['label']) for r in rows})}


def validation_quality(values,reference,new_domain):
    old=[v for k,v in values.items() if not k.startswith(new_domain+':')]
    new=[v for k,v in values.items() if k.startswith(new_domain+':')]
    retained=all(values[k]>=v-(POSITIVE_RETENTION_TOLERANCE if k.endswith(':1') else NEGATIVE_RETENTION_TOLERANCE) for k,v in reference.items()
                 if not k.startswith(new_domain+':'))
    return retained,.8*float(np.mean(old))+.2*float(np.mean(new))


def train_cycle(run,name,bundle_path,history,blocked):
    config,cfg,net,previous=load_bundle(bundle_path)
    checksum=fingerprint(Path(config['refiner']))
    rows=prepare(name,cfg,blocked)
    allrows=history+rows;check_splits(allrows)
    train=[r for r in allrows if r['split']=='train'];val=[r for r in allrows if r['split']=='val']
    test=[r for r in rows if r['split']=='test'];assert_development_allowed(train+val)
    reserve=json.loads(Path('dataset/test_only_reserve/manifest.json').read_text())
    assert not {r['decoded_sha256'] for r in reserve}&{r['decoded_sha256'] for r in train+val}
    out=Path('runs')/run;report=Path('reports')/run
    out.mkdir(exist_ok=False);report.mkdir(exist_ok=False)
    g.write_json(out/'training_manifest.json',{'train':train,'val':val});g.write_json(out/'new_data_manifest.json',rows)
    protocol={'previous_bundle':str(bundle_path),'previous_refiner_sha256':checksum,'new_dataset':name,
              'dataset_recordings':len(rows),'new_splits':dict(Counter(r['split'] for r in rows)),
              'training_rows_including_prior_variants':len(train),'validation_rows':len(val),
              'threshold':THRESHOLD,'cnn_updated':False,'continued_refiner_training':True,
              'maximum_added_trees':120,'selection':'Validation only; macro-domain quality with earlier-domain retention.',
              'positive_validation_retention_tolerance':POSITIVE_RETENTION_TOLERANCE,
              'negative_validation_retention_tolerance':NEGATIVE_RETENTION_TOLERANCE,
              'fine_grained_tree_selection':FINE_STAGES,
              'test_used_for_training_or_selection':False,'reserved_used_for_development':False,
              'seed':1920,'split_policy':SPLIT_POLICY}
    g.write_json(report/'protocol.json',protocol)
    xt,yt=vectors(net,train,cfg);xv,yv=vectors(net,val,cfg);xs,ys=vectors(net,test,cfg)
    before=write_predictions(test,ys,previous.predict_proba(xs)[:,1],report/'before_test')
    print('BEFORE',run,json.dumps(before),flush=True)
    reference=strata(yv,previous.predict_proba(xv)[:,1],val)
    best=validation_quality(reference,reference,name);selected=copy.deepcopy(previous);selected_added=0
    model=copy.deepcopy(previous);model.set_params(warm_start=True,random_state=1920)
    counts=Counter((r['domain'],int(r['label'])) for r in train)
    def importance(d):return 5. if d=='IndependentBagPops' else 2. if d in ('IndependentBalloonPops','LabImpulses','M16Firecrackers') else 1.
    mass={l:sum(importance(d) for d,k in counts if k==l) for l in (0,1)}
    weights=np.array([len(train)*.5*importance(r['domain'])/(mass[r['label']]*counts[(r['domain'],r['label'])]) for r in train])
    start=previous.n_estimators_;history_scores=[];best_trained=None;trained_candidate=None;trained_added=None
    if FINE_STAGES:
        model.set_params(n_estimators=start+120);model.fit(xt,yt,sample_weight=weights)
        predictions=((stage-start,p[:,1]) for stage,p in enumerate(model.staged_predict_proba(xv),1) if stage>start)
    else:
        def growing_predictions():
            for added in range(10,121,10):
                model.set_params(n_estimators=start+added);model.fit(xt,yt,sample_weight=weights)
                yield added,model.predict_proba(xv)[:,1]
        predictions=growing_predictions()
    def snapshot(added):
        saved=copy.deepcopy(model);cut=start+added
        saved.estimators_=saved.estimators_[:cut];saved.n_estimators=cut;saved.n_estimators_=cut
        saved.train_score_=saved.train_score_[:cut]
        return saved
    for added,prediction in predictions:
        values=strata(yv,prediction,val)
        quality=validation_quality(values,reference,name)
        row={'added_trees':added,'retention_met':quality[0],'quality':quality[1],'strata':values}
        history_scores.append(row)
        if best_trained is None or quality>best_trained:
            best_trained=quality;trained_candidate=snapshot(added);trained_added=added
        if quality>best:
            best=quality;selected=snapshot(added);selected_added=added
        if added%10==0 or added==1:print('VALIDATION',run,'added',added,'retained',quality[0],'quality',round(quality[1],5),flush=True)
    g.write_json(report/'validation_history.json',history_scores)
    g.write_json(report/'selection.json',{'added_trees':selected_added,'quality':best,'previous_candidate_allowed':True})
    joblib.dump(selected,out/'refiner.joblib')
    joblib.dump(trained_candidate,out/'trained_candidate.joblib')
    trained_test=write_predictions(test,ys,trained_candidate.predict_proba(xs)[:,1],report/'trained_candidate_test')
    updated={**config,'refiner':str(out/'refiner.joblib'),'refiner_sha256':fingerprint(out/'refiner.joblib'),
             'seen_hashes':sorted(set(config['seen_hashes'])|{r['decoded_sha256'] for r in train+val})}
    g.write_json(out/'bundle.json',updated)
    g.write_json(out/'trained_candidate_bundle.json',{**updated,'refiner':str(out/'trained_candidate.joblib'),
                 'refiner_sha256':fingerprint(out/'trained_candidate.joblib')})
    after=write_predictions(test,ys,selected.predict_proba(xs)[:,1],report/'after_test')
    result={**protocol,'before':before,'after':after,'selected_added_trees':selected_added,
            'training_updates_adopted':selected_added>0,'bundle':str(out/'bundle.json'),
            'trained_candidate_added_trees':trained_added,'trained_candidate_test':trained_test}
    g.write_json(report/'results.json',result)
    assert fingerprint(Path(config['refiner']))==checksum
    print('AFTER',run,json.dumps(after),flush=True)
    return out/'bundle.json',allrows,result


def main():
    import fcntl
    lock=Path('runs/continual_cycles/training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    saved=json.loads(Path('runs/run18_impulse_refiner/training_manifest.json').read_text())
    history=saved['train']+saved['val'];bundle=Path('runs/run18_impulse_refiner/bundle.json')
    config=json.loads(bundle.read_text());reserve=json.loads(Path('dataset/test_only_reserve/manifest.json').read_text())
    blocked=set(config['seen_hashes'])|{r['decoded_sha256'] for r in reserve}
    # Exclude prior development AND held-out material from all previous runs.
    for path in Path('runs').glob('*/new_data_manifest.json'):
        blocked|={r['decoded_sha256'] for r in json.loads(path.read_text())}
    results=[]
    for run,name in PLAN:
        completed=Path('reports')/run/'results.json'
        if completed.exists():
            result=json.loads(completed.read_text());bundle=Path(result['bundle'])
            current=json.loads((Path('runs')/run/'new_data_manifest.json').read_text())
            history+=current;blocked|={r['decoded_sha256'] for r in current};results.append(result)
            print('RESUMED',run,flush=True);continue
        bundle,history,result=train_cycle(run,name,bundle,history,blocked);results.append(result)
        g.write_json(ROOT/'cycles.json',results)
    print('TWO TRAINING CYCLES COMPLETE; MODELS FROZEN FOR EXTERNAL TEST',flush=True)


if __name__=='__main__':main()
