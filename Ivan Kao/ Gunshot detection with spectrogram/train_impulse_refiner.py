"""Train an impulse-feature classifier without changing the retained CNN."""
import copy
import csv
import fcntl
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier

import gunshot as g
from continual_cycles import fingerprint,infer,metric,feature
from impulse_refiner import CNNPrior,waveform_features
from run_five_new_cycles import check_splits
from training_data_policy import assert_development_allowed

BASE=Path('runs/run14_m16_firecrackers/best.pt')
OUT=Path('runs/run18_impulse_refiner')
REPORT=Path('reports/run18_impulse_refiner')
THRESHOLD=.20999999999999996


def matrix(net,rows,folder):
    folder.mkdir(parents=True,exist_ok=True)
    y,p=infer(net,rows);vectors=[]
    for i,(row,probability) in enumerate(zip(rows,p)):
        path=folder/(row['decoded_sha256']+'.npy')
        if not path.exists():np.save(path,waveform_features(g.read_clip(row['path'])))
        vectors.append(np.r_[probability,np.load(path)])
        if (i+1)%500==0:print('Waveform features',i+1,'/',len(rows),flush=True)
    return np.asarray(vectors,dtype=np.float32),y


def score(y,p,rows,reference):
    pred=p>THRESHOLD;strata={}
    for domain,label in sorted({(r['domain'],int(r['label'])) for r in rows}):
        ix=np.array([r['domain']==domain and int(r['label'])==label for r in rows])
        strata[f'{domain}:{label}']=float((pred[ix]==y[ix]).mean())
    new={'IndependentBagPops','IndependentBalloonPops','LabImpulses','CMURealSounds'}
    oldpos=[v for k,v in strata.items() if k.endswith(':1') and k.rsplit(':',1)[0] not in new]
    oldneg=[v for k,v in strata.items() if k.endswith(':0') and k.rsplit(':',1)[0] not in new]
    newneg=[v for k,v in strata.items() if k.rsplit(':',1)[0] in new]
    quality=.4*float(np.mean(oldpos))+.2*float(np.mean(oldneg))+.4*float(np.mean(newneg))
    retained=True if reference is None else all(strata[k]>=value-.05 for k,value in reference.items() if k.endswith(':1'))
    return retained,quality,strata


def write_predictions(rows,y,p,destination):
    report=metric(y,p,THRESHOLD)
    g.write_json(destination.with_suffix('.json'),report)
    with destination.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','path','true_label','probability','predicted_label']);writer.writeheader()
        writer.writerows({'dataset':r.get('domain',r.get('dataset')),'path':r['path'],'true_label':int(label),
                         'probability':float(prob),'predicted_label':int(prob>THRESHOLD)} for r,label,prob in zip(rows,y,p))
    return report


def main():
    lock=Path('runs/continual_cycles/training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    torch.set_num_threads(2);checksum=fingerprint(BASE)
    ckpt=torch.load(BASE,map_location='cpu',weights_only=True)
    net=g.model_for(ckpt['config']);net.load_state_dict(ckpt['state_dict'])
    saved=json.loads(Path('runs/run17_bag_confounds/training_manifest.json').read_text())
    train,validation=saved['train'],saved['val']
    new=json.loads(Path('runs/run17_bag_confounds/new_data_manifest.json').read_text())
    test=[r for r in new if r['split']=='test']
    check_splits(train+validation+test);assert_development_allowed(train+validation)
    reserve=json.loads(Path('dataset/test_only_reserve/manifest.json').read_text())
    assert not {r['decoded_sha256'] for r in reserve}&{r['decoded_sha256'] for r in train+validation}
    g.write_json(OUT/'training_manifest.json',saved)
    protocol={'base_checkpoint':str(BASE),'base_sha256':checksum,'threshold':THRESHOLD,
              'cnn_weights_updated':False,'refiner_trained':True,'reserved_data_used_for_development':False,
              'features':'Frozen CNN probability plus peak-normalized waveform energy, duration, spectral and zero-crossing features at sub-frame resolution.',
              'training_recordings_including_variants':len(train),'validation_recordings':len(validation),
              'new_heldout_test_recordings':len(test),'selection':'Validation macro-domain quality with original positive recall retention; no benchmark predictions used.',
              'candidates':[[2,.05,250],[3,.03,400],[2,.025,600]],'seed':1818}
    g.write_json(REPORT/'protocol.json',protocol)
    cache=Path('dataset/run17_bag_confounds/waveform_features')
    xt,yt=matrix(net,train,cache);xv,yv=matrix(net,validation,cache);xs,ys=matrix(net,test,cache)
    before=write_predictions(test,ys,xs[:,0],REPORT/'before_new_test')
    _,_,reference=score(yv,xv[:,0],validation,None)
    counts=Counter((r['domain'],int(r['label'])) for r in train)
    def importance(domain):return 5. if domain=='IndependentBagPops' else 2. if domain in ['IndependentBalloonPops','LabImpulses','M16Firecrackers'] else 1.
    denominators={label:sum(importance(d) for d,l in counts if l==label) for label in [0,1]}
    weights=np.array([len(train)*.5*importance(r['domain'])/(denominators[int(r['label'])]*counts[(r['domain'],int(r['label']))]) for r in train])
    best=None;selected=None;selection=None;history=[]
    for depth,lr,estimators in protocol['candidates']:
        model=GradientBoostingClassifier(init=CNNPrior(),n_estimators=estimators,learning_rate=lr,
             max_depth=depth,min_samples_leaf=12,subsample=.85,random_state=1818)
        model.fit(xt,yt,sample_weight=weights)
        for stage,prediction in enumerate(model.staged_predict_proba(xv),1):
            if stage%10:continue
            retained,quality,strata=score(yv,prediction[:,1],validation,reference)
            row={'depth':depth,'learning_rate':lr,'stage':stage,'retention_met':retained,'quality':quality,'validation_strata':strata}
            history.append(row)
            if best is None or (retained,quality)>best:
                best=(retained,quality);selection=row
                selected=copy.deepcopy(model);selected.estimators_=selected.estimators_[:stage]
                selected.n_estimators=stage;selected.n_estimators_=stage
        print('Candidate',depth,lr,estimators,'best validation',best,flush=True)
    g.write_json(REPORT/'selection.json',selection);g.write_json(REPORT/'validation_history.json',history)
    joblib.dump(selected,OUT/'refiner.joblib')
    development={r['decoded_sha256'] for r in train+validation}
    assert not {r['decoded_sha256'] for r in test+reserve}&development
    after=write_predictions(test,ys,selected.predict_proba(xs)[:,1],REPORT/'after_new_test')
    g.write_json(REPORT/'new_test_comparison.json',{'before':before,'after':after})
    # Only now read the reserved audio. The model and operating point are frozen.
    net.eval();vectors=[];labels=[]
    with torch.inference_mode():
        for i,row in enumerate(reserve):
            audio=g.read_clip(row['path']);x=torch.from_numpy(feature(audio,ckpt['config']).astype(np.float32))
            p=float(g.recording_logits(net,[x],torch.device('cpu')).sigmoid().item())
            vectors.append(np.r_[p,waveform_features(audio)]);labels.append(row['label'])
            if (i+1)%100==0:print('Reserved test',i+1,'/800',flush=True)
    prob=selected.predict_proba(np.asarray(vectors,dtype=np.float32))[:,1]
    metrics=write_predictions(reserve,np.array(labels),prob,REPORT/'external_test')
    by={}
    for name in dict.fromkeys(r['dataset'] for r in reserve):
        ix=np.array([r['dataset']==name for r in reserve]);by[name]=metric(np.array(labels)[ix],prob[ix],THRESHOLD)
    baseline=json.loads(Path('reports/run14_external_test/results.json').read_text())
    assert fingerprint(BASE)==checksum
    target=metrics['accuracy']>=baseline['combined']['accuracy']+.05 and metrics['gunshot_recall']>=baseline['combined']['gunshot_recall'] and metrics['false_alarms']<baseline['combined']['false_alarms']
    result={'protocol':protocol,'base_checkpoint_unchanged':True,'refiner_sha256':fingerprint(OUT/'refiner.joblib'),
            'combined':metrics,'by_dataset':by,'baseline':baseline['combined'],'target_met':target,
            'reserved_manifest_sha256':fingerprint(Path('dataset/test_only_reserve/manifest.json'))}
    g.write_json(REPORT/'results.json',result)
    g.write_json(OUT/'bundle.json',{'type':'cnn_with_impulse_refiner','cnn_checkpoint':str(BASE),
                 'cnn_sha256':checksum,'refiner':str(OUT/'refiner.joblib'),'threshold':THRESHOLD,
                 'seen_hashes':sorted(set(ckpt['seen_hashes'])|development)})
    print('REFINER FINAL',json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
