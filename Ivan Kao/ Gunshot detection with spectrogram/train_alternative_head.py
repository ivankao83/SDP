"""Select a new correction head on development data while preserving the old expert."""
import copy
import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier

import gunshot as g
from continual_cycles import fingerprint,metric
from predict_impulse_refined import load_bundle
from model_feature_matrix import matrix as feature_matrix
from impulse_refiner import ExpertFeatureHead,ProbabilityBlend,spectrogram_summary,cnn_embedding
from run_large_negative_cycles import vectors
from run_five_new_cycles import check_splits
from training_data_policy import assert_development_allowed
import train_joint_calibrated as c

OUT=Path('runs/run37_blended_correction');REPORT=Path('reports/run37_blended_correction')


def main():
    global OUT,REPORT
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--comparison-run',default='run41_balanced_validation')
    parser.add_argument('--minimum-accuracy-gain',type=float,default=.05)
    parser.add_argument('--adapted-checkpoint',type=Path)
    args=parser.parse_args();OUT=Path('runs')/args.run;REPORT=Path('reports')/args.run
    lock=Path('runs/continual_cycles/training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    config,cfg,net,prior=load_bundle(args.previous);checksum=fingerprint(Path(config['refiner']))
    input_config=copy.deepcopy(config)
    if args.adapted_checkpoint:
        adapted=torch.load(args.adapted_checkpoint,map_location='cpu',weights_only=True)
        assert set(adapted['seen_hashes'])<=set(config['seen_hashes'])
        assert all(adapted['config'][k]==cfg[k] for k in ['model','experiment','bands','frontend'])
        input_config.update(adapted_cnn_features=True,adapted_cnn_checkpoint=str(args.adapted_checkpoint),adapted_cnn_sha256=fingerprint(args.adapted_checkpoint))
    saved=json.loads((args.previous.parent/'training_manifest.json').read_text());train,val=saved['train'],saved['val']
    inherited=json.loads((args.previous.parent/'new_data_manifest.json').read_text())
    tests=[r for r in inherited if r['split']=='test']
    c.NEW.update(r['domain'] for r in inherited)
    check_splits(train+val+tests);assert_development_allowed(train+val)
    reserve=json.loads(Path('dataset/test_only_reserve/manifest.json').read_text())
    assert not {r['decoded_sha256'] for r in reserve+tests}&{r['decoded_sha256'] for r in train+val}
    g.write_json(OUT/'training_manifest.json',saved)
    g.write_json(OUT/'new_data_manifest.json',inherited)
    def matrix(rows):return feature_matrix(net,rows,cfg,input_config)
    def prior_probability(x):return prior.predict_proba(x[:,:prior.n_features_in_])[:,1]
    xt,yt=matrix(train);xv,yv=matrix(val);xs,ys=matrix(tests)
    wrapper=ExpertFeatureHead(prior,None);a=wrapper.inputs(xt);b=wrapper.inputs(xv)
    initial=prior_probability(xv);keys,masks=c.group_matrix(val)
    reference=((initial>config['threshold'])==yv).astype(np.float32)@masks
    c.POOLED_SPECIFICITY=True;c.GROUP_COUNTS=np.array([sum(r['domain']==d and r['label']==l for r in val) for d,l in keys])
    _,baseq,_=c.calibrated_scores(yv,initial,np.array([config['threshold']]),keys,masks,reference)
    threshold_grid=np.unique(np.r_[np.arange(.01,.901,.005),config['threshold']])
    protocol={'previous_bundle':str(args.previous),'previous_refiner_sha256':checksum,
        'previous_expert_frozen':True,'cnn_frozen':True,'training_rows':len(train),'validation_rows':len(val),
        'heldout_new_test_rows':len(tests),'selection':'Validation-only constrained specificity; pooled positive recall retained, macro positive recall within2pp, large negative groups within5pp. Best trained head within .001 of prior validation specificity may replace prior.',
        'candidates':['HistGB leaves15 L2=5 max_features=.5','HistGB leaves7 L2=10 max_features=.8','ExtraTrees 400 depth16 minleaf4 max_features=.5'],
        'threshold_grid':threshold_grid.tolist(),'reserved_used_for_training_or_validation':False,
        'correction_features':'Existing CNN/waveform/peak-aligned spectrogram features, optional frozen CNN pattern embeddings, and frozen expert probability.',
        'seed':3636,'maximum_histogram_iterations':300,'blend_weights':[.02,.05,.1,.2,.35,.5,.75,1.]}
    protocol.update(comparison_run=args.comparison_run,minimum_accuracy_gain=args.minimum_accuracy_gain,adapted_cnn_checkpoint=input_config.get('adapted_cnn_checkpoint'),adapted_cnn_sha256=input_config.get('adapted_cnn_sha256'))
    g.write_json(REPORT/'protocol.json',protocol)
    before=c.predictions(tests,ys,prior_probability(xs),config['threshold'],REPORT/'before_test')
    counts=Counter((r['domain'],r['label']) for r in train)
    def importance(d):return 3. if d in ['IndependentFieldMixed','AydemirGunshots'] else 5. if d in ['IndependentBagPops','IndependentPackaging','IndependentBalloonCollection'] else 2. if d in ('IndependentBalloonPops','LabImpulses','M16Firecrackers') else 1.
    mass={l:sum(importance(d) for d,k in counts if k==l) for l in (0,1)}
    weights=np.array([len(train)*.5*importance(r['domain'])/(mass[r['label']]*counts[(r['domain'],r['label'])]) for r in train])
    best=None;selected=None;selection=None;history=[]
    previous_probability=prior_probability(xv)
    plans=[('hist15',HistGradientBoostingClassifier(max_iter=300,learning_rate=.03,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=5,max_features=.5,early_stopping=False,random_state=3636)),
           ('hist7',HistGradientBoostingClassifier(max_iter=300,learning_rate=.03,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10,max_features=.8,early_stopping=False,random_state=3636)),
           ('extra_trees',ExtraTreesClassifier(n_estimators=400,max_depth=16,min_samples_leaf=4,max_features=.5,n_jobs=2,random_state=3636))]
    for name,head in plans:
        print('FIT HEAD',name,flush=True);head.fit(a,yt,sample_weight=weights)
        joblib.dump(head,OUT/(name+'_full.joblib'))
        staged=enumerate(head.staged_predict_proba(b),1) if name.startswith('hist') else [(400,head.predict_proba(b))]
        for stage,p in staged:
            if name.startswith('hist') and stage%5:continue
            for weight in protocol['blend_weights']:
                blended=(1-weight)*previous_probability+weight*p[:,1]
                ok,q,strata=c.calibrated_scores(yv,blended,threshold_grid,keys,masks,reference)
                ix=max(range(len(q)),key=lambda i:(bool(ok[i]),float(q[i])))
                row={'head':name,'stage':stage,'blend_weight':weight,'retention_met':bool(ok[ix]),'validation_specificity':float(q[ix]),'threshold':float(threshold_grid[ix])}
                history.append(row)
                if row['retention_met'] and row['validation_specificity']>=float(baseq[0])-.001:
                    quality=row['validation_specificity']
                    if best is None or (quality,weight)>(best,selection['blend_weight']):
                        best=quality;selected=copy.deepcopy(head);selection=row
                        if name.startswith('hist'):selected._predictors=selected._predictors[:stage];selected.max_iter=stage
                        np.testing.assert_allclose(selected.predict_proba(b[:10]),p[:10],atol=1e-10)
        print('HEAD VALIDATION',name,'best',selection,flush=True)
    g.write_json(REPORT/'validation_history.json',history)
    if selected is None:
        model=prior;threshold=config['threshold'];selection={'head':'prior_retained','stage':0,'threshold':threshold}
    else:
        correction=ExpertFeatureHead.from_fitted(copy.deepcopy(prior),selected)
        model=ProbabilityBlend(copy.deepcopy(prior),correction,selection['blend_weight']);threshold=selection['threshold']
    g.write_json(REPORT/'selection.json',selection);joblib.dump(model,OUT/'refiner.joblib')
    bundle={**(input_config if selected is not None else config),'refiner':str(OUT/'refiner.joblib'),'refiner_sha256':fingerprint(OUT/'refiner.joblib'),
            'threshold':threshold,'seen_hashes':sorted(set(config['seen_hashes'])|{r['decoded_sha256'] for r in train+val})}
    g.write_json(OUT/'bundle.json',bundle)
    p=model.predict_proba(xs[:,:model.n_features_in_])[:,1];after=c.predictions(tests,ys,p,threshold,REPORT/'after_test')
    by={}
    for name in sorted({r['domain'] for r in tests}):
        ix=np.array([r['domain']==name for r in tests]);by[name]={'after':metric(ys[ix],p[ix],threshold)}
    g.write_json(REPORT/'results.json',{'protocol':protocol,'before':before,'after':after,'by_dataset':by,'selection':selection,'bundle':str(OUT/'bundle.json')})
    assert fingerprint(Path(config['refiner']))==checksum
    subprocess.run([sys.executable,'evaluate_test_only_reserves.py','--bundle',str(OUT/'bundle.json'),'--out',str(REPORT/'external_test')],check=True)
    baseline=json.loads((Path('reports')/args.comparison_run/'external_test/results.json').read_text())
    final=json.loads((REPORT/'external_test/results.json').read_text());assert baseline['manifest_sha256']==final['manifest_sha256']
    a,b=baseline['combined'],final['combined'];success=b['accuracy']>a['accuracy'] and b['accuracy']>=a['accuracy']+args.minimum_accuracy_gain-1e-12 and b['gunshot_recall']>=a['gunshot_recall'] and b['false_alarms']<a['false_alarms']
    g.write_json(REPORT/'baseline_comparison.json',{'before':a,'after':b,'target_met':success,'baseline_used_for_adaptive_stopping':True})
    g.write_json(Path('reports/latest_model.json'),{'run':args.run,'model_type':'cnn_with_impulse_refiner','bundle':str(OUT/'bundle.json'),
        'checkpoint':bundle['cnn_checkpoint'],'sha256':bundle['cnn_sha256'],'refiner':bundle['refiner'],
        'refiner_sha256':bundle['refiner_sha256'],'threshold':threshold,'report':str(REPORT/'baseline_comparison.json')})
    print('ALTERNATIVE HEAD COMPLETE',json.dumps({'before':a,'after':b,'target_met':success},indent=2),flush=True)


if __name__=='__main__':main()
