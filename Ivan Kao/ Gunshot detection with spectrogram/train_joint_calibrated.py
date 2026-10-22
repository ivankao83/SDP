"""Continue a saved system on approved replay and new sources; select on validation."""
import copy
import csv
import fcntl
import json
import subprocess
import sys
import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch

import gunshot as g
from continual_cycles import fingerprint,metric
from predict_impulse_refined import load_bundle
from model_feature_matrix import matrix as feature_matrix
from run_five_new_cycles import check_splits
from run_large_negative_cycles import vectors
from training_data_policy import assert_development_allowed
from impulse_refiner import BoundedGradientBoostingClassifier,RefinerPrior,spectrogram_summary,cnn_embedding

OUT=Path('runs/run25_joint_calibrated');REPORT=Path('reports/run25_joint_calibrated')
NEW={'VietnamForest','RealImpactMeasured','MIMIIValve','ReaLISEDIndoor'}
POOLED_SPECIFICITY=False
GROUP_COUNTS=None


def group_matrix(rows):
    keys=sorted({(r['domain'],int(r['label'])) for r in rows})
    masks=np.array([[r['domain']==d and int(r['label'])==l for r in rows] for d,l in keys],dtype=np.float32)
    return keys,(masks/masks.sum(axis=1,keepdims=True)).T


def calibrated_scores(y,p,thresholds,keys,matrix,reference):
    correct=((p[None,:]>thresholds[:,None])==y[None,:]).astype(np.float32)
    strata=correct@matrix
    positive=np.array([l==1 for d,l in keys]);old=np.array([d not in NEW for d,l in keys])
    if POOLED_SPECIFICITY:
        counts=GROUP_COUNTS
        pooled=(strata[:,positive]@counts[positive])/counts[positive].sum()
        prior_pooled=float(reference[positive]@counts[positive]/counts[positive].sum())
        retained=pooled>=prior_pooled-1e-6
        retained&=strata[:,positive].mean(axis=1)>=reference[positive].mean()-.02-1e-6
        large_negative=(~positive)&(counts>=30)
        retained&=(strata[:,large_negative]>=reference[None,large_negative]-.05-1e-6).all(axis=1)
        reliability=np.minimum(counts/30.,1.)
        oldneg=old&~positive;newneg=~old&~positive
        quality=.7*(strata[:,oldneg]@reliability[oldneg])/reliability[oldneg].sum()+.3*(strata[:,newneg]@reliability[newneg])/reliability[newneg].sum()
    else:
        retained=(strata[:,positive]>=reference[None,positive]-1e-6).all(axis=1)
        retained&=(strata[:,~positive]>=reference[None,~positive]-.05-1e-6).all(axis=1)
        quality=.7*strata[:,old].mean(axis=1)+.3*strata[:,~old].mean(axis=1)
    return retained,quality,strata


def predictions(rows,y,p,threshold,path):
    result=metric(y,p,threshold);g.write_json(path.with_suffix('.json'),result)
    with path.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','path','true_label','probability','predicted_label']);w.writeheader()
        w.writerows({'dataset':r['domain'],'path':r['path'],'true_label':int(a),'probability':float(b),
                     'predicted_label':int(b>threshold)} for r,a,b in zip(rows,y,p))
    return result


def snapshot(model,cut):
    saved=copy.deepcopy(model);saved.estimators_=saved.estimators_[:cut]
    saved.n_estimators=cut;saved.n_estimators_=cut;saved.train_score_=saved.train_score_[:cut]
    for key in ['oob_improvement_','oob_scores_']:
        if hasattr(saved,key):setattr(saved,key,getattr(saved,key)[:cut])
    if hasattr(saved,'oob_scores_'):saved.oob_score_=saved.oob_scores_[-1]
    return saved


def balanced_log_loss(y,p):
    p=np.clip(p,1e-12,1-1e-12)
    return float(np.mean([-np.log(p[y==1]).mean(),-np.log1p(-p[y==0]).mean()]))


def main():
    global OUT,REPORT,POOLED_SPECIFICITY,GROUP_COUNTS
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',default='run25_joint_calibrated')
    parser.add_argument('--previous',type=Path,default=Path('runs/run24_realised/bundle.json'))
    parser.add_argument('--pooled-specificity',action='store_true')
    parser.add_argument('--bounded-updates',action='store_true')
    parser.add_argument('--training-only-manifest',type=Path)
    parser.add_argument('--soft-prior',action='store_true')
    parser.add_argument('--reuse-trained-model',type=Path)
    parser.add_argument('--adaptation-domain')
    parser.add_argument('--adaptation-tolerance',type=float,default=0.)
    parser.add_argument('--spectrogram-summary',action='store_true')
    parser.add_argument('--accept-near-equivalent',type=float,default=0.)
    parser.add_argument('--additional-manifest',type=Path)
    parser.add_argument('--cnn-embedding',action='store_true')
    parser.add_argument('--comparison-run',default='run20_vocal_sound')
    parser.add_argument('--minimum-accuracy-gain',type=float,default=0.)
    args=parser.parse_args();OUT=Path('runs')/args.run;REPORT=Path('reports')/args.run;POOLED_SPECIFICITY=args.pooled_specificity
    lock=Path('runs/continual_cycles/training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    previous_path=args.previous;config,cfg,net,prior=load_bundle(previous_path)
    use_summaries=bool(args.spectrogram_summary or config.get('spectrogram_summary'))
    use_embeddings=bool(args.cnn_embedding or config.get('cnn_embedding'))
    prior_hash=fingerprint(Path(config['refiner']))
    saved=json.loads((previous_path.parent/'training_manifest.json').read_text());train=saved['train'];val=saved['val']
    extra=json.loads(args.training_only_manifest.read_text()) if args.training_only_manifest else []
    if extra:
        assert len(extra)>=200 and all(r['split']=='train' for r in extra)
        assert not set(config['seen_hashes'])&{r['decoded_sha256'] for r in extra},'Additional pretest data was already used for development.'
        train=train+extra;saved={'train':train,'val':val}
    additional=json.loads(args.additional_manifest.read_text()) if args.additional_manifest else []
    if additional:
        blocked=set(config['seen_hashes'])|{r['decoded_sha256'] for r in json.loads(Path('dataset/test_only_reserve/manifest.json').read_text())}
        for historical in Path('runs').glob('*/new_data_manifest.json'):
            blocked.update(r['decoded_sha256'] for r in json.loads(historical.read_text()))
        excluded=[r for r in additional if r['decoded_sha256'] in blocked]
        additional=[r for r in additional if r['decoded_sha256'] not in blocked]
        g.write_json(REPORT/'additional_overlap_exclusions.json',excluded)
        assert len(additional)>=200
        NEW.update(r['domain'] for r in additional)
        train=train+[r for r in additional if r['split']=='train'];val=val+[r for r in additional if r['split']=='val']
        saved={'train':train,'val':val}
    tests=[]
    for run in ['run21_vietnam','run22_realimpact','run23_mimii','run24_realised']:
        rows=json.loads((Path('runs')/run/'new_data_manifest.json').read_text());tests.extend(r for r in rows if r['split']=='test')
    previous_new=previous_path.parent/'new_data_manifest.json'
    inherited_new=json.loads(previous_new.read_text()) if previous_new.exists() else []
    NEW.update(r['domain'] for r in inherited_new)
    known_test={r['decoded_sha256'] for r in tests}
    for r in inherited_new+additional:
        if r['split']=='test' and r['decoded_sha256'] not in known_test:
            tests.append(r);known_test.add(r['decoded_sha256'])
    check_splits(train+val+tests);assert_development_allowed(train+val)
    reserve=json.loads(Path('dataset/test_only_reserve/manifest.json').read_text())
    assert not {r['decoded_sha256'] for r in reserve+tests}&{r['decoded_sha256'] for r in train+val}
    g.write_json(OUT/'training_manifest.json',saved)
    development_names=NEW|{r['domain'] for r in extra}|{r['domain'] for r in train if r.get('role')=='training_after_pretest'}
    g.write_json(OUT/'new_data_manifest.json',[r for r in train+val+tests if r['domain'] in development_names and not r.get('augmentation')])
    thresholds=np.unique(np.r_[np.arange(.01,.901,.005),config['threshold']])
    protocol={'previous_bundle':str(previous_path),'previous_refiner_sha256':prior_hash,
              'new_corpora':sorted(NEW),'cnn_updated':False,'continued_refiner_training':True,
              'training_rows':len(train),'validation_rows':len(val),'new_heldout_test_rows':len(tests),
              'maximum_added_trees':200,'threshold_candidates':thresholds.tolist(),
              'selection':'Joint stage/threshold validation; preserve every positive validation stratum recall and retain negative strata within 5pp. Maximize .7 earlier + .3 new macro-domain correctness.',
              'reserved_used_for_training_validation_or_threshold_selection':False,
              'test_predictions_used_for_selection':False,'seed':2525,
              'baseline_used_only_after_selection_to_decide_whether_further_cycles_are_needed':True}
    if POOLED_SPECIFICITY:
        protocol['selection']='Validation-only constrained specificity: pooled positive recall cannot decrease; macro positive recall may fall at most 2pp; negative groups with >=30 examples may fall at most 5pp. Maximize support-shrunk macro negative rejection, .7 earlier/.3 new.'
    protocol['bounded_newton_updates']=args.bounded_updates
    protocol['additional_training_only_manifest']=str(args.training_only_manifest) if args.training_only_manifest else None
    protocol['additional_training_only_recordings']=len(extra)
    protocol['additional_corpus_after_training_retest_is_in_sample']=bool(extra)
    protocol['additional_source_manifest']=str(args.additional_manifest) if args.additional_manifest else None
    protocol['additional_source_recordings']=len(additional)
    protocol['additional_source_splits']={s:sum(r['split']==s for r in additional) for s in ['train','val','test']}
    protocol['inherited_training_only_recordings']=sum(r.get('role')=='training_after_pretest' for r in saved['train'])
    protocol['frozen_previous_refiner_as_soft_prior']=args.soft_prior
    protocol['additional_peak_aligned_spectrogram_features']=use_summaries
    protocol['frozen_cnn_pattern_embeddings']=use_embeddings
    protocol['comparison_run']=args.comparison_run
    protocol['minimum_accuracy_gain']=args.minimum_accuracy_gain
    protocol['reused_trained_candidate']=str(args.reuse_trained_model) if args.reuse_trained_model else None
    protocol['adaptation_domain']=args.adaptation_domain
    protocol['adaptation_tolerance_in_validation_specificity']=args.adaptation_tolerance
    protocol['trained_candidate_validation_equivalence_tolerance']=args.accept_near_equivalent
    if args.soft_prior:protocol['prior_probability_floor']=.001
    if args.bounded_updates:protocol['maximum_new_leaf_update']=3.
    g.write_json(REPORT/'protocol.json',protocol)
    input_config={**config,'spectrogram_summary':use_summaries,'cnn_embedding':use_embeddings}
    def inputs(rows):return feature_matrix(net,rows,cfg,input_config)
    xt,yt=inputs(train);xv,yv=inputs(val);xs,ys=inputs(tests)
    prior_features=prior.n_features_in_
    def old_probability(x):return prior.predict_proba(x[:,:prior_features])[:,1]
    expected_dimensions=91+(320 if use_summaries else 0)+(64 if use_embeddings else 0)+(65 if config.get('adapted_cnn_features') else 0)
    assert xt.shape[1]==xv.shape[1]==xs.shape[1]==expected_dimensions
    source_before=None;source_after=None;source_x=None;source_y=None
    if additional:
        source_x,source_y=inputs(additional)
        assert source_x.shape[1]==xt.shape[1]
        source_before=predictions(additional,source_y,old_probability(source_x),config['threshold'],REPORT/'new_source_before_training')
    before=predictions(tests,ys,old_probability(xs),config['threshold'],REPORT/'before_test')
    extra_before=None
    if extra:
        xe,ye=inputs(extra)
        extra_before=predictions(extra,ye,old_probability(xe),config['threshold'],REPORT/'additional_before_training')
    keys,matrix=group_matrix(val);p0=old_probability(xv)
    GROUP_COUNTS=np.array([sum(r['domain']==d and int(r['label'])==l for r in val) for d,l in keys],dtype=np.float64)
    reference=(((p0>config['threshold'])==yv).astype(np.float32)@matrix)
    ok,q,_=calibrated_scores(yv,p0,np.array([config['threshold']]),keys,matrix,reference)
    best=(True,float(q[0]));selected=copy.deepcopy(prior);threshold=config['threshold'];selected_added=0
    reference_quality=float(q[0]);best_adaptation_loss=None;xa=ya=None
    if args.adaptation_domain:
        ix=np.array([r['domain']==args.adaptation_domain for r in train]);xa,ya=xt[ix],yt[ix]
        assert len(xa)>=200 and len(set(ya))==2
        best_adaptation_loss=balanced_log_loss(ya,old_probability(xa))
    counts=Counter((r['domain'],r['label']) for r in train)
    def importance(d):return 5. if d=='ClapRIRPhone' else 3. if d in ['IndependentFieldMixed','AydemirGunshots'] else 5. if d in ['IndependentBagPops','IndependentPackaging','IndependentBalloonCollection'] else 2. if d in ('IndependentBalloonPops','LabImpulses','M16Firecrackers') else 1.
    mass={l:sum(importance(d) for d,k in counts if k==l) for l in (0,1)}
    weights=np.array([len(train)*.5*importance(r['domain'])/(mass[r['label']]*counts[(r['domain'],r['label'])]) for r in train])
    model=copy.deepcopy(prior);start=getattr(prior,'n_estimators_',0)
    if args.soft_prior:
        assert args.bounded_updates
        model=BoundedGradientBoostingClassifier(init=RefinerPrior(copy.deepcopy(prior),feature_count=prior_features),n_estimators=200,
             learning_rate=.03,max_depth=3,min_samples_leaf=12,subsample=.85,random_state=2525)
        model.max_leaf_update=3.;start=0
    elif args.bounded_updates:
        model.__class__=BoundedGradientBoostingClassifier
        model.max_leaf_update=3.
        assert np.array_equal(model.predict_proba(xv),prior.predict_proba(xv))
    model.set_params(warm_start=not args.soft_prior,n_estimators=start+200,random_state=2525)
    if args.reuse_trained_model:
        assert args.soft_prior
        model=joblib.load(args.reuse_trained_model)
        assert json.loads((args.reuse_trained_model.parent/'training_manifest.json').read_text())==saved
        prefix_preserved=False
        if model.n_features_in_==prior.n_features_in_ and prior.n_estimators_<=model.n_estimators_:
            prefix=snapshot(model,prior.n_estimators_)
            prefix_preserved=np.array_equal(prefix.predict_proba(xv)[:,1],old_probability(xv))
        if prefix_preserved:start=prior.n_estimators_
        else:
            expert=model.init_.expert
            assert np.array_equal(expert.predict_proba(xv[:,:expert.n_features_in_])[:,1],old_probability(xv))
            start=0
        protocol['preserved_direct_tree_prefix']=start
        g.write_json(REPORT/'protocol.json',protocol)
    else:model.fit(xt,yt,sample_weight=weights)
    joblib.dump(model,OUT/'full_trained_candidate.joblib',compress=3)
    history=[];best_near_equivalent=None
    adaptation_predictions=iter(model.staged_predict_proba(xa)) if xa is not None else None
    for stage,probs in enumerate(model.staged_predict_proba(xv),1):
        adapt=next(adaptation_predictions) if adaptation_predictions is not None else None
        if stage<=start:continue
        retained,quality,values=calibrated_scores(yv,probs[:,1],thresholds,keys,matrix,reference)
        # Prefer a feasible operating point, then quality; ties keep the earliest/lower threshold.
        ix=max(range(len(thresholds)),key=lambda i:(bool(retained[i]),float(quality[i])))
        candidate=(bool(retained[ix]),float(quality[ix]));added=stage-start
        row={'added_trees':added,'retention_met':candidate[0],'validation_quality':candidate[1],
             'threshold':float(thresholds[ix]),'strata':{f'{d}:{l}':float(v) for (d,l),v in zip(keys,values[ix])}}
        history.append(row)
        improve=candidate>best
        if args.accept_near_equivalent:
            improve=candidate[0] and candidate[1]>=reference_quality-args.accept_near_equivalent and (best_near_equivalent is None or candidate>best_near_equivalent)
            if improve:best_near_equivalent=candidate
        if adapt is not None:
            adaptation_loss=balanced_log_loss(ya,adapt[:,1]);row['new_domain_training_balanced_log_loss']=adaptation_loss
            improve=candidate[0] and candidate[1]>=reference_quality-args.adaptation_tolerance and adaptation_loss<best_adaptation_loss
        if improve:
            best=candidate;selected=snapshot(model,stage);threshold=float(thresholds[ix]);selected_added=added
            if adapt is not None:best_adaptation_loss=adaptation_loss
        if added%20==0:print('CALIBRATED STAGE',added,'best',best,'threshold',threshold,'selected added',selected_added,flush=True)
    g.write_json(REPORT/'validation_history.json',history)
    selected_row=next((r for r in history if r['added_trees']==selected_added),None)
    g.write_json(REPORT/'selection.json',{'added_trees':selected_added,'threshold':threshold,'quality':best,
                 'validation_selection':selected_row,'selected_before_reserved_prediction':True})
    joblib.dump(selected,OUT/'refiner.joblib',compress=3)
    updated={**config,'refiner':str(OUT/'refiner.joblib'),'refiner_sha256':fingerprint(OUT/'refiner.joblib'),
             'threshold':threshold,'seen_hashes':sorted(set(config['seen_hashes'])|{r['decoded_sha256'] for r in train+val})}
    if use_summaries:updated['spectrogram_summary']=bool(selected_added>0 or config.get('spectrogram_summary'))
    if use_embeddings:updated['cnn_embedding']=bool(selected_added>0 or config.get('cnn_embedding'))
    g.write_json(OUT/'bundle.json',updated)
    def selected_probability(x):return selected.predict_proba(x[:,:selected.n_features_in_])[:,1]
    after=predictions(tests,ys,selected_probability(xs),threshold,REPORT/'after_test')
    extra_after=predictions(extra,ye,selected_probability(xe),threshold,REPORT/'additional_after_training_seen') if extra else None
    if additional:source_after=predictions(additional,source_y,selected_probability(source_x),threshold,REPORT/'new_source_after_training_contains_seen')
    by={}
    for name in sorted(NEW):
        ix=np.array([r['domain']==name for r in tests])
        if not ix.any():continue
        by[name]={'before':metric(ys[ix],old_probability(xs[ix]),config['threshold']),
                                                                'after':metric(ys[ix],selected_probability(xs[ix]),threshold)}
    g.write_json(REPORT/'results.json',{'protocol':protocol,'before':before,'after':after,'by_dataset':by,
                 'selected_added_trees':selected_added,'threshold':threshold,'bundle':str(OUT/'bundle.json'),
                 'new_source_before_training':source_before,'new_source_after_training_contains_seen':source_after,'additional_before_training':extra_before,'additional_after_training_seen':extra_after})
    assert fingerprint(Path(config['refiner']))==prior_hash
    print('FROZEN CALIBRATED MODEL; EVALUATING BASELINE',flush=True)
    subprocess.run([sys.executable,'evaluate_test_only_reserves.py','--bundle',str(OUT/'bundle.json'),
                    '--out',str(REPORT/'external_test')],check=True)
    a=json.loads((Path('reports')/args.comparison_run/'external_test/results.json').read_text())
    b=json.loads((REPORT/'external_test/results.json').read_text());assert a['manifest_sha256']==b['manifest_sha256']
    target=b['combined']['accuracy']>a['combined']['accuracy'] and b['combined']['accuracy']>=a['combined']['accuracy']+args.minimum_accuracy_gain-1e-12 and b['combined']['gunshot_recall']>=a['combined']['gunshot_recall'] and b['combined']['false_alarms']<a['combined']['false_alarms']
    comparison={'comparison_run':args.comparison_run,'minimum_accuracy_gain':args.minimum_accuracy_gain,'before':a['combined'],'after':b['combined'],'target_met':target,
                'baseline_used_for_stopping_further_cycles':True,'baseline_used_for_training_or_within_run_selection':False}
    g.write_json(REPORT/'baseline_comparison.json',comparison)
    g.write_json(Path('reports/latest_model.json'),{'run':args.run,'model_type':'cnn_with_impulse_refiner',
                 'bundle':str(OUT/'bundle.json'),'checkpoint':updated['cnn_checkpoint'],'sha256':updated['cnn_sha256'],
                 'refiner':updated['refiner'],'refiner_sha256':updated['refiner_sha256'],'threshold':threshold,
                 'report':str(REPORT/'baseline_comparison.json')})
    print('CALIBRATED CYCLE COMPLETE',json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':main()
