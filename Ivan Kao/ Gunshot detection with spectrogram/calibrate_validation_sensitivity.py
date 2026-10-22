"""Freeze a higher-sensitivity operating point using development validation only."""
import argparse
import json
import joblib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import gunshot as g
from continual_cycles import fingerprint,metric
from impulse_refiner import spectrogram_summary,ProbabilityBlend,cnn_embedding
from predict_impulse_refined import load_bundle
from run_large_negative_cycles import vectors
from training_data_policy import assert_development_allowed
from train_joint_calibrated import predictions


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--minimum-recall',type=float,default=.97)
    parser.add_argument('--anchor',type=Path)
    parser.add_argument('--comparison-run',default='run20_vocal_sound')
    parser.add_argument('--minimum-accuracy-gain',type=float,default=0.)
    args=parser.parse_args();assert 0<args.minimum_recall<=1
    out=Path('runs')/args.run;report=Path('reports')/args.run
    out.mkdir(exist_ok=False);report.mkdir(exist_ok=False);torch.set_num_threads(2)
    config,cfg,net,model=load_bundle(args.previous);previous_model=model
    saved=json.loads((args.previous.parent/'training_manifest.json').read_text())
    val=saved['val'];assert_development_allowed(val)
    def matrix(rows):
        from model_feature_matrix import matrix as feature_matrix
        return feature_matrix(net,rows,cfg,config)
    x,y=matrix(val);p=model.predict_proba(x)[:,1];original_probability=p.copy()
    grid=np.arange(.01,.901,.005)
    viable=[float(t) for t in grid if (p[y==1]>t).mean()>=args.minimum_recall]
    assert viable,'No threshold meets the prescribed validation sensitivity'
    threshold=max(viable)
    selection_history=[];anchor_config=None;selected_anchor_weight=0.
    if args.anchor:
        anchor_config,anchor_cfg,anchor_net,anchor=load_bundle(args.anchor)
        assert anchor_config['cnn_sha256']==config['cnn_sha256']
        anchor_probability=anchor.predict_proba(x[:,:anchor.n_features_in_])[:,1]
        best=(-1.,0.)
        for weight in [0.,.1,.25,.5,.75,1.]:
            blend=(1-weight)*p+weight*anchor_probability
            thresholds=[float(t) for t in grid if (blend[y==1]>t).mean()>=args.minimum_recall]
            if not thresholds:continue
            t=max(thresholds);specificity=float((blend[y==0]<=t).mean())
            selection_history.append({'anchor_weight':weight,'threshold':t,'validation_specificity':specificity,'validation_recall':float((blend[y==1]>t).mean())})
            if (specificity,-weight)>best:
                best=(specificity,-weight);threshold=t;selected_anchor_weight=weight
        model=ProbabilityBlend(model,anchor,selected_anchor_weight)
        p=model.predict_proba(x)[:,1]
    g.write_json(report/'validation_blend_history.json',selection_history)
    protocol={'previous_bundle':str(args.previous),'training_performed':False,'cnn_updated':False,
        'refiner_updated':bool(args.anchor),'refiner_weights_retrained':False,'minimum_validation_gunshot_recall':args.minimum_recall,
        'selection':'Largest threshold on the fixed .005 grid meeting prescribed pooled validation recall. Optional anchor blend selected by maximum validation negative rejection.',
        'anchor_bundle':str(args.anchor) if args.anchor else None,'anchor_weight_candidates':[0.,.1,.25,.5,.75,1.] if args.anchor else None,
        'validation_rows':len(val),'validation_gunshots':int(sum(y==1)),
        'reserved_used_for_training_validation_or_threshold_selection':False,
        'baseline_used_for_adaptive_stopping_and_strategy_feedback':True}
    g.write_json(report/'protocol.json',protocol)
    g.write_json(report/'validation_before.json',metric(y,original_probability,config['threshold']))
    g.write_json(report/'validation_after.json',metric(y,p,threshold))
    selection={'threshold':threshold,'minimum_validation_recall':args.minimum_recall,
        'validation_recall':float((p[y==1]>threshold).mean()),'anchor_weight':selected_anchor_weight,'added_trees':0,
        'selected_before_reserved_prediction':True}
    g.write_json(report/'selection.json',selection)
    joblib.dump(model,out/'refiner.joblib',compress=3)
    updated={**config,'refiner':str(out/'refiner.joblib'),'refiner_sha256':fingerprint(out/'refiner.joblib'),'threshold':threshold}
    if anchor_config:updated['seen_hashes']=sorted(set(config['seen_hashes'])|set(anchor_config['seen_hashes']))
    g.write_json(out/'bundle.json',updated)
    g.write_json(out/'training_manifest.json',saved)
    source_manifest=json.loads((args.previous.parent/'new_data_manifest.json').read_text())
    g.write_json(out/'new_data_manifest.json',source_manifest)
    tests=[r for r in source_manifest if r['split']=='test'];x,y=matrix(tests);p=model.predict_proba(x)[:,1]
    old_probability=previous_model.predict_proba(x[:,:previous_model.n_features_in_])[:,1]
    before=predictions(tests,y,old_probability,config['threshold'],report/'before_test')
    after=predictions(tests,y,p,threshold,report/'after_test')
    by={}
    for name in sorted({r['domain'] for r in tests}):
        ix=np.array([r['domain']==name for r in tests]);by[name]={'before':metric(y[ix],old_probability[ix],config['threshold']),'after':metric(y[ix],p[ix],threshold)}
    g.write_json(report/'results.json',{'protocol':protocol,'before':before,'after':after,'by_dataset':by,'selection':selection,'bundle':str(out/'bundle.json')})
    subprocess.run([sys.executable,'evaluate_test_only_reserves.py','--bundle',str(out/'bundle.json'),'--out',str(report/'external_test')],check=True)
    a=json.loads((Path('reports')/args.comparison_run/'external_test/results.json').read_text())
    b=json.loads((report/'external_test/results.json').read_text());assert a['manifest_sha256']==b['manifest_sha256']
    a,b=a['combined'],b['combined'];target=b['accuracy']>a['accuracy'] and b['accuracy']>=a['accuracy']+args.minimum_accuracy_gain-1e-12 and b['gunshot_recall']>=a['gunshot_recall'] and b['false_alarms']<a['false_alarms']
    comparison={'comparison_run':args.comparison_run,'minimum_accuracy_gain':args.minimum_accuracy_gain,'before':a,'after':b,'target_met':target,'baseline_used_for_adaptive_stopping':True}
    g.write_json(report/'baseline_comparison.json',comparison)
    g.write_json(Path('reports/latest_model.json'),{'run':args.run,'model_type':'cnn_with_impulse_refiner','bundle':str(out/'bundle.json'),
        'checkpoint':updated['cnn_checkpoint'],'sha256':updated['cnn_sha256'],'refiner':updated['refiner'],'refiner_sha256':updated['refiner_sha256'],'threshold':threshold,'report':str(report/'baseline_comparison.json')})
    print('SENSITIVITY CALIBRATION COMPLETE',json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':main()
