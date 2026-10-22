"""Jointly fine-tune on all four existing training splits, then retest held-out files.
External test-only datasets are never loaded by this trainer.
"""
import argparse
import fcntl
import json
import random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gunshot as g
from continual_cycles import Features, fingerprint, infer, metric
from training_data_policy import assert_development_allowed

SOURCES = {'SESA':'run5_sesa_valid','FSDD':'run6_fsdd',
           'EDGE':'run7_edge_distilled','UrbanSound8K':'run8_urban'}


def load_rows():
    rows=[]
    for domain,folder in SOURCES.items():
        records=json.loads((Path('runs')/folder/'new_data_manifest.json').read_text())
        for row in records:
            assert row['domain']==domain
            assert 'test_only' not in row['path'] and 'shotspotter' not in row['path'].lower()
            assert Path(row['feature_path']).is_file(),row['feature_path']
        rows.extend(records)
    for a,b in [('train','val'),('train','test'),('val','test')]:
        left=[r for r in rows if r['split']==a];right=[r for r in rows if r['split']==b]
        assert not {r['path'] for r in left}&{r['path'] for r in right}
        assert not {r['decoded_sha256'] for r in left}&{r['decoded_sha256'] for r in right}
        assert not {(r['domain'],r['source_group']) for r in left}&{(r['domain'],r['source_group']) for r in right}
    assert_development_allowed([r for r in rows if r['split'] in ('train','val')])
    return rows


class WeightedFeatures(Dataset):
    def __init__(self,rows):
        assert_development_allowed(rows)
        if any(r.get('split') != 'train' for r in rows):
            raise ValueError('Optimization accepts only training split recordings.')
        self.rows=rows
        count=Counter((r['domain'],int(r['label'])) for r in rows)
        self.weights=[len(rows)/(len(count)*count[(r['domain'],int(r['label']))]) for r in rows]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        return torch.from_numpy(np.load(row['feature_path']).astype(np.float32)),float(row['label']),self.weights[i]


def collate(batch):
    x,y,w=zip(*batch)
    return list(x),torch.tensor(y,dtype=torch.float32),torch.tensor(w,dtype=torch.float32)


def choose(y,p,rows):
    domains=np.array([r['domain'] for r in rows]);candidates=[]
    for threshold in np.linspace(.03,.95,93):
        by={d:metric(y[domains==d],p[domains==d],float(threshold)) for d in SOURCES}
        quality=[];negatives=[]
        for report in by.values():
            if report['gunshot_recall'] is not None:quality.append(report['gunshot_recall'])
            if report['negative_rejection_rate'] is not None:
                quality.append(report['negative_rejection_rate']);negatives.append(report['negative_rejection_rate'])
        score=.6*float(np.mean(quality))+.4*min(quality)
        feasible=all(value>=.95 for value in negatives)
        candidates.append((feasible,score,float(threshold),by))
    # Prefer candidates meeting the false-alarm constraint, then balance recall
    # and rejection across domains. Only validation predictions enter selection.
    return max(candidates,key=lambda r:(r[0],r[1]))


def evaluate(net,rows,threshold):
    y,p=infer(net,rows)
    by={d:metric(y[np.array([r['domain']==d for r in rows])],p[np.array([r['domain']==d for r in rows])],threshold) for d in SOURCES}
    return {'combined':metric(y,p,threshold),'by_dataset':by},y,p


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--previous',type=Path,default=Path('runs/run8_urban/best.pt'))
    parser.add_argument('--out',type=Path,default=Path('runs/run9_joint_four'))
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--learning-rate',type=float,default=.0001)
    a=parser.parse_args()
    if a.out.exists():raise ValueError('Choose a fresh output directory.')
    lock=Path('runs/continual_cycles/training.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.manual_seed(909);np.random.seed(909);random.seed(909)
    rows=load_rows();splits={s:[r for r in rows if r['split']==s] for s in ['train','val','test']}
    checksum=fingerprint(a.previous);ckpt=torch.load(a.previous,map_location='cpu',weights_only=True)
    cfg=dict(ckpt['config']);cfg.update(initialization=str(a.previous),initialization_sha256=checksum,
                                      learning_rate=a.learning_rate,seed=909,training_datasets=list(SOURCES),
                                      selection='joint domain-balanced validation recall/rejection; best trained epoch only')
    net=g.model_for(cfg);net.load_state_dict(ckpt['state_dict'])
    teacher=g.model_for(cfg);teacher.load_state_dict(ckpt['state_dict']);teacher.eval()
    for p in teacher.parameters():p.requires_grad_(False)
    a.out.mkdir(parents=True)
    g.write_json(a.out/'split_manifest.json',splits)
    protocol={'initialization':str(a.previous),'initialization_sha256':checksum,
              'mode':'joint continued training on four datasets',
              'counts':{s:dict(Counter(r['domain'] for r in group)) for s,group in splits.items()},
              'test_used_for_optimization_or_selection':False,'external_test_only_datasets_loaded':False,
              'shotspotter_used_for_training_or_selection':False,
              'training':'Every training file visited once per epoch; inverse domain/class loss weighting.',
              'knowledge_retention':'Inherited weights; frozen BN statistics; teacher soft-target regularization weight 0.05, temperature 2.',
              'selection':'Prefer <=5% false positives in every negative validation domain; maximize 0.6 mean + 0.4 minimum of per-domain positive recall and negative rejection.',
              'trained_epochs_only':True,'learning_rate':a.learning_rate,'seed':909,
              'SESA_limitation':'Original Freesound source IDs unavailable; file/named-group separation does not prove all shared-source independence.'}
    g.write_json(a.out/'protocol.json',protocol)
    before,_,_=evaluate(net,splits['test'],cfg['threshold'])
    g.write_json(a.out/'before_test.json',before)
    print('COUNTS',json.dumps(protocol['counts']),flush=True)
    print('BEFORE',json.dumps(before['combined']),flush=True)
    train=DataLoader(WeightedFeatures(splits['train']),batch_size=16,shuffle=True,collate_fn=collate)
    optimizer=torch.optim.AdamW(net.parameters(),lr=a.learning_rate,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs,eta_min=a.learning_rate/5)
    history=[];best=None;stale=0
    for epoch in range(1,a.epochs+1):
        net.train()
        for module in net.modules():
            if isinstance(module,torch.nn.BatchNorm2d):module.eval()
        total=0.
        for bags,labels,weights in train:
            optimizer.zero_grad(set_to_none=True)
            logits=g.recording_logits(net,bags,torch.device('cpu'))
            with torch.no_grad():soft=g.recording_logits(teacher,bags,torch.device('cpu')).div(2).sigmoid()
            bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')
            kd=torch.nn.functional.binary_cross_entropy_with_logits(logits/2,soft,reduction='none')
            loss=(weights*(bce+.05*4*kd)).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            total+=loss.item()*len(labels)
        y,p=infer(net,splits['val'])
        feasible,score,threshold,by=choose(y,p,splits['val'])
        row={'epoch':epoch,'weighted_train_loss':total/len(splits['train']),
             'validation_score':score,'false_alarm_constraint_met':feasible,'threshold':threshold,
             'validation_by_dataset':by}
        history.append(row);g.write_json(a.out/'history.json',history)
        print(json.dumps({k:v for k,v in row.items() if k!='validation_by_dataset'}),flush=True)
        key=(feasible,score)
        if best is None or key>best:
            best=key;stale=0;cfg['threshold']=threshold
            updated={**ckpt,'state_dict':net.state_dict(),'config':dict(cfg),'epoch':epoch,
                     'seen_hashes':sorted(set(ckpt['seen_hashes'])|{r['decoded_sha256'] for r in rows if r['split']!='test'}),
                     'seen_sources':sorted(set(ckpt['seen_sources'])|{r['source_group'] for r in rows if r['split']!='test'})}
            torch.save(updated,a.out/'best.pt');g.write_json(a.out/'selection.json',row)
        else:stale+=1
        scheduler.step()
        if epoch>=6 and stale>=4:break
    final=torch.load(a.out/'best.pt',map_location='cpu',weights_only=True)
    assert final['epoch']>0
    changed=any(not torch.equal(v,ckpt['state_dict'][k]) for k,v in final['state_dict'].items())
    assert changed,'Expected a genuinely trained checkpoint.'
    net.load_state_dict(final['state_dict'])
    after,y,p=evaluate(net,splits['test'],final['config']['threshold'])
    assert not set(final['seen_hashes'])&{r['decoded_sha256'] for r in splits['test']}
    assert fingerprint(a.previous)==checksum
    g.write_json(a.out/'after_test.json',after)
    np.savez(a.out/'heldout_predictions.npz',true_label=y,probability=p,
             paths=np.array([r['path'] for r in splits['test']]),domains=np.array([r['domain'] for r in splits['test']]))
    result={'before':before,'after':after,'selected_epoch':final['epoch'],
            'trained_weight_changes_adopted':changed,'previous_checkpoint_unchanged':True,
            'checkpoint_sha256':fingerprint(a.out/'best.pt'),'threshold':final['config']['threshold'],
            'training_recordings':len(splits['train']),'validation_recordings':len(splits['val']),
            'test_recordings':len(splits['test'])}
    g.write_json(a.out/'results.json',result)
    print('AFTER',json.dumps(after['combined']),flush=True)
    print('FINISHED JOINT TRAINING',flush=True)


if __name__=='__main__':main()
