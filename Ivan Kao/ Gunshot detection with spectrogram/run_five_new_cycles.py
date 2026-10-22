"""Five fixed test -> continued-train -> held-out retest cycles.

The three external reserves are evaluated once after all five checkpoints have
been selected using development data only. Their scores never select weights.
"""
import csv
import fcntl
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import gunshot as g
from continual_cycles import feature, fingerprint, infer, metric
from train_joint_four import load_rows
from training_data_policy import assert_development_allowed

DATA = Path('dataset/five_new_cycles')
REPORT = Path('reports/five_new_cycles')
INITIAL = Path('runs/run9_joint_four/best.pt')
THRESHOLD = .20999999999999996
PLAN = [('run10_audiomnist','AudioMNIST'), ('run11_office','DCASEOffice'),
        ('run12_demand','DEMAND'), ('run13_forest','TropicalForest'),
        ('run14_m16_firecrackers','M16Firecrackers')]


def strata_metrics(y, p, rows):
    values = {}
    pred = p > THRESHOLD
    for domain, label in sorted({(r['domain'], int(r['label'])) for r in rows}):
        ix = np.array([r['domain']==domain and int(r['label'])==label for r in rows])
        values[f'{domain}:{label}'] = float((pred[ix] == y[ix]).mean())
    return values


def check_splits(rows):
    for a,b in [('train','val'), ('train','test'), ('val','test')]:
        left=[r for r in rows if r['split']==a];right=[r for r in rows if r['split']==b]
        for field in ['path','decoded_sha256']:
            assert not {r[field] for r in left} & {r[field] for r in right},field
        assert not {(r['domain'],r['source_group']) for r in left} & {(r['domain'],r['source_group']) for r in right}
    assert_development_allowed([r for r in rows if r['split'] in ('train','val')])


def prepare(cfg, original, checkpoint, plans=PLAN, previous_rows=()):
    reserved=json.loads(Path('dataset/test_only_reserve/manifest.json').read_text())
    blocked=set(checkpoint['seen_hashes']) | {r['decoded_sha256'] for r in reserved+original+list(previous_rows)}
    output={};all_rows=[]
    for run,name in plans:
        source=DATA/name/'raw_manifest.json'
        if not source.exists():raise FileNotFoundError(f'Dataset preparation unfinished: {source}')
        cache=DATA/name/'features';cache.mkdir(exist_ok=True)
        rows=[];excluded=[]
        # Exclude duplicates deterministically before any prediction/training.
        for row in json.loads(source.read_text()):
            audio=g.read_clip(row['path']);digest=hashlib.sha256(audio.tobytes()).hexdigest()
            if digest in blocked:
                excluded.append({'path':row['path'],'reason':'decoded duplicate of earlier selected, prior, or reserved audio'})
                continue
            blocked.add(digest)
            path=cache/(digest+'.npy')
            if not path.exists():np.save(path,feature(audio,cfg))
            rows.append({**row,'feature_path':str(path.resolve()),'decoded_sha256':digest,
                         'sha256':fingerprint(Path(row['path']))})
        check_splits(rows)
        assert all(any(r['split']==s for r in rows) for s in ['train','val','test'])
        g.write_json(DATA/name/'manifest.json',rows)
        g.write_json(DATA/name/'excluded_duplicates.json',excluded)
        output[name]=rows;all_rows.extend(rows)
        print('PREPARED',name,len(rows),'excluded',len(excluded),flush=True)
    check_splits(original+list(previous_rows)+all_rows)
    return output


def balanced_subset(rows, maximum=100, seed=1414):
    groups=defaultdict(list)
    for row in rows:groups[(row['domain'],int(row['label']))].append(row)
    rng=random.Random(seed);selected=[]
    for key in sorted(groups):
        group=sorted(groups[key],key=lambda r:r['path']);rng.shuffle(group)
        selected.extend(group[:maximum])
    return selected


class CycleFeatures(Dataset):
    def __init__(self, rows, new_domain):
        assert_development_allowed(rows)
        assert all(r['split']=='train' for r in rows)
        self.rows=rows
        counts=Counter((r['domain'],int(r['label'])) for r in rows)
        new_count=sum(domain==new_domain for domain,label in counts)
        old_count=len(counts)-new_count
        self.weights=[]
        for row in rows:
            key=(row['domain'],int(row['label']))
            strata=new_count if row['domain']==new_domain else old_count
            self.weights.append(len(rows)*.5/(strata*counts[key]))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        return torch.from_numpy(np.load(row['feature_path']).astype(np.float32)),float(row['label']),self.weights[i]


def collate(batch):
    x,y,w=zip(*batch)
    return list(x),torch.tensor(y,dtype=torch.float32),torch.tensor(w,dtype=torch.float32)


def evaluate(net, rows, path):
    y,p=infer(net,rows)
    report=metric(y,p,THRESHOLD)
    report['correct_classification_rate']=(report['gunshots_detected']+report['correct_non_gunshots'])/len(rows)
    g.write_json(path.with_suffix('.json'),report)
    with path.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','true_label','probability','predicted_label'])
        writer.writeheader()
        writer.writerows({'path':r['path'],'true_label':int(label),'probability':float(prob),'predicted_label':int(prob>THRESHOLD)}
                         for r,label,prob in zip(rows,y,p))
    return report


def train_cycle(run, name, previous, current, history_rows, original, floors):
    out=Path('runs')/run
    if out.exists():raise ValueError(f'Preserve existing run: {out}')
    out.mkdir(parents=True)
    checksum=fingerprint(previous)
    ckpt=torch.load(previous,map_location='cpu',weights_only=True)
    cfg=dict(ckpt['config']);assert cfg['threshold']==THRESHOLD
    cfg.update(initialization=str(previous),initialization_sha256=checksum,
               training_datasets=sorted(set(cfg.get('training_datasets',[]))|{name}),
               selection='fixed-threshold validation; original-domain retention floors; trained epochs only',seed=1414)
    net=g.model_for(cfg);net.load_state_dict(ckpt['state_dict'])
    teacher=g.model_for(cfg);teacher.load_state_dict(ckpt['state_dict']);teacher.eval()
    for p in teacher.parameters():p.requires_grad_(False)
    test=[r for r in current if r['split']=='test']
    replay=balanced_subset([r for r in original+history_rows if r['split']=='train'])
    training=replay+[r for r in current if r['split']=='train']
    validation=[r for r in original+history_rows+current if r['split']=='val']
    check_splits(training+validation+test)
    all_test={r['decoded_sha256'] for r in original+history_rows+current if r['split']=='test'}
    assert not all_test & {r['decoded_sha256'] for r in training+validation}
    g.write_json(out/'training_manifest.json',{'train':training,'val':validation})
    g.write_json(out/'new_data_manifest.json',current)
    protocol={'dataset':name,'previous_checkpoint':str(previous),'previous_sha256':checksum,
              'fixed_threshold':THRESHOLD,'training_recordings':len(training),
              'new_training_recordings':sum(r['split']=='train' for r in current),
              'validation_recordings':len(validation),'test_recordings':len(test),
              'training':'50% total loss weight for new domain; 50% balanced earlier domain/class replay',
              'optimizer':'AdamW lr=0.0001; cosine decay; gradient norm 5; frozen BatchNorm statistics',
              'epochs_max':10,'early_stopping_patience':3,'minimum_epochs':5,
              'teacher_regularization':.05,'seed':1414,
              'retention_floors':floors,'selection':'prefer original validation strata within 5 percentage points of run9; maximize 0.7 mean + 0.3 minimum stratum correct rate',
              'test_used_for_training_or_selection':False,'reserved_audio_used_for_development':False}
    g.write_json(out/'protocol.json',protocol)
    before=evaluate(net,test,out/'before_test')
    print('BEFORE',run,json.dumps(before),flush=True)
    torch.manual_seed(1414);random.seed(1414);np.random.seed(1414)
    batches=DataLoader(CycleFeatures(training,name),batch_size=16,shuffle=True,collate_fn=collate)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10,eta_min=.00002)
    best=None;stale=0;history=[]
    for epoch in range(1,11):
        net.train()
        for module in net.modules():
            if isinstance(module,torch.nn.BatchNorm2d):module.eval()
        loss_sum=0.
        for bags,labels,weights in batches:
            optimizer.zero_grad(set_to_none=True)
            logits=g.recording_logits(net,bags,torch.device('cpu'))
            with torch.no_grad():soft=g.recording_logits(teacher,bags,torch.device('cpu')).div(2).sigmoid()
            bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')
            kd=torch.nn.functional.binary_cross_entropy_with_logits(logits/2,soft,reduction='none')
            loss=(weights*(bce+.05*4*kd)).mean();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            loss_sum+=loss.item()*len(labels)
        y,p=infer(net,validation);strata=strata_metrics(y,p,validation)
        feasible=all(strata[key]>=floor for key,floor in floors.items())
        quality=.7*float(np.mean(list(strata.values())))+.3*min(strata.values())
        row={'epoch':epoch,'training_loss':loss_sum/len(training),'retention_floors_met':feasible,
             'validation_quality':quality,'validation_strata':strata}
        history.append(row);g.write_json(out/'history.json',history)
        print('EPOCH',run,epoch,'loss',round(row['training_loss'],4),'quality',round(quality,4),'retention',feasible,flush=True)
        if best is None or (feasible,quality)>best:
            best=(feasible,quality);stale=0
            updated={**ckpt,'config':dict(cfg),'state_dict':net.state_dict(),'epoch':epoch,
                     'seen_hashes':sorted(set(ckpt['seen_hashes'])|{r['decoded_sha256'] for r in training+validation}),
                     'seen_sources':sorted(set(ckpt['seen_sources'])|{r['source_group'] for r in training+validation})}
            torch.save(updated,out/'best.pt');g.write_json(out/'selection.json',row)
        else:stale+=1
        scheduler.step()
        if epoch>=5 and stale>=3:break
    final=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    assert any(not torch.equal(v,ckpt['state_dict'][k]) for k,v in final['state_dict'].items())
    assert not all_test & set(final['seen_hashes'])
    assert fingerprint(previous)==checksum
    net.load_state_dict(final['state_dict'])
    after=evaluate(net,test,out/'after_test')
    result={'run':run,'dataset':name,**protocol,'selected_epoch':final['epoch'],
            'trained_weight_changes_adopted':True,'previous_checkpoint_unchanged':True,
            'checkpoint_sha256':fingerprint(out/'best.pt'),'before':before,'after':after}
    g.write_json(out/'results.json',result)
    g.write_json(REPORT/(run+'.json'),result)
    print('AFTER',run,json.dumps(after),flush=True)
    return out/'best.pt',result


def main():
    lock=Path('runs/continual_cycles/training.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    REPORT.mkdir(parents=True,exist_ok=True)
    original=load_rows()
    ckpt=torch.load(INITIAL,map_location='cpu',weights_only=True)
    net=g.model_for(ckpt['config']);net.load_state_dict(ckpt['state_dict'])
    val=[r for r in original if r['split']=='val']
    y,p=infer(net,val)
    floors={key:max(0.,value-.05) for key,value in strata_metrics(y,p,val).items()}
    g.write_json(REPORT/'plan.json',{'cycles':PLAN,'initial_checkpoint':str(INITIAL),
               'fixed_threshold':THRESHOLD,'retention_floors':floors,
               'reserved_final_evaluation_only':True,'baseline':'reports/run9_external_test/results.json'})
    previous=INITIAL;history=[];results=[]
    for run,name in PLAN:
        new=prepare(ckpt['config'],original,ckpt,[(run,name)],history)[name]
        completed=Path('runs')/run/'results.json'
        if completed.exists():
            result=json.loads(completed.read_text())
            assert result['previous_sha256']==fingerprint(previous)
            assert result['dataset']==name and result['trained_weight_changes_adopted']
            previous=completed.parent/'best.pt'
            assert fingerprint(previous)==result['checkpoint_sha256']
            assert json.loads((completed.parent/'new_data_manifest.json').read_text())==new
            print('RESUMED completed',run,flush=True)
        else:
            previous,result=train_cycle(run,name,previous,new,history,original,floors)
        results.append(result);history.extend(new)
        g.write_json(REPORT/'cycles.json',results)
    external=Path('reports/run14_external_test/results.json')
    if external.exists():
        saved=json.loads(external.read_text())
        assert saved['checkpoint_sha256']==fingerprint(previous)
        assert saved['manifest_sha256']==fingerprint(Path('dataset/test_only_reserve/manifest.json'))
        assert saved['threshold']==THRESHOLD
    else:
        subprocess.run([sys.executable,'evaluate_test_only_reserves.py','--checkpoint',str(previous),
                        '--out','reports/run14_external_test'],check=True)
    baseline=json.loads(Path('reports/run9_external_test/results.json').read_text())
    final=json.loads(Path('reports/run14_external_test/results.json').read_text())
    assert baseline['manifest_sha256']==final['manifest_sha256']
    assert baseline['threshold']==final['threshold']==THRESHOLD
    g.write_json(REPORT/'baseline_comparison.json',{'baseline':baseline,'after_five_cycles':final})
    g.write_json(Path('reports/latest_model.json'),{'run':PLAN[-1][0],'checkpoint':str(previous),
                 'sha256':fingerprint(previous),'threshold':THRESHOLD,'report':str(REPORT/'baseline_comparison.json')})
    print('ALL FIVE CYCLES AND FINAL RESERVED COMPARISON COMPLETE',flush=True)


if __name__=='__main__':main()
