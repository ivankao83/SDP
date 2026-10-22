"""Joint new-corpus cycle; fixed operating point; reserved data test-only."""
import argparse
import csv
import fcntl
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import gunshot as g
from continual_cycles import feature, fingerprint, infer, metric
from train_joint_four import load_rows
from run_five_new_cycles import PLAN, balanced_subset, check_splits, collate
from training_data_policy import assert_development_allowed

NEW_DOMAINS=['GreatestHits','SixKSFX','TUTRare2017','HuntingGunshots']
THRESHOLD=.20999999999999996


def old_rows():
    rows=load_rows()
    for run,name in PLAN:
        rows+=json.loads((Path('runs')/run/'new_data_manifest.json').read_text())
    check_splits(rows)
    return rows


def prepare(root,cfg,old,ckpt):
    reserve=json.loads(Path('dataset/test_only_reserve/manifest.json').read_text())
    blocked=set(ckpt['seen_hashes'])|{r['decoded_sha256'] for r in old+reserve}
    rows=[];excluded=[]
    for domain in NEW_DOMAINS:
        folder=root/domain;cache=folder/'features';cache.mkdir(exist_ok=True)
        current=[]
        for row in json.loads((folder/'raw_manifest.json').read_text()):
            audio=g.read_clip(row['path']);digest=hashlib.sha256(audio.tobytes()).hexdigest()
            if digest in blocked:
                excluded.append({'path':row['path'],'reason':'exact decoded duplicate of prior, reserved, or earlier selected audio'})
                continue
            blocked.add(digest);path=cache/(digest+'.npy')
            if not path.exists():np.save(path,feature(audio,cfg))
            current.append({**row,'decoded_sha256':digest,'sha256':fingerprint(Path(row['path'])),
                            'feature_path':str(path.resolve())})
        rows+=current
        print('PREPARED',domain,len(current),dict(Counter(r['split'] for r in current)),flush=True)
    check_splits(old+rows)
    assert_development_allowed([r for r in rows if r['split'] in ['train','val']])
    g.write_json(root/'manifest.json',rows);g.write_json(root/'excluded.json',excluded)
    return rows


def augment_real_negatives(rows,root,cfg):
    """Room/device variations of training originals only; never independent data."""
    from scipy.signal import butter,sosfilt,resample_poly
    import soundfile as sf
    folder=root/'training_augmentation';folder.mkdir(exist_ok=True)
    rng=np.random.default_rng(1619);augmented=[]
    for row in rows:
        if row['split']!='train' or row['label']!=0:continue
        count=48 if row['domain']=='IndependentBagPops' else 24 if row['domain'] in ['IndependentBalloonPops','LabImpulses'] else 1 if row['domain']=='CMURealSounds' else 0
        if not count:continue
        original=g.read_clip(row['path'])
        for i in range(count):
            # Sample-rate, filtering, position, noise and simple echo variation.
            up=int(rng.choice([8,9,10,11,12]));audio=resample_poly(original,up,10).astype(np.float32)
            audio=sosfilt(butter(2,[float(rng.uniform(30,200)),float(rng.uniform(5500,11000))],btype='bandpass',fs=g.SR,output='sos'),audio).astype(np.float32)
            audio=audio[:int(2.4*g.SR)];peak=max(float(np.abs(audio).max()),1e-8);audio=audio/peak
            y=np.zeros(g.SAMPLES,dtype=np.float32);offset=int(rng.uniform(.1,.5)*g.SR);count_audio=min(len(audio),len(y)-offset);y[offset:offset+count_audio]+=audio[:count_audio]
            for low,high in [(.01,.05),(.06,.18),(.2,.4)]:
                delay=int(rng.uniform(low,high)*g.SR);gain=float(rng.uniform(.03,.3))
                stop=min(len(y),offset+delay+len(audio));y[offset+delay:stop]+=gain*audio[:stop-offset-delay]
            y+=rng.normal(0,float(rng.uniform(.0001,.002)),len(y)).astype(np.float32)
            y/=max(1.,float(np.abs(y).max()))
            digest=hashlib.sha256(y.tobytes()).hexdigest();path=folder/(digest+'.wav');fp=folder/(digest+'.npy')
            if not path.exists():sf.write(path,y,g.SR,subtype='FLOAT')
            if not fp.exists():np.save(fp,feature(y,cfg))
            augmented.append({**row,'path':str(path.resolve()),'feature_path':str(fp.resolve()),
                              'decoded_sha256':digest,'sha256':fingerprint(path),'augmentation':True,
                              'parent_path':row['path'],'parent_decoded_sha256':row['decoded_sha256']})
    print('Generated training-only negative variations',len(augmented),flush=True)
    return augmented


def assess(y,p,rows,old_domain_names):
    pred=p>THRESHOLD
    strata={}
    for domain,label in sorted({(r['domain'],int(r['label'])) for r in rows}):
        ix=np.array([r['domain']==domain and int(r['label'])==label for r in rows])
        strata[f'{domain}:{label}']=float((pred[ix]==y[ix]).mean())
    old=[v for k,v in strata.items() if k.rsplit(':',1)[0] in old_domain_names]
    newpos=[v for k,v in strata.items() if k.rsplit(':',1)[0] not in old_domain_names and k.endswith(':1')]
    newneg=[v for k,v in strata.items() if k.rsplit(':',1)[0] not in old_domain_names and k.endswith(':0')]
    # Macro averages prevent a large speech/impact domain dominating selection.
    quality=(.5*float(np.mean(old))+.25*float(np.mean(newpos))+.25*float(np.mean(newneg))
             if newpos and newneg else .5*float(np.mean(old))+.5*float(np.mean(newneg or newpos))
             if newneg or newpos else float(np.mean(old)))
    return strata,quality


def evaluate(net,rows,destination):
    y,p=infer(net,rows);r=metric(y,p,THRESHOLD)
    by={}
    for domain in sorted({x['domain'] for x in rows}):
        ix=np.array([x['domain']==domain for x in rows]);by[domain]=metric(y[ix],p[ix],THRESHOLD)
    g.write_json(destination.with_suffix('.json'),{'combined':r,'by_dataset':by})
    with destination.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','path','true_label','probability','predicted_label'])
        writer.writeheader();writer.writerows({'dataset':row['domain'],'path':row['path'],'true_label':int(label),
            'probability':float(prob),'predicted_label':int(prob>THRESHOLD)} for row,label,prob in zip(rows,y,p))
    return {'combined':r,'by_dataset':by}


class JointFeatures(Dataset):
    def __init__(self,rows,new_domains,augment=True,class_balance=False,bag_focus=False):
        assert_development_allowed(rows)
        assert all(r['split']=='train' for r in rows)
        self.rows=rows;self.augment=augment
        counts=Counter((r['domain'],int(r['label'])) for r in rows)
        new_count=sum(d in new_domains for d,label in counts);old_count=len(counts)-new_count
        self.weights=[]
        for row in rows:
            key=(row['domain'],int(row['label']));is_new=row['domain'] in new_domains
            mass=2/3 if is_new else 1/3
            importance=1.
            if bag_focus:
                # Half total mass for each binary class. When the new data is
                # negative-only, all positive mass comes from previous training.
                label=key[1]
                new_keys=[k for k in counts if k[0] in new_domains and k[1]==label]
                old_keys=[k for k in counts if k[0] not in new_domains and k[1]==label]
                pool=new_keys if is_new else old_keys
                mass=.5*((2/3 if is_new else 1/3) if new_keys and old_keys else 1.)
                def imp(domain):return 4. if domain=='IndependentBagPops' else 2. if domain=='IndependentBalloonPops' else 1.
                importance=imp(key[0])
                denominator=sum(imp(d) for d,l in pool)
            elif class_balance:
                keys=[k for k in counts if (k[0] in new_domains)==is_new]
                denominator=len({label for domain,label in keys})*sum(label==key[1] for domain,label in keys)
            else:denominator=new_count if is_new else old_count
            self.weights.append(len(rows)*mass*importance/(denominator*counts[key])*row.get('training_weight_multiplier',1.))
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];x=np.load(row['feature_path']).astype(np.float32)
        if self.augment and random.random()<.5:
            # Narrow masks encourage tolerance to microphone frequency response
            # and short interruptions, keeping labels and all windows intact.
            width=random.randint(0,6);start=random.randrange(max(1,x.shape[2]-width+1))
            x[:,:,start:start+width,:]=0
            width=random.randint(0,8);start=random.randrange(max(1,x.shape[3]-width+1))
            x[:,:,:,start:start+width]=0
        return torch.from_numpy(x),float(row['label']),self.weights[index]


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,default=Path('dataset/run15_joint'))
    p.add_argument('--previous',type=Path,default=Path('runs/run14_m16_firecrackers/best.pt'))
    p.add_argument('--out',type=Path,default=Path('runs/run15_joint_impulses'))
    p.add_argument('--epochs',type=int,default=24)
    p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--real-impulses',action='store_true')
    p.add_argument('--bag-focus',action='store_true')
    a=p.parse_args()
    global NEW_DOMAINS
    if a.real_impulses:
        NEW_DOMAINS=['CMURealSounds','LabImpulses','IndependentBalloonPops','RealImpactVariations']
    if a.bag_focus:
        NEW_DOMAINS=['CMURealSounds','LabImpulses','IndependentBalloonPops','IndependentBagPops']
    lock=Path('runs/continual_cycles/training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.out.exists():raise ValueError('Preserve completed runs; choose a fresh output directory.')
    torch.set_num_threads(2);torch.manual_seed(1515);np.random.seed(1515);random.seed(1515)
    checksum=fingerprint(a.previous);ckpt=torch.load(a.previous,map_location='cpu',weights_only=True)
    cfg=dict(ckpt['config']);assert cfg['threshold']==THRESHOLD
    old=old_rows();new=prepare(a.root,cfg,old,ckpt)
    train=balanced_subset([r for r in old if r['split']=='train'],maximum=120,seed=1515)
    train += [r for r in new if r['split']=='train']
    if a.real_impulses or a.bag_focus:
        train+=augment_real_negatives(new,a.root,cfg)
    validation=[r for r in old+new if r['split']=='val']
    test=[r for r in new if r['split']=='test']
    check_splits(train+validation+test)
    a.out.mkdir(parents=True)
    report=Path('reports')/a.out.name;report.mkdir(exist_ok=False)
    cfg.update(initialization=str(a.previous),initialization_sha256=checksum,
               training_datasets=sorted(set(cfg.get('training_datasets',[]))|set(NEW_DOMAINS)),
               seed=1515,selection='fixed-threshold validation, prior-domain recall retention, trained epochs only')
    g.write_json(a.out/'new_data_manifest.json',new)
    g.write_json(a.out/'training_manifest.json',{'train':train,'val':validation})
    net=g.model_for(cfg);net.load_state_dict(ckpt['state_dict'])
    teacher=g.model_for(cfg);teacher.load_state_dict(ckpt['state_dict']);teacher.eval()
    for param in teacher.parameters():param.requires_grad_(False)
    old_names={r['domain'] for r in old};oldval=[r for r in old if r['split']=='val']
    y,prob=infer(net,oldval);reference,_=assess(y,prob,oldval,old_names)
    floors={k:max(0,v-.1) for k,v in reference.items() if k.endswith(':1')}
    protocol={'previous_checkpoint':str(a.previous),'previous_sha256':checksum,'fixed_threshold':THRESHOLD,
              'training_recordings':len(train),'new_training_recordings':sum(r['split']=='train' for r in new),
              'validation_recordings':len(validation),'test_recordings':len(test),
              'datasets':NEW_DOMAINS,'mode':'one joint test-then-train cycle',
              'seed':1515,'learning_rate':a.lr,'epochs_max':a.epochs,
              'selection':'50% old-domain validation correct rate, 25% new positive recall, 25% new negative rejection; prefer prior positive retention within 10pp',
              'retention_floors':floors,'threshold_tuning_performed':False,
              'reserved_data_used_for_training_validation_or_selection':False,
              'baseline_comparison':'reports/run14_external_test/results.json',
              'success_target':{'accuracy_gain_at_least':.05,'gunshot_recall_at_least':.855072463768116},
              'label_limitations':'Hunting labels are semi-supervised (0.25 supervised training weight); synthesis lineage and original hunting-video IDs unavailable.'}
    if a.real_impulses or a.bag_focus:
        protocol['label_limitations']='Impact collection retains author event labels; original SESA IDs unavailable. New balloon source rooms are separated but one uploader spans splits.'
        protocol['training_only_waveform_augmentations']=sum(r.get('augmentation',False) for r in train)
    protocol['loss_weighting']={'new_domain_mass':2/3,'replay_mass':1/3,'balance_classes_before_domains':a.real_impulses}
    if a.bag_focus:
        protocol['loss_weighting']={'global_binary_class_mass':.5,'new_mass_within_each_class_when_present':2/3,
                                   'bag_domain_weight':4.,'balloon_domain_weight':2.,'other_domain_weight':1.}
    g.write_json(a.out/'protocol.json',protocol);g.write_json(report/'protocol.json',protocol)
    before=evaluate(net,test,a.out/'before_test');g.write_json(report/'before_test.json',before)
    print('BEFORE NEW COMBINED',json.dumps(before['combined']),flush=True)
    batches=DataLoader(JointFeatures(train,set(NEW_DOMAINS),class_balance=a.real_impulses,bag_focus=a.bag_focus),batch_size=16,shuffle=True,collate_fn=collate)
    optimizer=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=.02)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs,eta_min=a.lr/10)
    history=[];best=None;stale=0
    for epoch in range(1,a.epochs+1):
        net.train()
        for module in net.modules():
            if isinstance(module,torch.nn.BatchNorm2d):module.eval()
        total=0.
        for bags,labels,weights in batches:
            optimizer.zero_grad(set_to_none=True)
            logits=g.recording_logits(net,bags,torch.device('cpu'))
            with torch.no_grad():soft=g.recording_logits(teacher,bags,torch.device('cpu')).div(2).sigmoid()
            bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')
            kd=torch.nn.functional.binary_cross_entropy_with_logits(logits/2,soft,reduction='none')
            loss=(weights*(bce+.025*4*kd)).mean();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step();total+=loss.item()*len(labels)
        y,prob=infer(net,validation);strata,quality=assess(y,prob,validation,old_names)
        retention=all(strata[k]>=floor for k,floor in floors.items())
        row={'epoch':epoch,'training_loss':total/len(train),'validation_quality':quality,
             'retention_met':retention,'validation_strata':strata}
        history.append(row);g.write_json(a.out/'history.json',history)
        print('EPOCH',epoch,'loss',round(row['training_loss'],4),'quality',round(quality,4),'retention',retention,flush=True)
        if best is None or (retention,quality)>best:
            best=(retention,quality);stale=0
            updated={**ckpt,'config':dict(cfg),'epoch':epoch,'state_dict':net.state_dict(),
                     'seen_hashes':sorted(set(ckpt['seen_hashes'])|{r['decoded_sha256'] for r in train+validation}),
                     'seen_sources':sorted(set(ckpt['seen_sources'])|{r['source_group'] for r in train+validation})}
            torch.save(updated,a.out/'best.pt');g.write_json(a.out/'selection.json',row)
        else:stale+=1
        scheduler.step()
        if epoch>=10 and stale>=6:break
    final=torch.load(a.out/'best.pt',map_location='cpu',weights_only=True)
    assert any(not torch.equal(v,ckpt['state_dict'][k]) for k,v in final['state_dict'].items())
    assert not {r['decoded_sha256'] for r in old+new if r['split']=='test'}&set(final['seen_hashes'])
    assert fingerprint(a.previous)==checksum
    net.load_state_dict(final['state_dict']);after=evaluate(net,test,a.out/'after_test')
    result={**protocol,'selected_epoch':final['epoch'],'before':before,'after':after,
            'checkpoint_sha256':fingerprint(a.out/'best.pt'),'trained_weight_changes_adopted':True,
            'previous_checkpoint_unchanged':True}
    g.write_json(a.out/'results.json',result);g.write_json(report/'results.json',result)
    print('AFTER NEW COMBINED',json.dumps(after['combined']),flush=True)
    external=report/'external_test'
    subprocess.run([sys.executable,'evaluate_test_only_reserves.py','--checkpoint',str(a.out/'best.pt'),'--out',str(external)],check=True)
    baseline=json.loads(Path('reports/run14_external_test/results.json').read_text())
    scored=json.loads((external/'results.json').read_text())
    assert baseline['manifest_sha256']==scored['manifest_sha256'] and baseline['threshold']==scored['threshold']
    b,s=baseline['combined'],scored['combined']
    reached=s['accuracy']>=b['accuracy']+.05 and s['gunshot_recall']>=b['gunshot_recall'] and s['false_alarms']<b['false_alarms']
    comparison={'baseline':baseline,'after':scored,'operational_improvement_target_met':reached,
                'accuracy_change_percentage_points':100*(s['accuracy']-b['accuracy']),
                'false_alarm_reduction':b['false_alarms']-s['false_alarms']}
    g.write_json(report/'baseline_comparison.json',comparison)
    g.write_json(Path('reports/latest_model.json'),{'run':a.out.name,'checkpoint':str(a.out/'best.pt'),
                 'sha256':fingerprint(a.out/'best.pt'),'threshold':THRESHOLD,'report':str(report/'baseline_comparison.json')})
    print('BASELINE TARGET MET',reached,'accuracy',s['accuracy'],'recall',s['gunshot_recall'],'false alarms',s['false_alarms'],flush=True)


if __name__=='__main__':main()
