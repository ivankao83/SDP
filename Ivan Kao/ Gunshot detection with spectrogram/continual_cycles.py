"""Reproducible test -> continued training -> frozen recheck cycles.

Only split=train optimizes weights; split=val selects epoch/threshold. Test
results are written after model selection and never read by the trainer.
"""
import argparse
import csv
import hashlib
import json
import random
import shutil
import fcntl
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import gunshot as g
from train_c3gd import binary_label, fingerprint, make_pair

ROOT = Path('dataset/continual_cycles')
WORK = Path('runs/continual_cycles')


def metric(y, p, threshold):
    y, p = np.asarray(y, dtype=int), np.asarray(p)
    pred = p > threshold
    tn = int(((y == 0) & ~pred).sum()); fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum()); tp = int(((y == 1) & pred).sum())
    both = tn + fp > 0 and tp + fn > 0
    return {'recordings': len(y), 'accuracy': (tp + tn) / len(y) if both else None,
            'gunshot_recall': tp / (tp + fn) if tp + fn else None,
            'gunshot_precision': tp / (tp + fp) if both and tp + fp else None,
            'gunshot_f1': 2 * tp / (2 * tp + fp + fn) if both and 2 * tp + fp + fn else None,
            'roc_auc': g.metrics(y, p, threshold)['roc_auc'] if both else None,
            'false_positive_rate': fp / (fp + tn) if fp + tn else None,
            'negative_rejection_rate': tn / (tn + fp) if tn + fp else None,
            'threshold': float(threshold), 'gunshots_detected': tp, 'gunshots_missed': fn,
            'false_alarms': fp, 'correct_non_gunshots': tn}


def feature(audio, cfg):
    return g.features(np.stack([g.mel(w, cfg['frontend']) for _, w in g.audio_windows(audio)]), cfg).numpy().astype(np.float16)


class Features(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        return torch.from_numpy(np.load(row['feature_path']).astype(np.float32)), int(row['label'])


def loader(rows, train=False, draws=None):
    sampler = None
    if train:
        from training_data_policy import assert_development_allowed
        assert_development_allowed(rows)
        count = Counter((r['domain'], int(r['label'])) for r in rows)
        weights = [(2 if r.get('new_cycle') else 1) / count[(r['domain'], int(r['label']))] for r in rows]
        sampler = WeightedRandomSampler(weights, min(len(rows),draws) if draws else len(rows), replacement=True)
    return DataLoader(Features(rows), batch_size=16, sampler=sampler, collate_fn=g.collate_recordings)


def infer(net, rows):
    return g.run_epoch(net, loader(rows), torch.device('cpu'))[1:]


def reports(net, rows, threshold, output):
    y, p = infer(net, rows)
    report = {'all': metric(y, p, threshold)}
    for split in sorted(set(r['split'] for r in rows)):
        ix = np.array([r['split'] == split for r in rows])
        report[split] = metric(y[ix], p[ix], threshold)
    g.write_json(output.with_suffix('.json'), report)
    with output.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path','source_group','split','true_label','probability','predicted_label'])
        writer.writeheader()
        writer.writerows({'path':r['path'], 'source_group':r['source_group'], 'split':r['split'],
                          'true_label':int(label), 'probability':float(prob), 'predicted_label':int(prob > threshold)}
                         for r,label,prob in zip(rows,y,p))
    print(output.name, json.dumps(report), flush=True)
    return report


def prepare_replay(cfg):
    manifest = WORK / 'replay_manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if all(Path(r['feature_path']).is_file() for rows in saved.values() for r in rows):return saved
    cache = WORK / 'replay_features'; cache.mkdir(parents=True,exist_ok=True)
    old = json.loads(Path('runs/run4_background_revision/split_manifest.json').read_text())
    splits = {'train': [], 'val': []}
    rng = np.random.default_rng(505)
    def add(split, row, x):
        path = cache / f'{split}_{len(splits[split])}.npy'
        np.save(path, x)
        splits[split].append({**row, 'label': int(row['label']), 'split': split, 'feature_path': str(path.resolve())})
    for split, rows in old.items():
        for i, row in enumerate(rows):
            if row['domain'] == 'new_background': continue
            domain = row['domain']
            if domain == 'c3gd_augmented':
                pair = make_pair(g.read_clip(row['path']), g.read_clip(row['background_path']), rng,
                                 round(row['duration_seconds'] * g.SR), row['snr_db'], row['augmented_sample_rate'])
                audio = pair[0 if int(row['label']) == 1 else 1]
                label = int(row['label'])
            else:
                path = Path(row['path'])
                if domain == 'mad': path = Path('dataset/MAD/MAD_dataset') / path
                audio = g.read_clip(path)
                label = binary_label(row)
            add(split, {**row, 'label': label}, feature(audio, cfg))
            if (i+1) % 700 == 0: print('Replay prepared',split,i+1,flush=True)
        backgrounds = {r['path']:r for r in rows if r['domain'] == 'new_background'}
        for path, row in backgrounds.items():
            x = feature(g.read_clip(path), cfg)
            # Partition the complete continuous-window feature sequence. No
            # boundary windows are dropped, unlike separate 10-second clips.
            for start in range(0, len(x), 8):
                add(split, {**row, 'source_group':path, 'continuous_window_offset':start}, x[start:start+8])
    g.write_json(manifest, splits)
    return splits


def dataset_rows(name):
    rows = []
    if name == 'SESA':
        folder = ROOT / 'SESA/SESA'
        for path in sorted(folder.rglob('*.wav')):
            category = path.stem.split('_')[0]
            rows.append({'path':str(path.resolve()),'label':int(category == 'gunshot'),
                         'category':category, 'source_group':path.stem,
                         'split':'test' if path.parent.name == 'test' else 'development'})
        test_groups = {r['source_group'] for r in rows if r['split'] == 'test'}
        # The published archive repeats gunshot_000 across train/test under
        # different waveforms. Reserve the whole named group for test.
        rows = [r for r in rows if r['split'] == 'test' or r['source_group'] not in test_groups]
        indices = [i for i,r in enumerate(rows) if r['split'] == 'development']
        tr, va = train_test_split(indices, test_size=.2, random_state=505,
                                  stratify=[rows[i]['category'] for i in indices])
        for split, ids in [('train',tr),('val',va)]:
            for i in ids: rows[i]['split'] = split
    elif name == 'FSDD':
        for path in sorted((ROOT/'FSDD/free-spoken-digit-dataset-master/recordings').glob('*.wav')):
            speaker = path.stem.split('_')[1]
            split = 'test' if speaker == 'yweweler' else 'val' if speaker == 'theo' else 'train'
            rows.append({'path':str(path.resolve()),'label':0,'category':'spoken_digit',
                         'source_group':speaker,'split':split})
    elif name == 'EDGE':
        rows = json.loads((ROOT/'EDGE_manifest.json').read_text())
    elif name == 'UrbanSound8K':
        rows = json.loads((ROOT/'UrbanSound8K_manifest.json').read_text())
        priority = {'train':0,'val':1,'test':2}
        reserved = {}
        for row in rows:
            reserved[row['source_group']] = max(reserved.get(row['source_group'],0), priority[row['split']])
        removed = [r for r in rows if priority[r['split']] != reserved[r['source_group']]]
        g.write_json(ROOT/'UrbanSound8K_cross_fold_source_exclusions.json', removed)
        rows = [r for r in rows if priority[r['split']] == reserved[r['source_group']]]
    else: raise ValueError(name)
    for r in rows: r.update(domain=name,new_cycle=True)
    return rows


def prepare_dataset(name, cfg, checkpoint, output):
    rows = dataset_rows(name)
    excluded = []
    prepared = []
    cache = WORK / (name+'_features');cache.mkdir(exist_ok=True)
    seen = set(checkpoint['seen_hashes'])
    digests = {}
    for index,row in enumerate(rows):
        audio = g.read_clip(row['path'])
        digest = hashlib.sha256(audio.tobytes()).hexdigest()
        if digest in seen:
            excluded.append({**row,'reason':'Exact decoded overlap with previous train/validation'});continue
        if digest in digests:
            excluded.append({**row,'reason':'Duplicate decoded audio within new dataset','duplicate_of':digests[digest]});continue
        digests[digest] = row['path']
        path = cache / (hashlib.sha256(row['path'].encode()).hexdigest()+'.npy')
        if not path.exists(): np.save(path,feature(audio,cfg))
        prepared.append({**row,'decoded_sha256':digest,'feature_path':str(path.resolve())})
    g.write_json(output/'new_data_manifest.json', prepared)
    g.write_json(output/'overlap_exclusions.json', excluded)
    assert {r['split'] for r in prepared} == {'train','val','test'}
    groups = {s:{r['source_group'] for r in prepared if r['split']==s} for s in ['train','val','test']}
    assert not groups['train'] & (groups['val']|groups['test'])
    assert not groups['val'] & groups['test']
    print(name,'prepared',dict(Counter(r['split'] for r in prepared)),'excluded',len(excluded),flush=True)
    return prepared


def selection(y, p, rows, inherited, retention=None, baseline_floors=None):
    ds = np.array([r['domain'] for r in rows])
    # Floors fixed before any new test results are inspected.
    floors = {'original':(.95,.90), 'mad':(.85,.80), 'c3gd_augmented':(.75,.78)}
    if baseline_floors: floors = baseline_floors
    newname = next(r['domain'] for r in rows if r.get('new_cycle'))
    best = None
    for threshold in sorted(set(np.linspace(.03,.95,93).tolist()+[inherited])):
        by = {d:metric(y[ds==d],p[ds==d],threshold) for d in sorted(set(ds))}
        if not all(by[d]['gunshot_recall']>=rec and by[d]['gunshot_f1']>=f1 for d,(rec,f1) in floors.items()):continue
        if retention and any(by[d][key] < floor for d, values in retention.items()
                             for key, floor in values.items()):
            continue
        oldf1 = min(by[d]['gunshot_f1'] for d in floors)
        # Background replay groups pool ALL continuous windows into source files.
        source = defaultdict(list)
        for row,prob in zip(rows,p):
            if row['domain']=='new_background':source[row['path']].append(prob)
        rejection = np.mean([max(v)<=threshold for v in source.values()])
        new = by[newname]
        if new['gunshot_f1'] is not None:
            quality = .6*new['gunshot_f1']+.4*new['negative_rejection_rate']
        elif new['gunshot_recall'] is not None: quality=new['gunshot_recall']
        else: quality=new['negative_rejection_rate']
        score = float(quality+.3*oldf1+.2*rejection)
        if best is None or score>best[0]: best=(score,float(threshold),by,float(rejection))
    return best


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset',choices=['SESA','FSDD','EDGE','UrbanSound8K'],required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--distill',action='store_true',help='Freeze BN statistics and preserve inherited soft predictions during updates.')
    p.add_argument('--learning-rate',type=float,default=5e-5)
    p.add_argument('--adaptive-retention',action='store_true',help='Retain prior validation metrics within five percentage points of the inherited model.')
    p.add_argument('--epoch-draws',type=int)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Preserve previous runs; choose a new folder.')
    WORK.mkdir(parents=True,exist_ok=True)
    experiment_lock=(WORK/'training.lock').open('w')
    try:fcntl.flock(experiment_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another cycle is running. Wait for it before starting the next cycle.')
    a.out.mkdir(parents=True)
    torch.set_num_threads(2);torch.manual_seed(505);np.random.seed(505);random.seed(505)
    ckpt=torch.load(a.previous,map_location='cpu',weights_only=True)
    checksum=fingerprint(a.previous);cfg=dict(ckpt['config'])
    cfg.update(initialization=str(a.previous),initialization_sha256=checksum,learning_rate=a.learning_rate,seed=505,
               selection='Fixed old-domain recall/F1 floors; validation new-domain quality and continuous-background rejection.')
    cfg['training_datasets'] = sorted(set(cfg.get('training_datasets', [])) | {a.dataset})
    new=prepare_dataset(a.dataset,cfg,ckpt,a.out)
    net=g.model_for(cfg);net.load_state_dict(ckpt['state_dict'])
    before=reports(net,new,cfg['threshold'],a.out/'before')
    replay=prepare_replay(cfg)
    # Prior cycle training and validation examples replay only in the same split.
    previous_cycles=[]
    prev_manifest=a.previous.parent/'cumulative_new_data.json'
    if prev_manifest.exists():previous_cycles=json.loads(prev_manifest.read_text())
    cfg['training_datasets'] = sorted(set(cfg['training_datasets']) | {r['domain'] for r in previous_cycles})
    splits={s:[{**r,'new_cycle':False} for r in replay[s]]+
               [{**r,'new_cycle':False} for r in previous_cycles if r['split']==s]+
               [r for r in new if r['split']==s] for s in ['train','val']}
    from training_data_policy import assert_development_allowed
    assert_development_allowed(splits['train'] + splits['val'])
    g.write_json(a.out/'training_manifest.json',splits)
    g.write_json(a.out/'cumulative_new_data.json',previous_cycles+new)
    protocol={'dataset':a.dataset,'previous_checkpoint':str(a.previous),'previous_sha256':checksum,
              'initial_test_completed_before_training':True,'seed':505,'learning_rate':a.learning_rate,
              'knowledge_distillation':a.distill,'batchnorm_statistics_frozen':a.distill,
              'adaptive_retention_drop':.05 if a.adaptive_retention else None,
              'sampled_training_recordings_per_epoch':min(len(splits['train']),a.epoch_draws) if a.epoch_draws else len(splits['train']),
              'test_used_for_optimization_or_selection':False,'shotspotter_used_for_optimization_or_selection':False,
              'selection_floors':{'original':{'recall':.95,'F1':.90},'MAD':{'recall':.85,'F1':.80},'C3GD_replay':{'recall':.75,'F1':.78}},
              'selection':'new-validation quality + .3 minimum replay F1 + .2 continuous background-file rejection',
              'source_split_note': 'SESA has no original Freesound source IDs. Its official test split is preserved; train/val are file-disjoint, but shared-source or near-duplicate independence is not established.' if a.dataset=='SESA' else 'Groups preserved in new_data_manifest.json.',
              'dataset_source':{'SESA':'https://zenodo.org/records/3519845','FSDD':'https://github.com/Jakobovski/free-spoken-digit-dataset','EDGE':'https://zenodo.org/records/7004819','UrbanSound8K':'https://urbansounddataset.weebly.com/urbansound8k.html'}[a.dataset],
              'train_counts':dict(Counter(r['domain'] for r in splits['train'])),
              'val_counts':dict(Counter(r['domain'] for r in splits['val']))}
    g.write_json(a.out/'protocol.json',protocol)
    optimizer=torch.optim.AdamW(net.parameters(),lr=a.learning_rate,weight_decay=.01)
    teacher=None
    if a.distill:
        teacher=g.model_for(cfg);teacher.load_state_dict(ckpt['state_dict']);teacher.eval()
        for parameter in teacher.parameters():parameter.requires_grad_(False)
    history=[];best=-float('inf');stale=0;retention={};baseline_floors=None
    train_loader=loader(splits['train'],True,a.epoch_draws);val_loader=loader(splits['val'])
    for epoch in range(a.epochs+1):
        loss=None
        if epoch:
            if not a.distill:
                loss=g.run_epoch(net,train_loader,torch.device('cpu'),optimizer)[0]
            else:
                net.train()
                for module in net.modules():
                    if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
                total=0.
                for bags,labels in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits=g.recording_logits(net,bags,torch.device('cpu'))
                    with torch.no_grad():soft=g.recording_logits(teacher,bags,torch.device('cpu')).div(2).sigmoid()
                    objective=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
                    objective+=.5*4*torch.nn.functional.binary_cross_entropy_with_logits(logits/2,soft)
                    objective.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
                    total+=objective.item()*len(labels)
                loss=total/len(splits['train'])
        val_loss,y,prob=g.run_epoch(net,val_loader,torch.device('cpu'))
        if epoch == 0:
            domains = np.array([r['domain'] for r in splits['val']])
            for domain in {r['domain'] for r in previous_cycles}:
                baseline = metric(y[domains == domain], prob[domains == domain], ckpt['config']['threshold'])
                retention[domain] = {key: max(0., baseline[key] - .03)
                                     for key in ['gunshot_recall', 'gunshot_f1', 'negative_rejection_rate']
                                     if baseline[key] is not None}
            if a.adaptive_retention:
                minimum={'original':(.95,.90),'mad':(.80,.78),'c3gd_augmented':(.75,.78)}
                baseline_floors={}
                for domain in set(domains)-{a.dataset,'new_background'}:
                    baseline=metric(y[domains==domain],prob[domains==domain],ckpt['config']['threshold'])
                    retention[domain]={key:max(0.,baseline[key]-.05)
                                       for key in ['gunshot_recall','gunshot_f1','negative_rejection_rate']
                                       if baseline[key] is not None}
                    if domain in minimum:
                        baseline_floors[domain]=tuple(max(v,retention[domain][k]) for v,k in zip(minimum[domain],['gunshot_recall','gunshot_f1']))
                protocol['selection_floors']=baseline_floors
            protocol['prior_cycle_validation_retention_floors'] = retention
            g.write_json(a.out/'protocol.json', protocol)
        selected=selection(y,prob,splits['val'],ckpt['config']['threshold'],retention,baseline_floors)
        row={'epoch':epoch,'train_loss':loss,'validation_loss':val_loss,'eligible':selected is not None}
        if selected is None:
            row['validation_at_inherited_threshold']={d:metric(y[np.array([r['domain']==d for r in splits['val']])],prob[np.array([r['domain']==d for r in splits['val']])],ckpt['config']['threshold']) for d in sorted({r['domain'] for r in splits['val']})}
        if selected:
            score,threshold,by,rejection=selected
            row.update(score=score,threshold=threshold,new_validation=by[a.dataset],background_file_rejection=rejection)
            if score>best:
                best=score;stale=0;cfg['threshold']=threshold
                hashes={r['decoded_sha256'] for r in previous_cycles+new if r['split']!='test'}
                torch.save({**ckpt,'state_dict':net.state_dict(),'config':dict(cfg),'epoch':epoch,
                            'seen_hashes':sorted(set(ckpt['seen_hashes'])|hashes),
                            'seen_sources':sorted(set(ckpt['seen_sources'])|{r['source_group'] for r in new if r['split']!='test'})},a.out/'best.pt')
                g.write_json(a.out/'selection.json',row);g.write_json(a.out/'validation_by_domain.json',by)
            else:stale+=1
        else:stale+=1
        history.append(row);g.write_json(a.out/'history.json',history)
        print(json.dumps(row),flush=True)
        if epoch>= (6 if a.distill else 3) and stale>= (4 if a.distill else 3):break
    if not (a.out/'best.pt').exists():raise RuntimeError('No checkpoint met retention floors; previous checkpoint retained.')
    final=torch.load(a.out/'best.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(final['state_dict'])
    after=reports(net,new,final['config']['threshold'],a.out/'after')
    assert fingerprint(a.previous)==checksum
    g.write_json(a.out/'cycle_results.json',{'dataset':a.dataset,'before':before,'after':after,
                 'previous_checkpoint_unchanged':True,'selected_epoch':final['epoch'],
                 'trained_weight_changes_adopted': any(not torch.equal(value,ckpt['state_dict'][key]) for key,value in final['state_dict'].items()),
                 'checkpoint_sha256':fingerprint(a.out/'best.pt'),
                 'after_all_data_status':'Includes training/validation. Use after.test for held-out metrics.'})
    print('CYCLE COMPLETE',a.out,flush=True)


if __name__=='__main__':main()
