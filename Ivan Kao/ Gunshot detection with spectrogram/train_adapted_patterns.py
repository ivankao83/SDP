"""Fine-tune a copy of the existing CNN; retain the full old system as a frozen teacher."""
import argparse,copy,fcntl,json,random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
import gunshot as g
from continual_cycles import fingerprint
from predict_impulse_refined import load_bundle
from model_feature_matrix import matrix
from training_data_policy import assert_development_allowed
from run_five_new_cycles import check_splits


class Replay(Dataset):
    def __init__(self,rows,teacher):self.rows=rows;self.teacher=teacher
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return torch.from_numpy(np.load(self.rows[i]['feature_path']).astype(np.float32)),float(self.rows[i]['label']),float(self.teacher[i])


def collate(batch):
    x,y,p=zip(*batch);return list(x),torch.tensor(y,dtype=torch.float32),torch.tensor(p,dtype=torch.float32)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--previous',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--epochs',type=int,default=8);p.add_argument('--draws',type=int,default=4000)
    a=p.parse_args();out=Path('runs')/a.run;report=Path('reports')/a.run;out.mkdir(exist_ok=False);report.mkdir(exist_ok=False)
    lock=Path('runs/continual_cycles/training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.manual_seed(4444);np.random.seed(4444);random.seed(4444)
    config,cfg,teacher_net,expert=load_bundle(a.previous);original_sha=fingerprint(Path(config['cnn_checkpoint']))
    saved=json.loads((a.previous.parent/'training_manifest.json').read_text());train,val=saved['train'],saved['val'];check_splits(train+val);assert_development_allowed(train+val)
    reserve=json.loads(Path('dataset/test_only_reserve/manifest.json').read_text());assert not {r['decoded_sha256'] for r in reserve}&{r['decoded_sha256'] for r in train+val}
    g.write_json(out/'training_manifest.json',saved);g.write_json(out/'new_data_manifest.json',json.loads((a.previous.parent/'new_data_manifest.json').read_text()))
    xt,yt=matrix(teacher_net,train,cfg,config);teacher=expert.predict_proba(xt[:,:expert.n_features_in_])[:,1]
    counts=Counter((r['domain'],r['label']) for r in train)
    def importance(d):return 3. if d in ['IndependentBagPops','IndependentBalloonCollection','IndependentPackaging'] else 2. if d in ['IndependentFieldMixed','AydemirGunshots','IndependentBalloonPops','LabImpulses'] else 1.
    mass={l:sum(importance(d) for d,k in counts if k==l) for l in [0,1]}
    weights=[importance(r['domain'])/(mass[r['label']]*counts[r['domain'],r['label']]) for r in train]
    batches=DataLoader(Replay(train,teacher),batch_size=16,sampler=WeightedRandomSampler(weights,a.draws,replacement=True),collate_fn=collate)
    validation=DataLoader(Replay(val,np.zeros(len(val))),batch_size=16,collate_fn=collate)
    vcounts=Counter((r['domain'],r['label']) for r in val)
    vweight=np.array([min(vcounts[r['domain'],r['label']]/30,1)/vcounts[r['domain'],r['label']] for r in val]);vweight/=vweight.sum()
    student=copy.deepcopy(teacher_net);optimizer=torch.optim.Adam(student.parameters(),lr=3e-5,weight_decay=1e-4)
    protocol={'previous_bundle':str(a.previous),'initialization':config['cnn_checkpoint'],'initialization_sha256':original_sha,'cnn_reset':False,'frozen_previous_full_system_teacher':True,
        'training_rows':len(train),'validation_rows':len(val),'epochs':a.epochs,'draws_per_epoch':a.draws,'learning_rate':3e-5,'seed':4444,
        'loss':'0.85 labelled BCE + 0.15 frozen earlier full-system probability BCE; balanced domain/class replay.',
        'selection':'Lowest support-shrunk macro-domain/class validation log loss; validation only, best trained epoch.',
        'baseline_used_for_training_validation_or_epoch_selection':False,'baseline_not_evaluated_by_this_script':True}
    g.write_json(report/'protocol.json',protocol);best=float('inf');history=[]
    for epoch in range(1,a.epochs+1):
        student.train();losses=[]
        for x,y,prior in batches:
            optimizer.zero_grad(set_to_none=True);logits=g.recording_logits(student,x,torch.device('cpu'))
            loss=.85*torch.nn.functional.binary_cross_entropy_with_logits(logits,y)+.15*torch.nn.functional.binary_cross_entropy_with_logits(logits,prior)
            loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
        student.eval();probability=[]
        with torch.inference_mode():
            for x,y,prior in validation:probability.extend(g.recording_logits(student,x,torch.device('cpu')).sigmoid().numpy().tolist())
        q=np.clip(np.asarray(probability),1e-6,1-1e-6);yv=np.array([r['label'] for r in val]);score=float((-yv*np.log(q)-(1-yv)*np.log1p(-q))@vweight)
        row={'epoch':epoch,'training_loss':float(np.mean(losses)),'validation_macro_log_loss':score};history.append(row)
        if score<best:
            best=score;new_cfg={**cfg,'threshold':.5,'initialization':config['cnn_checkpoint'],'initialization_sha256':original_sha,'teacher_bundle':str(a.previous),'adaptation_epochs':a.epochs,'adaptation_learning_rate':3e-5,'training_datasets':sorted({r['domain'] for r in train})}
            torch.save({'config':new_cfg,'state_dict':student.state_dict(),'epoch':epoch,'validation_macro_log_loss':score,'seen_hashes':sorted({r['decoded_sha256'] for r in train+val})},out/'best.pt')
        print('ADAPTED CNN',json.dumps(row),'best',best,flush=True);g.write_json(report/'history.json',history)
    assert fingerprint(Path(config['cnn_checkpoint']))==original_sha
    ckpt=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    g.write_json(report/'selection.json',{'epoch':ckpt['epoch'],'validation_macro_log_loss':ckpt['validation_macro_log_loss'],'checkpoint':str(out/'best.pt'),'sha256':fingerprint(out/'best.pt'),'previous_system_unchanged':True,'selected_before_baseline_prediction':True})
    print('ADAPTED PATTERNS READY',str(out/'best.pt'),flush=True)


if __name__=='__main__':main()
