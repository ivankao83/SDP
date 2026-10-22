"""One new-dataset continuation followed by a frozen, test-only baseline check."""
import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

import torch
import gunshot as g
import run_large_negative_cycles as cycle
from continual_cycles import fingerprint


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--dataset',required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--split-policy',required=True)
    parser.add_argument('--strict-recall',action='store_true')
    parser.add_argument('--fine-stages',action='store_true')
    args=parser.parse_args()
    lock=Path('runs/continual_cycles/training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    previous=json.loads(args.previous.read_text())
    saved=json.loads((args.previous.parent/'training_manifest.json').read_text())
    history=saved['train']+saved['val']
    reserve=json.loads(Path('dataset/test_only_reserve/manifest.json').read_text())
    blocked=set(previous['seen_hashes'])|{r['decoded_sha256'] for r in reserve}
    for path in Path('runs').glob('*/new_data_manifest.json'):
        blocked|={r['decoded_sha256'] for r in json.loads(path.read_text())}
    cycle.ROOT=args.root;cycle.SPLIT_POLICY=args.split_policy
    if args.strict_recall:cycle.POSITIVE_RETENTION_TOLERANCE=0.
    cycle.FINE_STAGES=args.fine_stages
    # Reuse fixed-CNN vectors without recomputing the historical waveform/CNN features.
    cache=args.root/'vectors'
    if not cache.exists():cache.symlink_to((Path('dataset/run19_20_large_negatives/vectors')).resolve(),target_is_directory=True)
    completed=Path('reports')/args.run/'results.json'
    if completed.exists():
        result=json.loads(completed.read_text());assert result['previous_refiner_sha256']==fingerprint(Path(previous['refiner']))
        bundle=Path(result['bundle'])
    else:
        bundle,_,result=cycle.train_cycle(args.run,args.dataset,args.previous,history,blocked)
    external=Path('reports')/args.run/'external_test'
    if not (external/'results.json').exists():
        subprocess.run([sys.executable,'evaluate_test_only_reserves.py','--bundle',str(bundle),'--out',str(external)],check=True)
    tested=json.loads((external/'results.json').read_text())
    baseline=json.loads(Path('reports/run20_vocal_sound/external_test/results.json').read_text())
    assert tested['manifest_sha256']==baseline['manifest_sha256']
    a=baseline['combined'];b=tested['combined']
    target=b['accuracy']>a['accuracy'] and b['gunshot_recall']>=a['gunshot_recall'] and b['false_alarms']<a['false_alarms']
    comparison={'run':args.run,'baseline_run':'run20_vocal_sound','before':a,'after':b,
                'target_met':target,'target':'Accuracy strictly higher than Run20; recall no lower and false alarms fewer.',
                'checkpoint_selected_before_baseline_evaluation':True,
                'baseline_used_for_stopping_further_cycles':True,
                'limitation':'Repeated baseline evaluations guide stopping and constitute adaptive benchmark reuse, not an unbiased unseen test estimate.'}
    g.write_json(completed.parent/'baseline_comparison.json',comparison)
    updated=json.loads(bundle.read_text())
    g.write_json(Path('reports/latest_model.json'),{'run':args.run,'model_type':'cnn_with_impulse_refiner',
                 'bundle':str(bundle),'checkpoint':updated['cnn_checkpoint'],'sha256':updated['cnn_sha256'],
                 'refiner':updated['refiner'],'refiner_sha256':updated['refiner_sha256'],
                 'threshold':updated['threshold'],'report':str(completed.parent/'baseline_comparison.json')})
    print('CYCLE COMPLETE',json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':main()
