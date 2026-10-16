"""Frozen final-model evaluation on held-out portions of the new corpora."""
import json
from pathlib import Path
import numpy as np
import torch
import gunshot as g
from continual_cycles import fingerprint, infer, metric


def main():
    checkpoint=Path('runs/run8_urban/best.pt')
    checksum=fingerprint(checkpoint)
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net=g.model_for(ckpt['config']);net.load_state_dict(ckpt['state_dict'])
    torch.set_num_threads(2)
    output=Path('runs/run8_urban/final_new_datasets.json')
    if output.exists():raise ValueError('Preserve frozen results.')
    results=[]
    for folder,name in [('run5_sesa_valid','SESA'),('run6_fsdd','FSDD'),('run7_edge_distilled','EDGE'),('run8_urban','UrbanSound8K')]:
        rows=[r for r in json.loads((Path('runs')/folder/'new_data_manifest.json').read_text()) if r['split']=='test']
        assert not {r['decoded_sha256'] for r in rows}&set(ckpt['seen_hashes'])
        assert not {r['source_group'] for r in rows}&set(ckpt['seen_sources'])
        y,p=infer(net,rows)
        report=metric(y,p,ckpt['config']['threshold'])
        results.append({'dataset':name,'metrics':report})
        np.savez(Path('runs/run8_urban')/('final_'+name+'_heldout_predictions.npz'),
                 true_label=y,probability=p,paths=np.array([r['path'] for r in rows]))
        print(name,json.dumps(report),flush=True)
    assert fingerprint(checkpoint)==checksum
    g.write_json(output,{'checkpoint_sha256':checksum,'threshold':ckpt['config']['threshold'],
                        'training_or_selection_performed':False,'results':results})

if __name__=='__main__':main()
