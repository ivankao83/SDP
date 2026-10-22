"""Use this one-source field collection for training only after a pretest."""
import hashlib
import json
import zipfile
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import gunshot as g
from continual_cycles import feature,fingerprint


def main():
    root=Path('dataset/run28_albufera');folder=root/'AlbuferaField';folder.mkdir(parents=True,exist_ok=True)
    archive=Path('dataset/run25_albufera/Audio_test.zip')
    ckpt=torch.load('runs/run14_m16_firecrackers/best.pt',map_location='cpu',weights_only=True)
    cache=folder/'features';cache.mkdir(exist_ok=True);rows=[]
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if not name.endswith('.wav') or name.startswith('__MACOSX'):continue
            path=folder/(Path(name).parent.name+'__'+Path(name).name)
            if not path.exists():path.write_bytes(z.read(name))
            audio=g.read_clip(path);digest=hashlib.sha256(audio.tobytes()).hexdigest();fp=cache/(digest+'.npy')
            if not fp.exists():np.save(fp,feature(audio,ckpt['config']))
            rows.append({'path':str(path.resolve()),'domain':'AlbuferaField','label':int('/shooting/' in name),
                         'split':'train','source_group':'recording:SMA07394_20221113_083002',
                         'decoded_sha256':digest,'feature_path':str(fp.resolve()),'sha256':fingerprint(path),
                         'source_url':'https://github.com/gpinyero/EUSIPCO2023','original_member':name,
                         'role':'training_after_pretest','limitation':'Single source recording; after-training retest is in-sample and not an independent test.'})
    assert len(rows)>=200 and len({r['decoded_sha256'] for r in rows})==len(rows)
    g.write_json(folder/'manifest.json',rows)
    g.write_json(folder/'provenance.json',{'source':'https://github.com/gpinyero/EUSIPCO2023',
                 'original_archive':str(archive),'archive_sha256':fingerprint(archive),'original_recordings':1,
                 'use':'Pretest all clips, then training only; validation comes from other corpora. Seen retest reported separately.',
                 'labels':'Author shooting versus background directories; not author model TP/FP outcome folders.'})
    print('READY ALBUFERA',len(rows),Counter(r['label'] for r in rows),flush=True)


if __name__=='__main__':main()
