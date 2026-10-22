"""Compact MIMII industrial-valve negatives, with machine IDs held apart."""
import io
import json
import hashlib
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

import numpy as np
import soundfile as sf
from prepare_five_cycles import read_url
from prepare_joint_impulses import BufferedZip

ROOT=Path('dataset/run23_mimii')


def main():
    folder=ROOT/'MIMIIValve';folder.mkdir(exist_ok=True)
    if (folder/'raw_manifest.json').exists():return
    metadata=json.loads((ROOT/'source_metadata.json').read_text())
    source=next(f for f in metadata['files'] if f['key']=='0_dB_valve.zip')
    inventory=json.loads((ROOT/'inventory.json').read_text())
    plan=[('id_00','normal','train'),('id_02','normal','train'),('id_04','normal','val'),
          ('id_06','normal','test'),('id_06','abnormal','test')]
    rows=[];audit=[]
    for machine,state,split in plan:
        selected=sorted([r for r in inventory if f'/{machine}/{state}/' in r['name'] and r['name'].endswith('.wav')],key=lambda r:r['name'])[:100]
        assert len(selected)==100
        start=min(r['offset'] for r in selected);end=max(r['offset']+r['compressed']+len(r['name'].encode())+512 for r in selected)
        def chunk(left):
            right=min(end-1,left+7_999_999)
            data=read_url(source['links']['self']+f'?sdp23_range={left}',left,right,limit=8_000_000)
            assert len(data)==right-left+1
            print('VERIFIED MIMII RANGE',left,len(data),flush=True)
            return data
        with ThreadPoolExecutor(max_workers=3) as pool:data=b''.join(pool.map(chunk,range(start,end,8_000_000)))
        audit.append({'machine':machine,'state':state,'offset':start,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        remote=BufferedZip(source['links']['self']+f'?sdp23_tail={machine}_{state}',source['size']);remote.blocks=[(start,data)]
        with zipfile.ZipFile(remote) as archive:
            for r in selected:
                path=folder/(machine+'_'+state+'_'+Path(r['name']).name)
                if not path.exists():
                    audio,sr=sf.read(io.BytesIO(archive.read(r['name'])),dtype='float32',always_2d=True)
                    audio=audio[:,0];peak=int(np.abs(audio).argmax());left=max(0,min(peak-sr,len(audio)-3*sr))
                    sf.write(path,audio[left:left+3*sr],sr,subtype='FLOAT')
                rows.append({'path':str(path.resolve()),'domain':'MIMIIValve','label':0,'split':split,
                             'source_group':'machine:'+machine,'machine_id':machine,'machine_state':state,
                             'original_member':r['name'],'original_crc32':r['crc'],
                             'source_url':'https://zenodo.org/records/3384388',
                             'preprocessing':'Microphone channel 1; up to three seconds around strongest absolute sample.'})
        print('READY MIMII GROUP',machine,state,len(selected),flush=True)
    (folder/'raw_manifest.json').write_text(json.dumps(rows,indent=2))
    (folder/'provenance.json').write_text(json.dumps({'source':source,'plan':plan,'byte_ranges':audit,
        'selection':'First 100 original recording filenames per prespecified machine/state group, before inference.',
        'label_basis':'Normal and abnormal industrial valves are both non-gunshot sounds; malfunction label is not gunshot label.',
        'split':'Whole machine IDs; single SNR version avoids replicated SNR variants.',
        'integrity':'ZIP CRC32 checked by archive.read for every complete original multichannel WAV before excerpting.'},indent=2))
    print('READY MIMII',len(rows),Counter(r['split'] for r in rows),flush=True)


if __name__=='__main__':main()
