"""Select a compact public gunshot subset; whole firearm model groups stay apart."""
import hashlib,io,json,re,zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import gunshot as g
from continual_cycles import feature,fingerprint
from prepare_five_cycles import read_url
from prepare_joint_impulses import BufferedZip
ROOT=Path('dataset/run42_aydemir');SIZE=398678732
URL='https://www.kaggle.com/api/v1/datasets/download/emrahaydemr/gunshot-audio-dataset?datasetVersionNumber=1'
SPLITS={'AK-12':'train','AK-47':'train','IMI Desert Eagle':'val','M16':'train','M4':'train','M249':'train','MG-42':'test','MP5':'train','Zastava M92':'test'}


def main():
    torch.set_num_threads(2);ROOT.mkdir(exist_ok=True);folder=ROOT/'AydemirGunshots';folder.mkdir(exist_ok=True);cache=folder/'features';cache.mkdir(exist_ok=True)
    inventory=json.loads(Path('dataset/source_search/20261022/kaggle_zip_inventory.json').read_text())
    metadata=json.loads(Path('dataset/source_search/20261022/kaggle_metadata.json').read_text())
    cfg=torch.load('runs/run14_m16_firecrackers/best.pt',map_location='cpu',weights_only=True)['config']
    plans={model:sorted([r for r in inventory if r['name'].split('/')[0]==model],key=lambda r:r['name'])[:40] for model in SPLITS}
    (ROOT/'prespecified_selection.json').write_text(json.dumps({'version':1,'split_by_model':SPLITS,'clips_per_folder':40,'selected':plans,'model_alias_group':'M16 and M4 share AR15_family, both training.'},indent=2))
    def one(model):
        saved=ROOT/(re.sub(r'[^A-Za-z0-9]+','_',model)+'_manifest.json')
        if saved.exists():return json.loads(saved.read_text())
        selected=plans[model];assert len(selected)==40
        start=min(r['offset'] for r in selected);end=max(r['offset']+30+len(r['name'].encode())+r['compressed']+1024 for r in selected);end=min(end,SIZE)
        data=[];audit=[]
        for left in range(start,end,8_000_000):
            right=min(end-1,left+7_999_999);raw=read_url(URL,left,right,limit=8_000_000);assert len(raw)==right-left+1
            data.append(raw);audit.append({'offset':left,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        remote=BufferedZip(URL,SIZE);remote.blocks=[(start,b''.join(data))]
        rows=[]
        with zipfile.ZipFile(remote) as archive:
            for spec in selected:
                raw=archive.read(spec['name']);assert len(raw)==spec['size']
                path=folder/(re.sub(r'[^A-Za-z0-9]+','_',model)+'__'+Path(spec['name']).name)
                # Convert the complete author segment to compact mono 24 kHz FLOAT audio.
                audio,sr=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True);audio=audio.mean(axis=1)
                if sr!=g.SR:audio=g.librosa.resample(audio,orig_sr=sr,target_sr=g.SR)
                audio=np.asarray(audio,dtype=np.float32);assert np.isfinite(audio).all() and len(audio)>0
                sf.write(path,audio,g.SR,subtype='FLOAT');digest=hashlib.sha256(audio.tobytes()).hexdigest();fp=cache/(digest+'.npy')
                if not fp.exists():np.save(fp,feature(audio,cfg))
                group='AR15_family' if model in ['M4','M16'] else model
                rows.append({'path':str(path.resolve()),'domain':'AydemirGunshots','label':1,'split':SPLITS[model],'source_group':'firearm_model:'+group,
                    'firearm_model':model,'original_member':spec['name'],'original_crc32':spec['crc'],'original_member_sha256':hashlib.sha256(raw).hexdigest(),
                    'source_url':'https://www.kaggle.com/datasets/emrahaydemr/gunshot-audio-dataset','source_version':1,'original_sample_rate':sr,
                    'duration_seconds':len(audio)/g.SR,'decoded_sha256':digest,'sha256':fingerprint(path),'feature_path':str(fp.resolve()),
                    'license_metadata':metadata['licenseName'],'label_basis':'Author reports manually reviewed real gunshots from public videos; original video IDs unavailable.'})
        (ROOT/(re.sub(r'[^A-Za-z0-9]+','_',model)+'_ranges.json')).write_text(json.dumps(audit,indent=2));saved.write_text(json.dumps(rows,indent=2));print('READY FIREARM MODEL',model,len(rows),SPLITS[model],flush=True);return rows
    with ThreadPoolExecutor(max_workers=2) as pool:rows=[r for batch in pool.map(one,SPLITS) for r in batch]
    unique={};duplicates=[]
    for r in rows:
        if r['decoded_sha256'] in unique:duplicates.append(r);continue
        unique[r['decoded_sha256']]=r
    rows=list(unique.values());assert len(rows)>=200
    (folder/'manifest.json').write_text(json.dumps(rows,indent=2));(ROOT/'duplicates_excluded.json').write_text(json.dumps(duplicates,indent=2))
    (folder/'provenance.json').write_text(json.dumps({'source_url':'https://www.kaggle.com/datasets/emrahaydemr/gunshot-audio-dataset','source_metadata':metadata,
        'version':1,'archive_bytes':SIZE,'whole_archive_downloaded':False,'selected_count':len(rows),'selection':'First 40 original filenames per folder; determined before predictions.',
        'integrity':'Every selected original ZIP member CRC32 verified by ZipFile.read, original SHA256 retained, then complete clip resampled to mono24k FLOAT.',
        'split':'Entire firearm model folders; M4 and M16 grouped as AR15 family and both training.',
        'citation':'Tuncer, Dogan, Akbal, Aydemir (2021), An Automated Gunshot Audio Classification Method Based On Finger Pattern Feature Generator And Iterative Relieff Feature Selector.',
        'limitations':'Original video IDs unavailable; splitting whole model groups cannot prove full video independence across models or cross-corpus lineage. Kaggle licence field is Unknown; source audio is kept local, not included in Git.',
        'counts_by_split':dict(Counter(r['split'] for r in rows))},indent=2))
    print('READY AYDEMIR',len(rows),dict(Counter(r['split'] for r in rows)),flush=True)


if __name__=='__main__':main()
