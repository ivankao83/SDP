"""Collect new author-described firearm and non-firearm recordings from public previews."""
import hashlib,json,random,re
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import gunshot as g
from continual_cycles import feature,fingerprint
from prepare_independent_pops import get
ROOT=Path('dataset/run42_independent_firearms');FOLDER=ROOT/'IndependentFieldMixed'


def main():
    torch.set_num_threads(2);FOLDER.mkdir(exist_ok=True);meta=ROOT/'source_metadata';meta.mkdir(exist_ok=True);cache=FOLDER/'features';cache.mkdir(exist_ok=True)
    cfg=torch.load('runs/run14_m16_firecrackers/best.pt',map_location='cpu',weights_only=True)['config']
    candidates=json.loads((ROOT/'filtered_candidates.json').read_text());rng=random.Random(4242)
    removed={'408727','541164','653542','328073','595494'};bad_authors={'hz37','filmmakersmanual'}
    positives=[r for r in candidates if r['label']==1 and r['id'] not in removed and r['author'].casefold() not in bad_authors]
    negatives=[r for r in candidates if r['label']==0];rng.shuffle(positives);rng.shuffle(negatives)
    count=Counter();ordered=[]
    for r in positives+negatives:
        author=r['author'].casefold()
        if count[author]>=8:continue
        count[author]+=1;ordered.append(r)
    (ROOT/'prespecified_order.json').write_text(json.dumps([r['id'] for r in ordered],indent=2))
    def one(r):
        identifier=r['id'];record=meta/(identifier+'.json')
        if record.exists():
            saved=json.loads(record.read_text())
            if Path(saved['path']).exists() and Path(saved['feature_path']).exists():return saved
        try:
            page='https://freesound.org'+r['href'];html_path=meta/(identifier+'.html')
            if not html_path.exists():
                previous=Path('dataset/run38_independent_pops/source_metadata')/(identifier+'.html')
                html_path.write_bytes(previous.read_bytes() if previous.exists() else get(page,2_000_000))
            html=html_path.read_text();licenses=set(re.findall(r'https?://creativecommons.org/[^\"<>\s]+',html))
            license=next((s for s in licenses if '/publicdomain/zero/' in s or re.search(r'/licenses/by/\d',s)),None)
            if not license or len(licenses)!=1:raise ValueError('Single CC0 or attribution-only licence not verified')
            urls=re.findall(r'https://cdn\.freesound\.org/previews/[^\"\s<>]+-hq\.mp3',html)
            if not urls:raise ValueError('Public HQ preview unavailable')
            preview=get(urls[0],5_000_000);temporary=FOLDER/(identifier+'.download.mp3');temporary.write_bytes(preview)
            try:audio=g.read_clip(temporary)
            finally:temporary.unlink(missing_ok=True)
            start=0
            if r['label']==0:
                peak=int(np.abs(audio).argmax());start=max(0,min(peak-g.SR,len(audio)-g.SAMPLES));audio=audio[start:start+g.SAMPLES]
            # Keep complete positive recordings so a peak crop cannot remove quiet shots.
            assert len(audio)<=60*g.SR
            path=FOLDER/(identifier+'.wav');sf.write(path,audio,g.SR,subtype='FLOAT');digest=hashlib.sha256(audio.tobytes()).hexdigest();fp=cache/(digest+'.npy')
            if not fp.exists():np.save(fp,feature(audio,cfg))
            row={'path':str(path.resolve()),'domain':'IndependentFieldMixed','label':r['label'],'source_group':'uploader:'+r['author'].casefold(),
                'uploader':r['author'],'freesound_id':identifier,'category':r['category'],'source_url':page,'preview_url':urls[0],'license':license,
                'original_description':r['description'],'original_title':r['file_name'],'preview_sha256':hashlib.sha256(preview).hexdigest(),
                'excerpt_start_seconds':start/g.SR,'duration_seconds':len(audio)/g.SR,'decoded_sha256':digest,'sha256':fingerprint(path),'feature_path':str(fp.resolve()),
                'label_basis':'Original author description. Synthetic/ambiguous gun-like effects excluded. No generated captions used.'}
            record.write_text(json.dumps(row,indent=2));return row
        except Exception as error:return {'excluded_id':identifier,'reason':type(error).__name__+': '+str(error)}
    rows=[];excluded=[]
    for offset in range(0,len(ordered),40):
        with ThreadPoolExecutor(max_workers=2) as pool:batch=list(pool.map(one,ordered[offset:offset+40]))
        for item in batch:
            if 'excluded_id' in item:excluded.append(item)
            else:rows.append(item)
        print('COLLECTED',len(rows),'labels',dict(Counter(r['label'] for r in rows)),'excluded',len(excluded),flush=True)
        (ROOT/'collection_progress.json').write_text(json.dumps({'downloaded':rows,'excluded':excluded},indent=2))
        if len(rows)>=300:break
    assert len(rows)>=200 and sum(r['label']==1 for r in rows)>=25
    by_author=defaultdict(list)
    for r in rows:by_author[r['uploader'].casefold()].append(r)
    assignments={}
    for label in [0,1]:
        authors=sorted(a for a,items in by_author.items() if max(r['label'] for r in items)==label);rng.shuffle(authors)
        a=int(.6*len(authors));b=int(.8*len(authors))
        assignments.update({author:'train' if i<a else 'val' if i<b else 'test' for i,author in enumerate(authors)})
    seen=set();clean=[]
    for r in rows:
        if r['decoded_sha256'] in seen:excluded.append({'excluded_id':r['freesound_id'],'reason':'Decoded duplicate'});continue
        seen.add(r['decoded_sha256']);clean.append({**r,'split':assignments[r['uploader'].casefold()]})
    assert len(clean)>=200
    for s in ['train','val','test']:
        assert sum(r['split']==s for r in clean)>=20
        assert sum(r['split']==s and r['label']==1 for r in clean)>=3
    (FOLDER/'manifest.json').write_text(json.dumps(clean,indent=2));(ROOT/'excluded.json').write_text(json.dumps(excluded,indent=2))
    (ROOT/'ATTRIBUTIONS.md').write_text('# Original audio attribution\n\n'+'\n'.join(f"- {r['freesound_id']}: {r['uploader']}; {r['source_url']}; {r['license']}" for r in clean)+'\n')
    (FOLDER/'provenance.json').write_text(json.dumps({'audio':'Original public Freesound high-quality previews; author CC0/BY licences verified.',
        'discovery_index_provenance':json.loads(Path('dataset/run38_independent_pops/index_provenance.json').read_text()),
        'collection_is_curated_subset_not_new_published_dataset':True,'labels':'Original author descriptions filtered to remove synthetic gunshots, firearm mechanisms and ambiguous effects.',
        'original_positive_recordings_preserved_complete':True,'negative_peak_excerpts_seconds':3,
        'source_split':'Whole case-normalized uploaders stratified by label, 60/20/20, seed4242, before any predictions.',
        'overlap':'Known FSD50K IDs and prior used uploaders excluded; waveform hashes checked before training.',
        'author_split':assignments,'limitations':'Author labels may contain errors. Multiple microphones from a session remain in the same uploader group. Public previews are lossy.'},indent=2))
    print('READY MIXED',len(clean),'splits',dict(Counter(r['split'] for r in clean)),'labels',dict(Counter(r['label'] for r in clean)),flush=True)


if __name__=='__main__':main()
