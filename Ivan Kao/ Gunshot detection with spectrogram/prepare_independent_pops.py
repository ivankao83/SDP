"""Collect author-labelled packaging/balloon/clap negatives from public licensed previews."""
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch
import gunshot as g
from continual_cycles import feature,fingerprint

ROOT=Path('dataset/run38_independent_pops')
FOLDER=ROOT/'IndependentPops'


def get(url,limit):
    req=urllib.request.Request(url,headers={'User-Agent':'SDP-acoustic-research/1.0'})
    with urllib.request.urlopen(req,timeout=20) as response:
        data=response.read(limit+1)
        if len(data)>limit:raise ValueError('Download exceeds compact-subset cap')
        return data


def main():
    torch.set_num_threads(2);FOLDER.mkdir(exist_ok=True);meta=ROOT/'source_metadata';meta.mkdir(exist_ok=True)
    cache=FOLDER/'features';cache.mkdir(exist_ok=True)
    cfg=torch.load('runs/run14_m16_firecrackers/best.pt',map_location='cpu',weights_only=True)['config']
    candidates=json.loads((ROOT/'filtered_candidates.json').read_text());rng=random.Random(3838)
    groups={c:[r for r in candidates if r['category']==c] for c in ['bag_pop','balloon_pop','packaging','clap']}
    for items in groups.values():rng.shuffle(items)
    ordered=groups['bag_pop']+groups['balloon_pop']
    packs,claps=groups['packaging'][:],groups['clap'][:]
    while packs or claps:
        for _ in range(2):
            if packs:ordered.append(packs.pop())
        if claps:ordered.append(claps.pop())
    count=Counter();chosen=[]
    for r in ordered:
        if count[r['author']]>=(12 if r['category'] in ['bag_pop','balloon_pop'] else 8):continue
        count[r['author']]+=1;chosen.append(r)
    (ROOT/'prespecified_order.json').write_text(json.dumps([r['id'] for r in chosen],indent=2))
    def one(r):
        identifier=r['id'];record=meta/(identifier+'.json')
        if record.exists():
            saved=json.loads(record.read_text())
            if Path(saved['path']).exists() and Path(saved['feature_path']).exists():return saved
        try:
            page='https://freesound.org'+r['href'];html_path=meta/(identifier+'.html')
            if not html_path.exists():html_path.write_bytes(get(page,2_000_000))
            html=html_path.read_text();licenses=re.findall(r'https?://creativecommons.org/[^\"<>\s]+',html)
            license=next((s for s in licenses if '/publicdomain/zero/' in s or re.search(r'/licenses/by/\d',s)),None)
            if not license:raise ValueError('No author CC0 or attribution-only licence verified')
            urls=re.findall(r'https://cdn\.freesound\.org/previews/[^\"\s<>]+-hq\.mp3',html)
            if not urls:raise ValueError('Public high-quality preview unavailable')
            preview=get(urls[0],5_000_000);temporary=FOLDER/(identifier+'.download.mp3');temporary.write_bytes(preview)
            try:audio=g.read_clip(temporary)
            finally:temporary.unlink(missing_ok=True)
            peak=int(np.abs(audio).argmax());start=max(0,min(peak-g.SR,len(audio)-g.SAMPLES));audio=audio[start:start+g.SAMPLES]
            path=FOLDER/(identifier+'.wav');sf.write(path,audio,g.SR,subtype='FLOAT')
            digest=hashlib.sha256(audio.tobytes()).hexdigest();fp=cache/(digest+'.npy')
            if not fp.exists():np.save(fp,feature(audio,cfg))
            domain='IndependentBalloonCollection' if r['category']=='balloon_pop' else 'IndependentPackaging' if r['category'] in ['packaging','bag_pop'] else 'IndependentClaps'
            row={'path':str(path.resolve()),'domain':domain,'label':0,'source_group':'uploader:'+r['author'].casefold(),
                 'uploader':r['author'],'freesound_id':identifier,'category':r['category'],'source_url':page,
                 'preview_url':urls[0],'license':license,'original_description':r['description'],'original_title':r['file_name'],
                 'preview_sha256':hashlib.sha256(preview).hexdigest(),'excerpt_start_seconds':start/g.SR,
                 'decoded_sha256':digest,'sha256':fingerprint(path),'feature_path':str(fp.resolve()),
                 'label_basis':'Original author material/action description; no WavCaps-generated caption used as the label.'}
            record.write_text(json.dumps(row,indent=2));return row
        except Exception as error:return {'excluded_id':identifier,'reason':type(error).__name__+': '+str(error)}
    rows=[];excluded=[]
    for offset in range(0,len(chosen),50):
        with ThreadPoolExecutor(max_workers=2) as pool:batch=list(pool.map(one,chosen[offset:offset+50]))
        for item in batch:
            if 'excluded_id' in item:excluded.append(item)
            else:rows.append(item)
        print('COLLECTED',len(rows),'excluded',len(excluded),'categories',dict(Counter(r['category'] for r in rows)),flush=True)
        (ROOT/'collection_progress.json').write_text(json.dumps({'downloaded':rows,'excluded':excluded},indent=2))
        if len(rows)>=400:break
    assert len(rows)>=200,('Insufficient licensed, available original clips',len(rows))
    # Entire uploaders, across every category, remain in one split.
    by_author=defaultdict(list)
    for r in rows:by_author[r['uploader']].append(r)
    priorities={'bag_pop':0,'balloon_pop':1,'packaging':2,'clap':3}
    authors_by_type=defaultdict(list)
    for author,items in by_author.items():authors_by_type[min(items,key=lambda r:priorities[r['category']])['category']].append(author)
    assignment={}
    for category,authors in sorted(authors_by_type.items()):
        authors=sorted(authors);rng.shuffle(authors);n=len(authors)
        a=max(1,int(.6*n));b=min(n,a+max(1,int(.2*n)))
        assignment.update({author:'train' if i<a else 'val' if i<b else 'test' for i,author in enumerate(authors)})
    seen=set();clean=[]
    for r in rows:
        if r['decoded_sha256'] in seen:excluded.append({'excluded_id':r['freesound_id'],'reason':'Decoded duplicate within collection'});continue
        seen.add(r['decoded_sha256']);clean.append({**r,'split':assignment[r['uploader']]})
    assert len(clean)>=200
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not {r['uploader'] for r in clean if r['split']==a}&{r['uploader'] for r in clean if r['split']==b}
    assert all(sum(r['split']==s for r in clean)>=20 for s in ['train','val','test'])
    (FOLDER/'manifest.json').write_text(json.dumps(clean,indent=2));(ROOT/'excluded.json').write_text(json.dumps(excluded,indent=2))
    (ROOT/'ATTRIBUTIONS.md').write_text('# Original audio attribution\n\n'+'\n'.join(f"- {r['freesound_id']}: {r['uploader']}; {r['source_url']}; {r['license']}" for r in clean)+'\n')
    (FOLDER/'provenance.json').write_text(json.dumps({'index':'https://huggingface.co/datasets/cvssp/WavCaps','audio':'Original author public high-quality previews; never login-gated originals.',
        'labeling':'Original author descriptions, not generated captions. Packaging, balloon pops and claps are non-firearm sounds.',
        'preprocessing':'One original source ID -> one up-to-three-second peak excerpt; no augmentation counted as original clips.',
        'split':'Entire uploaders separated across categories. Assignment uses metadata only, before scoring.',
        'overlap_exclusions':json.loads((ROOT/'overlap_exclusions.json').read_text()),
        'author_split':assignment,'limitation':'Author labels may contain errors; claps/packaging are diverse confounds, not all clips are bag explosions.'},indent=2))
    print('READY POPS',len(clean),Counter(r['split'] for r in clean),Counter(r['domain'] for r in clean),flush=True)


if __name__=='__main__':main()
