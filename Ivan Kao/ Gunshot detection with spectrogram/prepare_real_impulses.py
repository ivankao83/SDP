"""Independent real-world transient negatives and source-filtered gunshots."""
import io
import json
import random
import re
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from prepare_five_cycles import read_url,download

ROOT=Path('dataset/run16_real_impulses')
META=ROOT/'source_metadata'


def save(name,rows,provenance):
    folder=ROOT/name;folder.mkdir(exist_ok=True)
    (folder/'raw_manifest.json').write_text(json.dumps(rows,indent=2))
    (folder/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('READY',name,len(rows),dict(Counter(r['split'] for r in rows)),flush=True)


def excerpt(data,path):
    if path.exists():return
    audio,sr=sf.read(io.BytesIO(data),dtype='float32',always_2d=True)
    audio=audio.mean(axis=1);peak=int(np.abs(audio).argmax())
    start=max(0,min(peak-sr,len(audio)-3*sr))
    sf.write(path,audio[start:start+3*sr],sr,subtype='FLOAT')


def cmu():
    name='CMURealSounds';folder=ROOT/name;folder.mkdir(exist_ok=True)
    if (folder/'raw_manifest.json').exists():
        saved=json.loads((folder/'raw_manifest.json').read_text())
        if all(Path(r['path']).is_file() for r in saved):
            print('READY',name,len(saved),'reusing saved audio',flush=True);return
    entries=[]
    for path in sorted((ROOT/'archives').glob('*.zip')):
        if path.name=='Impact-set.zip':continue
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if member.lower().endswith(('.wav','.aif','.aiff')) and not member.startswith('__MACOSX'):
                    entries.append({'archive':str(path),'member':member,'category':Path(member).parent.name})
            for member in archive.namelist():
                if Path(member).name=='readmeSoundEvents.txt' and not member.startswith('__MACOSX'):
                    (folder/'SOURCE_README.txt').write_bytes(archive.read(member));break
    categories=sorted({r['category'] for r in entries});random.Random(1616).shuffle(categories)
    n=len(categories);assignment={s:'train' if i<int(.7*n) else 'val' if i<int(.85*n) else 'test' for i,s in enumerate(categories)}
    rows=[]
    for entry in entries:
        category=entry['category'];path=folder/(category+'__'+Path(entry['member']).stem+'.wav')
        if not path.exists():
            with zipfile.ZipFile(entry['archive']) as archive:excerpt(archive.read(entry['member']),path)
        rows.append({'path':str(path.resolve()),'domain':name,'label':0,'split':assignment[category],
                     'source_group':'object_event:'+category,'source_category':category,
                     'source_url':'https://www.auditorylab.org/sound-events-database',
                     'original_member':entry['member']})
    save(name,rows,{'source':'https://www.auditorylab.org/sound-events-database',
         'citation':'Sounds courtesy of the Sound Events Database. Copyright 2008, Laurie M. Heller. Funding NSF award 0446955.',
         'label_basis':'Author-recorded everyday air, object impact, and deformation events; no firearm discharges.',
         'split':'Entire object/event categories held apart; excerpts up to three seconds around the strongest absolute impulse.'})


def lab():
    name='LabImpulses';folder=ROOT/name;folder.mkdir(exist_ok=True)
    files=[r for r in json.loads((META/'impulses.json').read_text())['files'] if r['key'].endswith('.wav')]
    rows=[]
    # The real balloon and starter clapper remain training examples; different
    # recorded objects provide validation and test. No augmented file crosses.
    for file in files:
        index=int(re.match(r'S(\d+)',file['key']).group(1));split='val' if index in (5,7) else 'test' if index in (3,6) else 'train'
        path=download(file['links']['self'],folder/file['key'],size=file['size'],md5=file['checksum'].split(':')[1])
        rows.append({'path':str(path.resolve()),'domain':name,'label':0,'split':split,
                     'source_group':'original_impulse:'+str(index),'source_url':'https://zenodo.org/records/8112384'})
    save(name,rows,{'source':'https://zenodo.org/records/8112384','label_basis':'Author-recorded balloon, cutlery, book, dishes, glass, knife, keys and starter clapper; all negative for firearm discharge.',
         'limitation':'Nine short recordings only; augmentation does not create independent recordings.'})


def balloons():
    name='IndependentBalloonPops';folder=ROOT/name;folder.mkdir(exist_ok=True)
    sources=[('Sadiquecat',814343,'bathroom','val'),('Sadiquecat',866967,'shower','test'),
             ('Sadiquecat',869521,'bedroom','train'),('Sadiquecat',850645,'caravan','train'),
             ('Sadiquecat',862038,'chimney','train'),('Sadiquecat',860766,'outdoor_park','train'),
             ('Breviceps',458398,'balloon_pop','train')]
    known=set()
    for path in Path('dataset/test_only_reserve/FSD50K_metadata').rglob('*clips_info_FSD50K.json'):known|=set(json.loads(path.read_text()))
    def one(item):
        author,id,room,split=item
        assert str(id) not in known
        page=f'https://freesound.org/people/{author}/sounds/{id}/'
        html_path=META/f'balloon_{id}.html'
        if not html_path.exists():html_path.write_bytes(read_url(page,limit=2_000_000))
        html=html_path.read_text();assert 'publicdomain/zero' in html, 'Expected author CC0 license'
        urls=re.findall(r'https://cdn\.freesound\.org/previews/[^\"\s<>]+-hq\.mp3',html)
        assert urls,page
        preview=download(urls[0],folder/(str(id)+'.mp3'),size=None)
        return {'path':str(preview.resolve()),'domain':name,'label':0,'split':split,
                'source_group':author+':'+room,'freesound_id':str(id),'uploader':author,
                'source_url':page,'preview_url':urls[0],'license':'CC0',
                'label_basis':'Author explicitly describes recorded balloon pops, retaining the pop itself.'}
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,sources))
    save(name,rows,{'source':'Original Freesound author pages saved in source_metadata.',
         'label_basis':'Author-described balloon-pop recordings, CC0 public high-quality previews.',
         'split':'Different recordings and rooms held apart. The main uploader appears in all splits, so uploader independence is not claimed.',
         'overlap':'All seven source IDs excluded from every FSD50K source ID; none is a reserved test recording.'})


def variations():
    name='RealImpactVariations';folder=ROOT/name;folder.mkdir(exist_ok=True)
    if (folder/'raw_manifest.json').exists():
        saved=json.loads((folder/'raw_manifest.json').read_text())
        if all(Path(r['path']).is_file() for r in saved):
            print('READY',name,len(saved),'reusing saved audio',flush=True);return
    metadata=json.loads((META/'impact_audio_metadata.json').read_text())
    known=set()
    for path in Path('dataset/test_only_reserve/FSD50K_metadata').rglob('*clips_info_FSD50K.json'):known|=set(json.loads(path.read_text()))
    # Known ESC-50 and UrbanSound source IDs cannot enter the new set either.
    import csv
    for path in [Path('dataset/esc50_manifest.csv'),Path('dataset/continual_cycles/urban_train.csv'),Path('dataset/continual_cycles/urban_test.csv')]:
        for row in csv.DictReader(path.open()):known.add(row.get('source_id',Path(row.get('path','')).stem.split('-')[0]))
    reserve=json.loads(Path('dataset/test_only_reserve/FSD50K_mixed/manifest.json').read_text())
    old_authors={r['clip_metadata']['uploader'].lower() for r in reserve}
    chosen=[]
    for row in metadata:
        if row['freesound_id'] in known or row['author'].lower() in old_authors:continue
        if row['category'] not in ['gunshots','bounce','knocks','smash','punch']:continue
        if row['category']=='gunshots' and re.search(r'balloon|laser|sci-fi|sci_fi|synth|video.?game|fake|cartoon',row['title'],re.I):continue
        chosen.append(row)
    authors=sorted({r['author'] for r in chosen});random.Random(1617).shuffle(authors)
    n=len(authors);assignment={a:'train' if i<int(.7*n) else 'val' if i<int(.85*n) else 'test' for i,a in enumerate(authors)}
    # Prespecified caps within label/split, keeping source uploads together.
    selected=[];rng=random.Random(1618)
    for split,cap in [('train',180),('val',45),('test',45)]:
        for label in [0,1]:
            group=[r for r in chosen if int(r['category']=='gunshots')==label and assignment[r['author']]==split]
            rng.shuffle(group);ids=[]
            for r in group:
                if r['freesound_id'] not in ids:ids.append(r['freesound_id'])
            kept=set(ids[:cap]);selected.extend(r for r in group if r['freesound_id'] in kept)
    rows=[]
    with zipfile.ZipFile(ROOT/'archives/Impact-set.zip') as archive:
        for row in selected:
            member=f"Impact-set/{row['original_set']}/{row['category']}/{row['index']}.wav"
            path=folder/(row['category']+'_'+row['index']+'.wav')
            excerpt(archive.read(member),path)
            rows.append({'path':str(path.resolve()),'domain':name,'label':int(row['category']=='gunshots'),
                         'split':assignment[row['author']],'source_group':'author:'+row['author'],
                         'freesound_id':row['freesound_id'],'uploader':row['author'],'source_title':row['title'],
                         'source_url':'https://zenodo.org/records/14286414'})
    save(name,rows,{'source':'https://zenodo.org/records/14286414','label_basis':'Author impact categories; exclude explicit balloon/synthetic/game sounds from gunshot category.',
         'split':'Entire Freesound uploaders held apart. All fragments of any source ID remain in its uploader split.',
         'overlap':'Exclude all FSD50K, ESC-50 and UrbanSound source IDs and all known reserved FSD50K uploaders.',
         'limitation':'Category labels may retain errors; original SESA source IDs are unavailable.'})


if __name__=='__main__':
    {'cmu':cmu,'lab':lab,'balloons':balloons,'variations':variations}[sys.argv[1]]()
