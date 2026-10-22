"""Prepare independent new impact/gunshot sources for one joint cycle."""
import csv
import hashlib
import io
import json
import random
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf
import pyarrow.parquet as pq

from prepare_five_cycles import RemoteZip, read_url, remote_archive, download
import zipfile

ROOT=Path('dataset/run15_joint')
META=ROOT/'source_metadata'


def save(name,rows,notes):
    folder=ROOT/name
    (folder/'raw_manifest.json').write_text(json.dumps(rows,indent=2))
    (folder/'provenance.json').write_text(json.dumps(notes,indent=2))
    print('READY',name,len(rows),dict(Counter(r['split'] for r in rows)),flush=True)


def hits():
    name='GreatestHits';folder=ROOT/name;folder.mkdir(exist_ok=True)
    if (folder/'raw_manifest.json').exists():
        saved=json.loads((folder/'raw_manifest.json').read_text())
        if all(Path(r['path']).is_file() for r in saved):
            print('READY',name,len(saved),'reusing saved audio',flush=True);return
    table=pq.read_table(ROOT/'greatest_hits.parquet').to_pylist()
    days=sorted({r['id'][2:12] for r in table});random.Random(1501).shuffle(days)
    n=len(days);assignment={day:'train' if i<int(.7*n) else 'val' if i<int(.85*n) else 'test' for i,day in enumerate(days)}
    rows=[]
    for row in table:
        day=row['id'][2:12];path=folder/(row['id']+'.wav')
        if not path.exists():path.write_bytes(row['audio']['bytes'])
        rows.append({'path':str(path.resolve()),'domain':name,'label':0,'split':assignment[day],
                     'source_group':'day:'+day,'source_id':row['id'],
                     'source_url':'https://www.andrewowens.com/vis/hits.html'})
    save(name,rows,{'original_source':'https://www.andrewowens.com/vis/hits.html',
         'mirror':'https://huggingface.co/datasets/dukesun99/GreatestHits-V2A',
         'parquet_sha256':hashlib.sha256((ROOT/'greatest_hits.parquet').read_bytes()).hexdigest(),
         'label_basis':'Recorded drumstick interactions with everyday objects; all negative for firearm discharge.',
         'split':'Recording day separated, including all original-video fragments on one day.',
         'days':assignment})


class BufferedZip(RemoteZip):
    def __init__(self,url,size):
        super().__init__(url,size);self.blocks=[]
    def read(self,count=-1):
        count=self.size-self.position if count<0 else min(count,self.size-self.position)
        for start,data in self.blocks:
            if start<=self.position and self.position+count<=start+len(data):
                offset=self.position-start;self.position+=count
                return data[offset:offset+count]
        return super().read(count)


def synthetic():
    name='SixKSFX';folder=ROOT/name;folder.mkdir(exist_ok=True)
    source=json.loads((META/'6ksfx.json').read_text())['files'][0]
    classes=['GunshotSynth','FireworksSynth','CracklingSynth','DebrisGlassSynth','MetalDebrisSynth','ApplauseSynth']
    groups=list(range(50));random.Random(1502).shuffle(groups)
    assignment={key:'train' if i<30 else 'val' if i<40 else 'test' for i,key in enumerate(groups)}
    def one(cls):
        remote=BufferedZip(source['links']['self'],source['size'])
        with zipfile.ZipFile(remote) as archive:
            infos=[r for r in archive.infolist() if r.filename.startswith('6KSFx/'+cls+'/') and r.filename.endswith('.wav')]
            # Adjacent ZIP members for one class can be read in a single range.
            # Avoid thousands of small HTTP requests and a 1.6 GB local archive.
            start=min(r.header_offset for r in infos)
            end=max(r.header_offset+30+len(r.filename.encode())+len(r.extra)+r.compress_size+128 for r in infos)
            remote.blocks=[(start,read_url(source['links']['self'],start,end-1,limit=end-start))]
            rows=[]
            for info in infos:
                index=int(re.search(r'Synth(\d+)-',Path(info.filename).name).group(1))
                path=folder/Path(info.filename).name
                if not path.exists():path.write_bytes(archive.read(info.filename))
                group=index%50
                rows.append({'path':str(path.resolve()),'domain':name,'label':int(cls=='GunshotSynth'),
                             'split':assignment[group],'source_group':f'{cls}:family{group}',
                             'original_member':info.filename,'original_crc32':info.CRC,
                             'source_url':'https://zenodo.org/records/14517916'})
        print('Downloaded synthetic',cls,len(rows),flush=True)
        return rows
    with ThreadPoolExecutor(max_workers=3) as pool:rows=[r for group in pool.map(one,classes) for r in group]
    save(name,rows,{'source':'https://zenodo.org/records/14517916',
         'label_basis':'Author synthesis categories. GunshotSynth positive; fireworks/crackling/glass/metal/applause negative.',
         'split':'Conservative shared family index modulo 50 groups possible numbered noise variants together.',
         'limitation':'Generator seed and augmentation lineage are not supplied; this assumption does not prove independent synthetic source families.',
         'classes':classes,'family_split':assignment})


def hunting():
    name='HuntingGunshots';folder=ROOT/name;folder.mkdir(exist_ok=True)
    groups=list(range(28));random.Random(1503).shuffle(groups)
    assignment={key:'train' if i<18 else 'val' if i<23 else 'test' for i,key in enumerate(groups)}
    rows=[]
    with tarfile.open(ROOT/'Gunshot_hunting.tar.gz') as archive:
        for member in sorted(archive.getmembers(),key=lambda r:r.name):
            if not member.isfile() or not member.name.lower().endswith('.mp3'):continue
            index=int(re.search(r'hunting_(\d+)',member.name).group(1));group=index//20
            path=folder/Path(member.name).name
            if not path.exists():path.write_bytes(archive.extractfile(member).read())
            rows.append({'path':str(path.resolve()),'domain':name,'label':1,'split':assignment[group],
                         'source_group':'numbered_block:'+str(group),'source_url':'https://zenodo.org/records/5068271',
                         'label_confidence':'weak_author_semisupervised','training_weight_multiplier':.25})
    save(name,rows,{'source':'https://zenodo.org/records/5068271','label_basis':'Author semi-supervised gunshot labels from hunting videos.',
         'split':'Adjacent numbered clips grouped in blocks of 20; original YouTube IDs unavailable.',
         'limitation':'Clip grouping cannot establish full original-video independence. Weak labels receive 0.25 relative weight in supervised training.'})


def tut():
    name='TUTRare2017';folder=ROOT/name;folder.mkdir(exist_ok=True)
    original_ids=set()
    for path in Path('dataset/test_only_reserve/FSD50K_metadata').rglob('*.csv'):
        if path.name in ['dev.csv','eval.csv']:original_ids|={r['fname'] for r in csv.DictReader(path.open())}
    for path in [Path('dataset/esc50_manifest.csv'),Path('dataset/continual_cycles/urban_train.csv'),Path('dataset/continual_cycles/urban_test.csv')]:
        for row in csv.DictReader(path.open()):
            original_ids.add(row.get('source_id',Path(row.get('path','')).stem.split('-')[0]))
    source=next(r for r in json.loads((META/'tut2017.json').read_text())['files'] if r['key'].endswith('source_data_events.zip'))
    with remote_archive({'download_url':source['links']['self'],'size':source['size']}) as archive:
        selected=[r for r in archive.infolist() if r.filename.endswith('.wav') and Path(r.filename).stem not in original_ids]
        rows=[];rng=random.Random(1504)
        # Avoid every FSD50K source ID, even those outside the 100-file reserve.
        for cls in ['gunshot','glassbreak','babycry']:
            items=sorted([r for r in selected if Path(r.filename).parent.name==cls],key=lambda r:r.filename);rng.shuffle(items)
            for i,info in enumerate(items):
                split='train' if i<int(.6*len(items)) else 'val' if i<int(.8*len(items)) else 'test'
                path=folder/(cls+'_'+Path(info.filename).name)
                if not path.exists():path.write_bytes(archive.read(info.filename))
                rows.append({'path':str(path.resolve()),'domain':name,'label':int(cls=='gunshot'),'split':split,
                             'source_group':'freesound:'+Path(info.filename).stem,
                             'source_url':'https://zenodo.org/records/1160455'})
    save(name,rows,{'source':'https://zenodo.org/records/1160455','all_FSD50K_source_ids_excluded':True,
         'additional_ESC50_UrbanSound8K_source_ids_excluded':True,
         'label_basis':'Author rare-event category; gunshot positive, glassbreak and babycry negative.',
         'limitation':'Small remainder after overlap exclusion; Freesound uploader independence from SESA cannot be established.'})


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    {'hits':hits,'synthetic':synthetic,'hunting':hunting,'tut':tut}[sys.argv[1]]()
