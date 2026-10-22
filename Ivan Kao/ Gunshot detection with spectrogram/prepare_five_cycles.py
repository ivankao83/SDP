"""Download compact portions of five new corpora; split before inference."""
import concurrent.futures
import hashlib
import io
import json
import random
import re
import struct
import time
import urllib.request
import zipfile
import datetime
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path('dataset/five_new_cycles')
META = ROOT / 'source_metadata'


def read_url(url, start=None, end=None, limit=150_000_000):
    for attempt in range(5):
        try:
            headers = {'User-Agent': 'SDP-audio-research/1.0'}
            if start is not None:
                headers['Range'] = f'bytes={start}-{end}'
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=50) as response:
                if start is not None and response.status != 206:
                    raise ValueError('Server did not honor selective download request.')
                data = response.read(limit + 1)
                if len(data) > limit:
                    raise ValueError('Download exceeded size cap.')
                return data
        except (OSError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def download(url, path, sha256=None, md5=None, size=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        data = read_url(url, limit=size or 150_000_000)
        if size is not None:
            assert len(data) == size
        if sha256:
            assert hashlib.sha256(data).hexdigest() == sha256
        if md5:
            assert hashlib.md5(data).hexdigest() == md5
        path.write_bytes(data)
    return path


class RemoteZip(io.RawIOBase):
    """Read only selected public ZIP members using HTTP byte ranges."""
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.tail_start = max(0, size - 2_000_000)
        self.tail = read_url(url, self.tail_start, size - 1, limit=2_000_000)
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position
    def read(self, count=-1):
        if count < 0: count = self.size - self.position
        count = min(count, self.size - self.position)
        if count <= 0: return b''
        if self.position >= self.tail_start:
            data = self.tail[self.position-self.tail_start:self.position-self.tail_start+count]
        else:
            data = read_url(self.url, self.position, self.position+count-1, limit=max(count, 1024))
        assert len(data) == count, (self.url, self.position, count, len(data))
        self.position += count
        return data


def remote_archive(details):
    return zipfile.ZipFile(RemoteZip(details['download_url'], details['size']))


def audiomnist():
    name = 'AudioMNIST'
    folder = ROOT / name
    folder.mkdir(exist_ok=True)
    tree = json.loads((META/'audiomnist.txt').read_text())
    entries = [r for r in tree['tree'] if r['path'].endswith('.wav')]
    # All 60 speakers, five examples of each digit per speaker. Speaker-separated.
    speakers = sorted({Path(r['path']).parent.name for r in entries})
    rng = random.Random(1010); rng.shuffle(speakers)
    split = {s: 'train' if i < 40 else 'val' if i < 50 else 'test' for i, s in enumerate(speakers)}
    selected = [r for r in entries if int(Path(r['path']).stem.split('_')[-1]) < 5]
    (folder/'selection.json').write_text(json.dumps({'revision':tree['sha'], 'speaker_split':split,'entries':selected},indent=2))
    def one(row):
        source = f"https://raw.githubusercontent.com/soerenab/AudioMNIST/{tree['sha']}/{row['path']}"
        path = download(source, folder/Path(row['path']).name, size=row['size'])
        data = path.read_bytes()
        assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest() == row['sha']
        speaker = Path(row['path']).parent.name
        return {'path':str(path.resolve()), 'domain':name, 'label':0, 'split':split[speaker],
                'source_group':'speaker:'+speaker, 'source_url':'https://github.com/soerenab/AudioMNIST'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        rows=[]
        for i,row in enumerate(pool.map(one,selected)):
            rows.append(row)
            if (i+1)%300==0:print('AudioMNIST downloaded',i+1,flush=True)
    save(name,rows)


def save(name, rows):
    (ROOT/name/'raw_manifest.json').write_text(json.dumps(rows,indent=2))
    print('READY',name,len(rows),dict(Counter(r['split'] for r in rows)),flush=True)


def inventory_remote():
    specs = []
    for row in json.loads((META/'m16_files.json').read_text()):
        specs.append(('M16_'+row['filename'],row['content_details']))
    files = json.loads((META/'forest_folder_7c501143-6f46-412f-b784-b84fad4ccdb0.json').read_text())
    specs.append(('Forest_background',files[0]['content_details']))
    for name,details in specs:
        path=META/(name+'_inventory.json')
        if not path.exists():
            with remote_archive(details) as archive:
                rows=[{'name':r.filename,'size':r.file_size,'compressed':r.compress_size,'crc':r.CRC}
                      for r in archive.infolist()]
            path.write_text(json.dumps(rows,indent=2))
        rows=json.loads(path.read_text())
        print('INVENTORY',name,len(rows),rows[:12],flush=True)


def extract_remote(details, selected, consume, workers=4):
    # A separate ZIP reader per worker keeps the seek/read sequence thread-safe.
    batches = [selected[i::workers] for i in range(workers)]
    def batch(items):
        if not items: return []
        with remote_archive(details) as archive:
            return [consume(item, archive) for item in items]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return [row for group in pool.map(batch, batches) for row in group]


def m16_distance(path):
    stem = Path(path).stem
    if '-' in stem:
        return int(re.search(r'-(\d+)',stem).group(1))
    index = int(re.match(r'\d+',stem).group())
    return 50 if index <= 6 else 60 if index <= 12 else 70 if index <= 19 else 80 if index <= 25 else 90 if index <= 31 else 100


def m16():
    name='M16Firecrackers';folder=ROOT/name;folder.mkdir(exist_ok=True)
    rows=[];selection=[]
    for file in json.loads((META/'m16_files.json').read_text()):
        label=int(file['filename']=='gunshot.zip')
        entries=json.loads((META/('M16_'+file['filename']+'_inventory.json')).read_text())
        entries=[r for r in entries if r['name'].lower().endswith('.wav')]
        if label:
            used=set();chosen=[]
            for row in entries:
                angle=Path(row['name']).parent.name;key=(angle,m16_distance(row['name']))
                if key in used:continue
                used.add(key);chosen.append(row)
            entries=chosen
        selection.extend([{**r,'label':label,'distance_m':m16_distance(r['name'])} for r in entries])
        def consume(item,archive):
            distance=m16_distance(item['name'])
            split='train' if distance in (50,60,70) else 'val' if distance==80 else 'test'
            path=folder/(('gun_' if label else 'cracker_')+item['name'].replace('/','_'))
            if not path.exists():
                # Fixed microphone; retain the 3-second interval around the
                # largest absolute impulse, identically for both classes.
                audio,sr=sf.read(io.BytesIO(archive.read(item['name'])),dtype='float32',always_2d=True)
                audio=audio[:,0];peak=int(np.abs(audio).argmax())
                start=max(0,min(peak-sr,len(audio)-3*sr))
                sf.write(path,audio[start:start+3*sr],sr,subtype='FLOAT')
            return {'path':str(path.resolve()),'domain':name,'label':label,'split':split,
                    'source_group':'distance:'+str(distance),'original_member':item['name'],
                    'original_zip_crc32':item['crc'],'preprocessing':'channel 1; 3 seconds around absolute peak',
                    'source_url':'https://data.mendeley.com/datasets/kbgw323248/1'}
        rows.extend(extract_remote(file['content_details'],entries,consume))
        print('M16 downloaded class',label,len(entries),flush=True)
    (folder/'selection.json').write_text(json.dumps(selection,indent=2))
    save(name,rows)


def forest():
    name='TropicalForest';folder=ROOT/name;folder.mkdir(exist_ok=True)
    trainpos=json.loads((META/'forest_folder_d21ad8ca-ca07-4a75-a852-29c2a505e62b.json').read_text())
    testpos=json.loads((META/'forest_folder_83dd9428-6b72-4db7-90e8-70f84be9fa7d.json').read_text())
    testneg=json.loads((META/'forest_folder_86d76b5d-a89c-41c7-9bab-76b0c9de9e63.json').read_text())
    trainneg=json.loads((META/'Forest_background_inventory.json').read_text())
    def day(filename):
        return datetime.datetime.fromtimestamp(int(Path(filename).name[:8],16),datetime.timezone.utc).date().isoformat()
    testdays={day(r['filename']) for r in testpos+testneg}
    candidates=[]
    for label,items in [(1,trainpos),(0,trainneg)]:
        for item in items:
            filename=item.get('filename',item.get('name'))
            if not filename.lower().endswith('.wav') or day(filename) in testdays:continue
            candidates.append({**item,'label':label,'day':day(filename),'remote_zip':not label})
    rng=random.Random(1313)
    # Each day has exactly one split, even across positive and negative labels.
    days=sorted({r['day'] for r in candidates});rng.shuffle(days)
    valdays=set(days[:max(1,len(days)//5)])
    selected=[]
    for split,maximum in [('train',160),('val',40),('test',60)]:
        for label in [0,1]:
            if split=='test':
                pool=[{**r,'label':label,'day':day(r['filename']),'remote_zip':False} for r in (testpos if label else testneg)]
            else:pool=[r for r in candidates if r['label']==label and (r['day'] in valdays)==(split=='val')]
            rng.shuffle(pool);counts=Counter();chosen=[]
            for row in pool:
                if counts[row['day']]>=10:continue
                counts[row['day']]+=1;chosen.append({**row,'split':split})
                if len(chosen)==maximum:break
            assert len(chosen)>=10,(split,label,len(chosen))
            selected.extend(chosen)
    (folder/'selection.json').write_text(json.dumps(selected,indent=2))
    def record(row,path):
        return {'path':str(path.resolve()),'domain':name,'label':row['label'],'split':row['split'],
                'source_group':'day:'+row['day'],'source_url':'https://data.mendeley.com/datasets/x48cwz364j/3'}
    def direct(row):
        d=row['content_details'];path=download(d['download_url'],folder/row['filename'],sha256=d['sha256_hash'],size=d['size'])
        return record(row,path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        rows=list(pool.map(direct,[r for r in selected if not r['remote_zip']]))
    details=json.loads((META/'forest_folder_7c501143-6f46-412f-b784-b84fad4ccdb0.json').read_text())[0]['content_details']
    def remote(row,archive):
        path=folder/Path(row['name']).name
        if not path.exists():path.write_bytes(archive.read(row['name']))
        return record(row,path)
    rows.extend(extract_remote(details,[r for r in selected if r['remote_zip']],remote))
    save(name,rows)


def demand():
    name='DEMAND';folder=ROOT/name;folder.mkdir(exist_ok=True)
    files=sorted([r for r in json.loads((META/'demand.json').read_text())['files'] if r['key'].endswith('_16k.zip')],key=lambda r:r['key'])
    rng=random.Random(1212);rng.shuffle(files)
    # Entire recorded scenes (all excerpts from one source) share one split.
    selection=[{'key':r['key'],'split':'train' if i<12 else 'val' if i<15 else 'test'} for i,r in enumerate(files)]
    (folder/'selection.json').write_text(json.dumps(selection,indent=2))
    def one(pair):
        file,selected=pair;scene=file['key'].split('_')[0];rows=[]
        mono=folder/(scene+'_ch01.wav')
        if not mono.exists():
            with remote_archive({'download_url':file['links']['self'],'size':file['size']}) as archive:
                members=[s for s in archive.namelist() if re.search(r'ch0?1\.wav$',s,re.I)]
                assert len(members)==1,archive.namelist()
                mono.write_bytes(archive.read(members[0]))
        audio,sr=sf.read(mono,dtype='float32');assert audio.ndim==1
        for i,start in enumerate(range(0,len(audio)-3*sr+1,10*sr)):
            path=folder/(scene+f'_{i:03d}.wav')
            if not path.exists():sf.write(path,audio[start:start+3*sr],sr,subtype='FLOAT')
            rows.append({'path':str(path.resolve()),'domain':name,'label':0,'split':selected['split'],
                         'source_group':'scene:'+scene,'original_recording':str(mono.resolve()),
                         'excerpt_start_seconds':start/sr,'source_url':'https://zenodo.org/records/1227121'})
        print('DEMAND downloaded',scene,len(rows),flush=True)
        return rows
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows=[r for group in pool.map(one,zip(files,selection)) for r in group]
    save(name,rows)


def office():
    name='DCASEOffice';folder=ROOT/name;folder.mkdir(exist_ok=True)
    if (folder/'raw_manifest.json').exists():
        saved=json.loads((folder/'raw_manifest.json').read_text())
        if all(Path(r['path']).is_file() for r in saved):
            print('READY',name,len(saved),'reusing saved excerpts',flush=True)
            return
    # Only isolated author training events enter development. Synthetic author
    # evaluation recordings use a separate source pool and supply the test set.
    train=ROOT/'archives/dcase2016_task2_train_dev.zip'
    test=ROOT/'archives/dcase2016_task2_test_public.zip'
    for target,metadata,identifier in [
        (train,META/'dcase_train.json','dcase2016_task2_train_dev'),
        (test,Path('dataset/test_only_reserve/dcase_archive_metadata.json'),'dcase2016_task2_test_public')]:
        if not target.exists():
            source=next(r for r in json.loads(metadata.read_text())['files'] if r['name']==target.name)
            download(f'https://archive.org/download/{identifier}/{target.name}',target,
                     size=int(source['size']),md5=source['md5'])
    rows=[];rng=random.Random(1111)
    with zipfile.ZipFile(train) as archive:
        names=[s for s in archive.namelist() if s.lower().endswith('.wav') and '/dcase2016_task2_train/' in s]
        classes=sorted({re.sub(r'\d+$','',Path(s).stem) for s in names})
        plan=[]
        for cls in classes:
            group=sorted(s for s in names if re.sub(r'\d+$','',Path(s).stem)==cls)
            rng.shuffle(group)
            plan.extend((s,'val' if i<4 else 'train') for i,s in enumerate(group))
        for s,split in plan:
            rows.append(office_excerpt(archive,s,split,folder,name))
        (folder/'SOURCE_README.txt').write_bytes(archive.read('dcase2016_task2_train_dev/README.txt'))
    with zipfile.ZipFile(test) as archive:
        for s in sorted(archive.namelist()):
            if s.lower().endswith('.wav'):
                rows.append(office_excerpt(archive,s,'test',folder,name))
    (folder/'selection.json').write_text(json.dumps(rows,indent=2))
    save(name,rows)


def office_excerpt(archive,member,split,folder,name):
    path=folder/Path(member).name
    if not path.exists():
        audio,sr=sf.read(io.BytesIO(archive.read(member)),dtype='float32',always_2d=True)
        audio=audio.mean(axis=1);peak=int(np.abs(audio).argmax())
        start=max(0,min(peak-sr,len(audio)-3*sr))
        sf.write(path,audio[start:start+3*sr],sr,subtype='FLOAT')
    return {'path':str(path.resolve()),'domain':name,'label':0,'split':split,
            'source_group':'author_evaluation_pool' if split=='test' else 'isolated:'+Path(member).stem,
            'original_member':member,'preprocessing':'up to 3 seconds around absolute peak',
            'grouping_limitation':'Isolated author training clips have no original-session identifiers; evaluation uses separate author source pool.',
            'source_url':'https://dcase.community/challenge2016/task-sound-event-detection-in-synthetic-audio'}


if __name__ == '__main__':
    import sys
    ROOT.mkdir(parents=True,exist_ok=True)
    {'inventory':inventory_remote,'audiomnist':audiomnist,'forest':forest,
     'm16':m16,'demand':demand,'office':office}[sys.argv[1]]()
