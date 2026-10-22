"""Independent indoor recordings, excluding television whose playback may include gunshots."""
import hashlib
import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

from prepare_five_cycles import read_url

ROOT=Path('dataset/run24_realised')


def main():
    folder=ROOT/'ReaLISEDIndoor';folder.mkdir(exist_ok=True)
    if (folder/'raw_manifest.json').exists():return
    metadata=json.loads((ROOT/'source_metadata.json').read_text());f=next(f for f in metadata['files'] if f['key'].endswith('.zip'))
    path=ROOT/f['key']
    if not path.exists():
        def chunk(start):
            end=min(f['size']-1,start+7_999_999)
            data=read_url(f['links']['self']+f'?sdp24_range={start}',start,end,limit=8_000_000)
            assert len(data)==end-start+1
            print('VERIFIED REALISED RANGE',start,len(data),flush=True)
            return data
        with ThreadPoolExecutor(max_workers=3) as pool:data=b''.join(pool.map(chunk,range(0,f['size'],8_000_000)))
        assert hashlib.md5(data).hexdigest()==f['checksum'].split(':')[1];path.write_bytes(data)
    rows=[]
    with zipfile.ZipFile(path) as archive:
        names=sorted(n for n in archive.namelist() if n.lower().endswith('.flac') and not n.startswith('__MACOSX') and not Path(n).name.startswith('tel_'))
        classes=sorted({Path(n).stem.split('_')[0] for n in names});random.Random(2424).shuffle(classes)
        assignment={c:'train' if i<11 else 'val' if i<14 else 'test' for i,c in enumerate(classes)}
        assert len(classes)==17
        for n in names:
            target=folder/Path(n).name
            if not target.exists():target.write_bytes(archive.read(n))
            cls=Path(n).stem.split('_')[0]
            rows.append({'path':str(target.resolve()),'domain':'ReaLISEDIndoor','label':0,'split':assignment[cls],
                         'source_group':'event_class:'+cls,'event_class':cls,'original_member':n,
                         'source_url':'https://zenodo.org/records/6488321'})
    (folder/'raw_manifest.json').write_text(json.dumps(rows,indent=2))
    (folder/'provenance.json').write_text(json.dumps({'source':f,'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'class_split':assignment,'television_excluded':True,'selection':'All non-television original FLAC recordings, selected before model scoring.',
        'split':'Entire event classes separated; no original-session or physical-object IDs are supplied, so room/object independence is not claimed.',
        'citation':'Mohino-Herranz et al., Real-Life Indoor Sound Event Dataset (ReaLISED), 2022.',
        'label_basis':'Author-labelled real indoor non-firearm activities; television excluded to avoid labelled-negative playback gunshots.'},indent=2))
    print('READY REALISED',len(rows),Counter(r['split'] for r in rows),flush=True)


if __name__=='__main__':main()
