"""Vietnam-only mixed corpus, with whole recording days kept in one split."""
import hashlib
import json
import random
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import Counter

from prepare_five_cycles import read_url
from prepare_joint_impulses import BufferedZip

ROOT=Path('dataset/run21_vietnam')


def main():
    folder=ROOT/'VietnamForest';folder.mkdir(exist_ok=True)
    if (folder/'raw_manifest.json').exists():return
    metadata=json.loads((ROOT/'source_metadata.json').read_text())
    source=next(f for f in metadata['files'] if f['key']=='clipsforzenodo.zip')
    inventory=json.loads((ROOT/'inventory.json').read_text())
    main=[r for r in inventory if '/imagesvietnamunbalanced_clips/' in r['name'] and r['name'].endswith('.wav')]
    extra=sorted([r for r in inventory if '/testdatacombined_clips/test/noise/' in r['name']
                  and r['name'].endswith('.wav') and not r['name'].startswith('__MACOSX')],key=lambda r:r['offset'])[:80]
    def day(name):return re.search(r'(20\d{6})_',name).group(1)
    days=sorted({day(r['name']) for r in main});random.Random(2121).shuffle(days)
    assignment={d:'train' if i<int(.6*len(days)) else 'val' if i<int(.8*len(days)) else 'test' for i,d in enumerate(days)}
    # The three noise recording days and new evaluation day are separated too.
    assignment.update({'20190824':'train','20190723':'val','20190722':'test','20191120':'test'})
    selected=main+extra;blocks=[];audit=[]
    for collection in [main,extra]:
        start=min(r['offset'] for r in collection)
        end=max(r['offset']+r['compressed']+len(r['name'].encode())+400 for r in collection)
        ranges=list(range(start,end,4_000_000))
        def chunk(left):
            right=min(end-1,left+3_999_999)
            data=read_url(source['links']['self']+f'?sdp_vietnam={left}',left,right,limit=4_000_000)
            assert len(data)==right-left+1
            print('Verified range',left,len(data),flush=True)
            return data
        with ThreadPoolExecutor(max_workers=4) as pool:data=b''.join(pool.map(chunk,ranges))
        blocks.append((start,data));audit.append({'start':start,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    remote=BufferedZip(source['links']['self']+'?sdp_vietnam_tail=21',source['size']);remote.blocks=blocks
    rows=[]
    with zipfile.ZipFile(remote) as archive:
        for r in selected:
            name=r['name'];path=folder/Path(name).name
            if not path.exists():path.write_bytes(archive.read(name))  # ZIP verifies each CRC.
            date=day(name);split=assignment.get(date,'test')
            rows.append({'path':str(path.resolve()),'domain':'VietnamForest','label':int('/gunshot/' in name),
                         'split':split,'source_group':'day:'+date,'original_member':name,
                         'source_url':'https://zenodo.org/records/13893977','original_crc32':r['crc']})
    assert len({r['path'] for r in rows})==len(rows)
    (folder/'raw_manifest.json').write_text(json.dumps(rows,indent=2))
    (folder/'provenance.json').write_text(json.dumps({'source':source,'byte_ranges':audit,'day_split':assignment,
            'selection':'All Vietnam-only author corpus, plus first 80 evaluation noise WAVs by archive position, fixed before any scoring.',
            'Belize_excluded':True,'split':'Whole recording days; author random split reassigned to prevent shared days.',
            'citation':'Vu, Phan, Le and Clink (2024), Ecological Indicators; Zenodo 13893977.',
            'limitation':'Multiple clips share a recording day; individual clips are not independent field sessions.'},indent=2))
    print('READY',len(rows),Counter((r['split'],r['label']) for r in rows),flush=True)


if __name__=='__main__':main()
