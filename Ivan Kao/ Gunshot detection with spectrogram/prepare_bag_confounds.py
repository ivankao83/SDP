"""Small independent bag-pop negatives; no reserved collection is copied."""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prepare_five_cycles import read_url,download

ROOT=Path('dataset/run17_bag_confounds')


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    # Reuse fixed, independently recorded negative corpora only. The real-
    # impact collection's broadly labeled positive SFX are excluded entirely.
    for name in ['CMURealSounds','LabImpulses','IndependentBalloonPops']:
        source=Path('dataset/run16_real_impulses')/name
        target=ROOT/name;target.mkdir(exist_ok=True)
        for filename in ['raw_manifest.json','provenance.json']:
            (target/filename).write_text((source/filename).read_text())
    name='IndependentBagPops';folder=ROOT/name;folder.mkdir(exist_ok=True)
    meta=ROOT/'source_metadata';meta.mkdir(exist_ok=True)
    sources=[('qubodup',167219,'train'),('ARodLRU2018',662318,'train'),
             ('George_Arkin',188622,'val'),('anagar',267966,'test')]
    known=set()
    for path in Path('dataset/test_only_reserve/FSD50K_metadata').rglob('*clips_info_FSD50K.json'):known|=set(json.loads(path.read_text()))
    reserve=json.loads(Path('dataset/test_only_reserve/FSD50K_mixed/manifest.json').read_text())
    authors={r['clip_metadata']['uploader'].lower() for r in reserve}
    def one(item):
        author,id,split=item;assert str(id) not in known and author.lower() not in authors
        page=f'https://freesound.org/people/{author}/sounds/{id}/';html_path=meta/f'{id}.html'
        if not html_path.exists():html_path.write_bytes(read_url(page,limit=2_000_000))
        html=html_path.read_text();assert 'publicdomain/zero' in html
        urls=re.findall(r'https://cdn\.freesound\.org/previews/[^\"\s<>]+-hq\.mp3',html);assert urls
        path=download(urls[0],folder/(str(id)+'.mp3'))
        return {'path':str(path.resolve()),'domain':name,'label':0,'split':split,
                'source_group':'uploader:'+author,'source_url':page,'preview_url':urls[0],
                'freesound_id':str(id),'uploader':author,'license':'CC0',
                'label_basis':'Author describes recorded plastic/paper bag popping; not firearm discharge.'}
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,sources))
    (folder/'raw_manifest.json').write_text(json.dumps(rows,indent=2))
    (folder/'provenance.json').write_text(json.dumps({'source':'Original author pages saved in source_metadata',
        'sources':sources,'split':'Uploader-separated; two plastic bag recordings train, one paper bag validation, one plastic bag test.',
        'overlap':'Exclude all FSD50K source IDs and all reserved FSD50K uploaders. Decoded-audio hashes checked by trainer.',
        'limitation':'Four independent original recordings only; augmentation is not additional independent data.'},indent=2))
    print('READY IndependentBagPops',len(rows),flush=True)


if __name__=='__main__':main()
