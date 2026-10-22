"""Prepare fixed external TEST-ONLY reserves. No model inference or training.
Mixed: FSD50K eval subset; positive: HEAR gunshot triangulation;
negative: plastic bag pops. Dataset selection never uses model scores.
"""
import concurrent.futures
import csv
import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
import soundfile as sf
import gunshot as g
import numpy as np

ROOT=Path('dataset/test_only_reserve')


def fetch(url,path,limit=4_000_000,expected_size=None,git_sha=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(5):
        try:
            if not path.exists():
                request=urllib.request.Request(url,headers={'User-Agent':'SDP-test-only-reserve/1.0'})
                with urllib.request.urlopen(request,timeout=60) as response:data=response.read(limit+1)
                if len(data)>limit:raise ValueError('Refusing unexpectedly large download.')
                if expected_size is not None and len(data)!=expected_size:raise ValueError('Download size mismatch.')
                path.write_bytes(data)
            data=path.read_bytes()
            if expected_size is not None and len(data)!=expected_size:raise ValueError('Cached file size mismatch.')
            if git_sha:
                assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==git_sha
            return path
        except (urllib.error.URLError,TimeoutError) as error:
            if attempt==4:raise
            time.sleep(min(30,2**(attempt+1)))


def audio_record(path,dataset,label,url,**extra):
    info=sf.info(path)
    audio=g.read_clip(path)
    return {'path':str(path.resolve()),'dataset':dataset,'role':'test_only','label':label,
            'source_url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'decoded_sha256':hashlib.sha256(audio.tobytes()).hexdigest(),
            'duration_seconds':info.duration,'sample_rate':info.samplerate,'channels':info.channels,
            'bytes':path.stat().st_size,**extra}


def mixed():
    folder=ROOT/'FSD50K_mixed';folder.mkdir(exist_ok=True)
    metadata={}
    for name in ['dev_clips_info_FSD50K.json','eval_clips_info_FSD50K.json']:
        metadata.update(json.loads(next((ROOT/'FSD50K_metadata').rglob(name)).read_text()))
    old={r['source_id'] for r in csv.DictReader(Path('dataset/esc50_manifest.csv').open())}
    for name in ['urban_train.csv','urban_test.csv']:
        old|={Path(r['path']).stem.split('-')[0] for r in csv.DictReader(Path('dataset/continual_cycles',name).open())}
    old_uploaders={metadata[i]['uploader'] for i in old if i in metadata}
    candidates=[r for r in csv.DictReader(next((ROOT/'FSD50K_metadata').rglob('eval.csv')).open())
                if r['fname'] not in old and metadata[r['fname']]['uploader'] not in old_uploaders]
    rng=random.Random(919);rng.shuffle(candidates)
    used=set();selected=[]
    # Prespecified balanced sample, each clip from a different uploader.
    for label in [1,0]:
        count=0
        for row in candidates:
            info=metadata[row['fname']]
            if int('Gunshot_and_gunfire' in row['labels'].split(','))!=label or info['uploader'] in used:continue
            used.add(info['uploader']);selected.append({**row,'label':label,'metadata':info});count+=1
            if count==50:break
        assert count==50
    (folder/'selection_before_inference.json').write_text(json.dumps(selected,indent=2))
    (folder/'overlap_policy.json').write_text(json.dumps({'excluded_known_freesound_ids':len(old),
            'excluded_known_uploaders':len(old_uploaders),'seed':919,'unique_uploaders':len(used),
            'policy':'Exclude all ESC-50 and UrbanSound8K source IDs and associated known FSD50K uploaders. Model scores never used.',
            'limitation':'Original SESA Freesound IDs were not provided, so complete shared-source independence cannot be guaranteed.'},indent=2))
    def one(row):
        source='https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/clips/eval/'+row['fname']+'.wav'
        path=fetch(source,folder/(row['fname']+'.wav'),limit=3_000_000)
        return audio_record(path,'FSD50K_mixed',row['label'],'https://zenodo.org/records/4060432',
                            mirror_url=source,freesound_id=row['fname'],source_group='uploader:'+row['metadata']['uploader'],
                            labels=row['labels'],clip_metadata=row['metadata'],label_basis='Official exhaustively annotated FSD50K evaluation labels.')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows=[]
        for i,row in enumerate(pool.map(one,selected)):
            rows.append(row)
            if (i+1)%20==0:print('FSD50K reserve',i+1,'/100',flush=True)
    (folder/'manifest.json').write_text(json.dumps(rows,indent=2))
    return rows


def positives():
    folder=ROOT/'GunshotTriangulation'
    rows=[]
    for path in sorted(folder.rglob('*.wav')):
        shot=re.search(r'_shot(\d+)_',path.stem).group(1)
        rows.append(audio_record(path,'GunshotTriangulation',1,'https://hearbenchmark.com/hear-tasks.html',
                    original_source='https://doi.org/10.5061/dryad.wm37pvmkc',
                    source_group='shot:'+shot,label_basis='HEAR gunshot event snippets from the Cooper/Shaw outdoor gunshot recordings.'))
    assert len(rows)==88 and len({r['source_group'] for r in rows})==22
    (folder/'manifest.json').write_text(json.dumps(rows,indent=2))
    return rows


def negatives():
    folder=ROOT/'PlasticBagPop';folder.mkdir(exist_ok=True)
    tree=json.loads((ROOT/'bag_tree.json').read_text())
    entries=[r for r in tree['tree'] if r['type']=='blob' and r['path'].lower().endswith('.wav')]
    assert sum(r['size'] for r in entries)<400_000_000
    request=urllib.request.Request('https://api.github.com/repos/rbsingh13/Plastic-Bag-Pop-sounds/commits/main',headers={'User-Agent':'SDP-test-only/1.0'})
    commit=json.load(urllib.request.urlopen(request,timeout=40))
    assert commit['commit']['tree']['sha']==tree['sha'],'Source changed since inventory; inspect again.'
    revision=commit['sha']
    (folder/'selection_before_inference.json').write_text(json.dumps({'commit':revision,'entries':entries},indent=2))
    def one(row):
        url='https://raw.githubusercontent.com/rbsingh13/Plastic-Bag-Pop-sounds/'+revision+'/'+row['path']
        path=fetch(url,folder/Path(row['path']).name,limit=8_000_000,expected_size=row['size'],git_sha=row['sha'])
        return audio_record(path,'PlasticBagPop',0,'https://github.com/rbsingh13/Plastic-Bag-Pop-sounds',
                    source_group=path.stem,git_blob_sha1=row['sha'],source_commit=revision,
                    label_basis='Author describes the collection as plastic-bag-pop sounds, not firearm discharges.',
                    grouping_limitation='Multi-microphone event correspondence is not supplied in the file listing.')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows=[]
        for i,row in enumerate(pool.map(one,entries)):
            rows.append(row)
            if (i+1)%100==0:print('Plastic-bag reserve',i+1,'/',len(entries),flush=True)
    for name in ['README.md','LICENSE']:
        fetch('https://raw.githubusercontent.com/rbsingh13/Plastic-Bag-Pop-sounds/'+revision+'/'+name,folder/('SOURCE_'+name),limit=50_000)
    (folder/'manifest.json').write_text(json.dumps(rows,indent=2))
    return rows


def main():
    rows=[]
    for prepare in [mixed,positives,negatives]:
        new=prepare();rows.extend(new);print('Reserved',new[0]['dataset'],len(new),flush=True)
    # Decoded-audio overlap check includes previous development hashes and all
    # four prior held-out sets. No network is called and no model is loaded.
    import torch
    checkpoint=torch.load('runs/run9_joint_four/best.pt',map_location='cpu',weights_only=True)
    seen=set(checkpoint['seen_hashes'])
    from train_joint_four import load_rows
    seen|={r['decoded_sha256'] for r in load_rows()}
    duplicates=[r for r in rows if r['decoded_sha256'] in seen]
    assert not duplicates, f'{len(duplicates)} reserve recordings overlap prior data; review before accepting.'
    assert len({r['decoded_sha256'] for r in rows})==len(rows),'Duplicate audio inside reserve; review before accepting.'
    (ROOT/'manifest.json').write_text(json.dumps(rows,indent=2))
    registry={'policy':'test_only','training_allowed':False,'validation_allowed':False,
              'threshold_tuning_allowed':False,'checkpoint_selection_allowed':False,
              'predictions_generated':False,'root':'dataset/test_only_reserve',
              'locked_before_first_prediction_checkpoint':'runs/run9_joint_four/best.pt',
              'datasets':[{'name':name,'recordings':sum(r['dataset']==name for r in rows),
                           'type':kind,'role':'test_only','source_url':url}
                for name,kind,url in [('FSD50K_mixed','mixed','https://zenodo.org/records/4060432'),
                  ('GunshotTriangulation','positive_only','https://hearbenchmark.com/hear-tasks.html'),
                  ('PlasticBagPop','negative_only','https://github.com/rbsingh13/Plastic-Bag-Pop-sounds')]]}
    Path('dataset_roles.json').write_text(json.dumps(registry,indent=2))
    Path('reports/run9_joint_four').mkdir(parents=True,exist_ok=True)
    Path('reports/run9_joint_four/test_only_reserves.json').write_text(json.dumps(registry,indent=2))
    print('ALL RESERVES READY',json.dumps(registry),flush=True)


if __name__=='__main__':main()
