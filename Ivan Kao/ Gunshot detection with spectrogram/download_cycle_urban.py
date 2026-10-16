"""Prespecify a compact source-disjoint UrbanSound8K subset, excluding ESC-50 IDs."""
import concurrent.futures,csv,hashlib,json,random,time,urllib.request,urllib.error
from collections import defaultdict
from pathlib import Path
ROOT=Path('dataset/continual_cycles')
rows=[]
for name in ['urban_train.csv','urban_test.csv']:
 rows+=list(csv.DictReader((ROOT/name).open()))
rows=list({r['path']:r for r in rows}.values())
old={r['source_id'] for r in csv.DictReader(Path('dataset/esc50_manifest.csv').open())}
strata=defaultdict(lambda:defaultdict(list));excluded=[]
for r in rows:
 source=Path(r['path']).stem.split('-')[0]
 if source in old:excluded.append(r);continue
 fold=int(r['path'].split('/')[1][4:]);split='test' if fold>=9 else 'val' if fold==8 else 'train'
 strata[(split,r['classname'])][source].append(r)
rng=random.Random(808);selected=[]
for (split,category),sources in sorted(strata.items()):
 ids=sorted(sources);rng.shuffle(ids)
 limit=len(ids) if category=='gun shot' else {'train':20,'val':5,'test':10}[split]
 for src in ids[:limit]:
  row=rng.choice(sorted(sources[src],key=lambda r:r['path']))
  selected.append({'path':str((ROOT/'UrbanSound8K'/Path(row['path']).name).resolve()),
                   'download_path':row['path'],'source_group':'freesound:'+src,'freesound_id':src,
                   'label':int(category=='gun shot'),'category':category,'split':split,
                   'source_url':'https://urbansounddataset.weebly.com/urbansound8k.html',
                   'mirror':'https://huggingface.co/datasets/MahiA/UrbanSound8K'})
(ROOT/'UrbanSound8K_selection_before_download.json').write_text(json.dumps(selected,indent=2))
(ROOT/'UrbanSound8K_ESC50_exclusions.json').write_text(json.dumps(excluded,indent=2))
print('Preselected',len(selected),'different Freesound sources; excluded',len(excluded),'ESC-50-related clips.',flush=True)
def fetch(row):
 path=Path(row['path']);path.parent.mkdir(exist_ok=True)
 if not path.exists():
  for attempt in range(5):
   try:
    url='https://huggingface.co/datasets/MahiA/UrbanSound8K/resolve/main/'+row['download_path']
    with urllib.request.urlopen(url,timeout=60) as response:data=response.read(8_000_001)
    if len(data)>8_000_000:raise ValueError('Unexpectedly large WAV')
    path.write_bytes(data);break
   except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError) as error:
    if attempt==4:raise
    time.sleep(min(30,2**attempt))
  time.sleep(.3)
 row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 row['bytes']=path.stat().st_size
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 result=[]
 for i,row in enumerate(pool.map(fetch,selected)):
  result.append(row)
  if (i+1)%50==0:print('UrbanSound8K downloaded',i+1,'/',len(selected),flush=True)
(ROOT/'UrbanSound8K_manifest.json').write_text(json.dumps(result,indent=2))
print('URBANSOUND DOWNLOAD COMPLETE',sum(r['bytes'] for r in result)/1e6,'MB',flush=True)
