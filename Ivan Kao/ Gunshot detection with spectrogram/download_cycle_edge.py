"""Download a preselected, recording-session-separated EDGE subset via ZIP ranges."""
import concurrent.futures,csv,hashlib,json,random,zipfile,struct,time,zlib,urllib.error
from collections import defaultdict
from pathlib import Path
from prepare_false_alarm_data import RemoteZip, request
ROOT=Path('dataset/continual_cycles')
URL='https://zenodo.org/records/7004819/files/edge-collected-gunshot-audio.zip?download=1'
meta=list(csv.DictReader((ROOT/'EDGE_metadata/gunshot-audio-all-metadata.csv').open()))
inv=json.loads((ROOT/'EDGE_zip_inventory.json').read_text())
byname={Path(r['name']).stem:r for r in inv if r['name'].endswith('.wav')}
groups=defaultdict(list)
for r in meta:
 if r['filename'] in byname:groups[r['uuid']].append(r)
strata=defaultdict(list)
for uuid,rows in groups.items():strata[(rows[0]['recording_session '],rows[0]['caliber'])].append(uuid)
rng=random.Random(505);selected=[]
for (session,caliber),ids in sorted(strata.items()):
 ids=sorted(ids);rng.shuffle(ids);ids=ids[:30]
 for index,uuid in enumerate(ids):
  rows=groups[uuid]
  # One channel/clip per original recording; mean channel when available.
  row=min(rows,key=lambda r:(0 if '_mean_' in r['filename'] else 1,r['filename']))
  split='test' if session=='2' else 'val' if index%5==0 else 'train'
  selected.append({'path':str((ROOT/'EDGE'/Path(byname[row['filename']]['name']).name).resolve()),
                   'archive_member':byname[row['filename']]['name'],'label':1,'category':caliber,
                   'source_group':'EDGE:'+uuid,'split':split,'recording_session':session,
                   'device':row['device_name'],'raw_metadata':row,'bytes':byname[row['filename']]['size']})
assert sum(r['bytes'] for r in selected)<300_000_000
(ROOT/'EDGE_selection_before_download.json').write_text(json.dumps(selected,indent=2))
print('Prespecified',len(selected),'files',sum(r['bytes'] for r in selected)/1e6,'MB',flush=True)
# Each worker owns a separate seekable stream and ZIP object.
chunks=[selected[i::2] for i in range(2)]
def retry_request(headers):
 for attempt in range(6):
  try:return request(URL, headers)
  except urllib.error.HTTPError as error:
   if error.code not in [429,500,502,503,504] or attempt==5:raise
   delay=min(45,10*(attempt+1))
   print('Server rate limit/transient error; retry in',delay,'seconds',flush=True)
   time.sleep(delay)
# Reuse the central directory locally to avoid several HTTP requests per WAV.
for attempt in range(6):
 try:
  remote=RemoteZip(URL)
  with zipfile.ZipFile(remote) as archive:
   info={i.filename:i for i in archive.infolist()}
  length=remote.length
  break
 except urllib.error.HTTPError as error:
  if error.code!=429 or attempt==5:raise
  print('Waiting for archive server rate limit',flush=True);time.sleep(30)
def fetch(rows):
  for index,row in enumerate(rows):
   path=Path(row['path']);path.parent.mkdir(exist_ok=True)
   item=info[row['archive_member']]
   if not path.exists():
    start=item.header_offset;end=min(length-1,start+item.compress_size+4096)
    with retry_request({'Range':f'bytes={start}-{end}'}) as response:
     assert response.status==206
     assert response.headers['Content-Range']==f'bytes {start}-{end}/{length}'
     payload=response.read(end-start+2)
    assert len(payload)==end-start+1 and payload[:4]==b'PK\x03\x04'
    header=struct.unpack('<4s5H3I2H',payload[:30])
    offset=30+header[-2]+header[-1]
    compressed=payload[offset:offset+item.compress_size]
    assert len(compressed)==item.compress_size
    data=zlib.decompress(compressed,-15) if item.compress_type==zipfile.ZIP_DEFLATED else compressed
    assert len(data)==item.file_size and zlib.crc32(data)==item.CRC
    path.write_bytes(data)
    time.sleep(1)
   data=path.read_bytes();assert len(data)==row['bytes']
   assert zlib.crc32(data)==item.CRC
   row['sha256']=hashlib.sha256(data).hexdigest()
   if index%10==0:print('Downloaded worker progress',index+1,'/',len(rows),flush=True)
  return rows
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 results=list(pool.map(fetch,chunks))
(ROOT/'EDGE_manifest.json').write_text(json.dumps([r for batch in results for r in batch],indent=2))
print('EDGE DOWNLOAD COMPLETE',flush=True)
