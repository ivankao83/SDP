"""Inspect/download a bounded subset of the published EDGE gunshot archive."""
import json,zipfile
from pathlib import Path
from prepare_false_alarm_data import RemoteZip
root=Path('dataset/continual_cycles')
url='https://zenodo.org/records/7004819/files/edge-collected-gunshot-audio.zip?download=1'
with zipfile.ZipFile(RemoteZip(url)) as z:
 for i in z.infolist():
  if not i.filename.lower().endswith(('.wav','/')):
   print(i.filename,i.file_size,flush=True)
   if i.file_size < 10_000_000:
    out=root/'EDGE_metadata'/Path(i.filename).name
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(z.read(i))
