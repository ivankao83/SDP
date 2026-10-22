"""Use original recorded phone handclaps, excluding the paired inferred RIRs."""
import csv,hashlib,json,random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import gunshot as g
from continual_cycles import feature,fingerprint
from prepare_independent_pops import get
ROOT=Path('dataset/run46_claprir')


def main():
    torch.set_num_threads(2);folder=ROOT/'ClapRIRPhone';folder.mkdir(exist_ok=True);cache=folder/'features';cache.mkdir(exist_ok=True)
    tree=json.loads((ROOT/'github_tree.json').read_text());revision=tree['sha'];entries={r['path']:r for r in tree['tree']}
    records=[r for r in csv.DictReader((ROOT/'manifest.csv').open()) if r['asset_type']=='phone_recording'];assert len(records)==280
    rooms=sorted({r['room'] for r in records});random.Random(4646).shuffle(rooms)
    assignment={room:'train' if i<4 else 'val' if i<5 else 'test' for i,room in enumerate(rooms)}
    assert len(rooms)==7
    (ROOT/'prespecified_selection.json').write_text(json.dumps({'revision':revision,'rooms':assignment,'asset_ids':[r['asset_id'] for r in records],'only_original_recordings':True,'inferred_rirs_excluded':True},indent=2))
    cfg=torch.load('runs/run14_m16_firecrackers/best.pt',map_location='cpu',weights_only=True)['config']
    def one(r):
        name='docs/static/assets/'+r['export_path'];entry=entries[name];url='https://raw.githubusercontent.com/Akinesia112/ClapRIR/'+revision+'/'+name
        saved=ROOT/(r['asset_id']+'.json')
        if saved.exists():return json.loads(saved.read_text())
        data=get(url,2_000_000)
        assert len(data)==entry['size'];assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==entry['sha']
        assert hashlib.sha256(data).hexdigest()==r['export_sha256']
        temporary=folder/(r['asset_id']+'.download.wav');temporary.write_bytes(data)
        try:audio=g.read_clip(temporary)
        finally:temporary.unlink(missing_ok=True)
        path=folder/(r['asset_id']+'.wav');sf.write(path,audio,g.SR,subtype='FLOAT');digest=hashlib.sha256(audio.tobytes()).hexdigest();fp=cache/(digest+'.npy')
        if not fp.exists():np.save(fp,feature(audio,cfg))
        row={'path':str(path.resolve()),'domain':'ClapRIRPhone','label':0,'split':assignment[r['room']],'source_group':'room:'+r['room'],
            'source_url':'https://github.com/Akinesia112/ClapRIR','raw_url':url,'github_revision':revision,'github_blob_sha1':entry['sha'],'original_export_sha256':r['export_sha256'],
            'original_source_sha256':r['source_sha256'],'room':r['room'],'clap_mode':r['clap_mode'],'repetition':r['repetition'],
            'decoded_sha256':digest,'sha256':fingerprint(path),'feature_path':str(fp.resolve()),'label_basis':'Author-labelled physical phone handclap recording; inferred RIR outputs excluded.',
            'source_release_status':r['release_status'],'processing':'Complete author one-second interval and original support/padding; playback peak scaling retained; resampled mono24k FLOAT.'}
        saved.write_text(json.dumps(row,indent=2));return row
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,row in enumerate(pool.map(one,records),1):
            rows.append(row)
            if i%40==0:print('CLAP DOWNLOAD',i,'/',len(records),flush=True)
    unique={};duplicates=[]
    for r in rows:
        if r['decoded_sha256'] in unique:duplicates.append(r);continue
        unique[r['decoded_sha256']]=r
    rows=list(unique.values());assert len(rows)>=200
    (folder/'manifest.json').write_text(json.dumps(rows,indent=2));(ROOT/'duplicates_excluded.json').write_text(json.dumps(duplicates,indent=2))
    (folder/'provenance.json').write_text(json.dumps({'source_url':'https://github.com/Akinesia112/ClapRIR','github_revision':revision,'selected_asset_type':'phone_recording',
        'original_selected_events':len(records),'unique_decoded_clips':len(rows),'split_by_room':assignment,'no_inferred_RIRs_counted_as_recordings':True,
        'integrity':'Git blob SHA1 and author export SHA256 verified for every original WAV before resampling.',
        'processing':'Author phone intervals one second, with documented playback gain and source padding; no new excerpts or synthetic variants counted.',
        'licence':'Repository code MIT; author asset manifest states no separate dataset licence asserted. Audio remains local, not in Git.',
        'limitations':'Seven acoustic rooms; phone-event participant identities are not provided, so performer independence is not established. Room-separated evaluation.'},indent=2))
    print('READY CLAP RIR',len(rows),dict(Counter(r['split'] for r in rows)),flush=True)


if __name__=='__main__':main()
