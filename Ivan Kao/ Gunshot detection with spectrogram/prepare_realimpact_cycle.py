"""Small measured-impact subset; read only prefixes of the public compressed arrays."""
import hashlib
import io
import json
import struct
import urllib.request
import zlib
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

import numpy as np
import soundfile as sf
from prepare_five_cycles import read_url,remote_archive

ROOT=Path('dataset/run22_realimpact')
OBJECTS=[('73_PlasticBin','train'),('48_PlasticBowl','train'),('27_WoodPlate','train'),
         ('86_MetalHoledSpoon','val'),('49_PlasticBowl','test'),('22_Cup','test')]


def one(spec):
    obj,split=spec;folder=ROOT/'RealImpactMeasured';folder.mkdir(exist_ok=True)
    output=ROOT/(obj+'_manifest.json')
    if output.exists():return json.loads(output.read_text())
    url=f'https://downloads.cs.stanford.edu/viscam/RealImpact/{obj}.zip'
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url+f'?sdp22_size={attempt}',method='HEAD'),timeout=40) as response:
                size=int(response.headers['Content-Length'])
            break
        except OSError:
            if attempt==4:raise
            time.sleep(2**attempt)
    with remote_archive({'download_url':url+'?sdp22_inventory=1','size':size}) as archive:
        info=next(i for i in archive.infolist() if i.filename.endswith('/deconvolved_0db.npy'))
        metadata={}
        for key in ['micID','angle','distance','vertexID']:
            member=next(i for i in archive.infolist() if i.filename.endswith('/'+key+'.npy'))
            metadata[key]=np.load(io.BytesIO(archive.read(member))).tolist()
    offset=info.header_offset
    header=read_url(url+'?sdp22_header=1',offset,offset+1023,limit=1024)
    assert header[:4]==b'PK\x03\x04'
    filename,extra=struct.unpack_from('<HH',header,26);position=offset+30+filename+extra
    decoder=zlib.decompressobj(-15);parts=[];count=0;needed=None;audit=[];array_header=None
    while needed is None or count<needed:
        right=min(position+3_999_999,offset+30+filename+extra+info.compress_size-1)
        data=read_url(url+f'?sdp22_prefix={position}',position,right,limit=4_000_000)
        assert len(data)==right-position+1
        audit.append({'offset':position,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        decoded=decoder.decompress(data,0 if needed is None else needed-count)
        parts.append(decoded);count+=len(decoded);position=right+1
        if needed is None:
            stream=io.BytesIO(decoded);version=np.lib.format.read_magic(stream)
            shape,fortran,dtype=np.lib.format._read_array_header(stream,version)
            assert not fortran and len(shape)==2 and shape[0]>=100
            array_header=stream.tell();needed=array_header+100*shape[1]*dtype.itemsize
        print('PREFIX',obj,round(count/needed*100,1),'%',flush=True)
    data=b''.join(parts);array=np.frombuffer(data,dtype=dtype,count=100*shape[1],offset=array_header).reshape(100,shape[1])
    rows=[]
    for i,audio in enumerate(array):
        assert np.isfinite(audio).all() and np.max(np.abs(audio))>0
        path=folder/(obj+f'_{i:03d}.wav')
        if not path.exists():sf.write(path,audio/max(float(np.max(np.abs(audio))),1e-12),48000,subtype='FLOAT')
        rows.append({'path':str(path.resolve()),'domain':'RealImpactMeasured','label':0,'split':split,
                     'source_group':'object:'+obj,'original_array_row':i,
                     'source_url':'https://github.com/samuel-clarke/RealImpact',
                     'object':obj,'microphone':metadata['micID'][i],'angle':metadata['angle'][i],
                     'distance':metadata['distance'][i],'vertex_id':metadata['vertexID'][i],
                     'preprocessing':'Author hammer-deconvolved measured audio; per-clip peak normalization; original 48kHz rate.'})
    (ROOT/(obj+'_provenance.json')).write_text(json.dumps({'source':url,'archive_size':size,
            'member':info.filename,'member_crc32_expected_but_not_fully_verified':info.CRC,
            'array_shape':shape,'dtype':str(dtype),'selected_rows':[0,100],'byte_ranges':audit,
            'integrity':'Each downloaded range length and SHA256 recorded; complete ZIP/member checksum cannot be verified from a partial prefix.',
            'limitation':'Multiple microphone channels share hammer strikes; objects, not clips, define the split.'},indent=2))
    output.write_text(json.dumps(rows,indent=2));print('READY OBJECT',obj,len(rows),flush=True)
    return rows


def main():
    ROOT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=1) as pool:rows=[r for group in pool.map(one,OBJECTS) for r in group]
    folder=ROOT/'RealImpactMeasured'
    (folder/'raw_manifest.json').write_text(json.dumps(rows,indent=2))
    (folder/'provenance.json').write_text(json.dumps({'source':'https://github.com/samuel-clarke/RealImpact',
        'citation':'Clarke et al., RealImpact: A Dataset of Impact Sound Fields for Real Objects, CVPR 2023.',
        'selection':'First 100 measured microphone rows from each of six physical objects, before model scoring.',
        'objects':OBJECTS,'negative_label_basis':'Measured hammer strikes on household objects, no firearm discharge.',
        'split':'Whole physical objects held apart; microphones from a strike never cross splits.',
        'limitation':'600 microphone clips do not represent 600 independent hammer strikes; test has only two physical objects.'},indent=2))
    print('READY REALIMPACT',len(rows),Counter(r['split'] for r in rows),flush=True)


if __name__=='__main__':main()
