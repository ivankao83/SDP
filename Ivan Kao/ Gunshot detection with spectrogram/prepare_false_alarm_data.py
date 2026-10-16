"""Download compact, fixed background-only subsets for TEST ONLY.

MS-SNSD: official GitHub noise_test WAV files, verified against Git blob hashes.
DEMAND: one microphone channel per predeclared environment; HTTP ZIP ranges avoid
downloading all 16 channels. ZIP CRC and WAV validity are checked on extraction.
"""
import argparse
import concurrent.futures
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import soundfile as sf

ROOT = Path('dataset/false_alarm_run4')
DEMAND_ENVIRONMENTS = ['DKITCHEN', 'NPARK', 'OHALLWAY', 'SPSQUARE', 'STRAFFIC', 'TBUS']


def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'SDP-background-evaluation/1.0', 'Accept-Encoding': 'identity', **(headers or {})}), timeout=120)


def audio_record(path, dataset, category, url, **extra):
    info = sf.info(path)
    if info.frames == 0:
        raise ValueError(f'Empty audio: {path}')
    return {'path': str(path.resolve()), 'dataset': dataset, 'category': category,
            'source_url': url, 'assumed_label': 'non_gunshot', 'label_basis': 'Environmental-noise corpus; not individually gunshot-audited.',
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size,
            'duration_seconds': info.duration, 'sample_rate': info.samplerate, 'channels': info.channels, **extra}


def ms_snsd():
    folder = ROOT / 'MS-SNSD'
    folder.mkdir(parents=True, exist_ok=True)
    inventory = folder / 'noise_test_listing.json'
    if not inventory.exists():
        with request('https://api.github.com/repos/microsoft/MS-SNSD/contents/noise_test?ref=master') as response:
            inventory.write_bytes(response.read())
    entries = sorted([r for r in json.loads(inventory.read_text()) if r['name'].lower().endswith('.wav')], key=lambda r: r['name'])
    if not entries or sum(r['size'] for r in entries) > 200_000_000:
        raise ValueError('Unexpected size or missing MS-SNSD subset; inspect inventory.')
    if any(re.search('gun|firework|shoot|explos|artillery', r['name'], re.I) for r in entries):
        raise ValueError('Potential weapon/explosion class in presumed-negative subset; review labels.')
    def fetch(row):
        path = folder / row['name']
        if Path(row['name']).name != row['name']:
            raise ValueError('Unsafe filename.')
        if not path.exists():
            with request(row['download_url']) as response:
                data = response.read(row['size'] + 1)
            if len(data) != row['size']:
                raise ValueError('Unexpected download size.')
            path.write_bytes(data)
        data = path.read_bytes()
        git_hash = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if git_hash != row['sha']:
            raise ValueError(f'Git blob checksum mismatch: {path}')
        return audio_record(path, 'MS-SNSD', row['name'].rsplit('_', 1)[0], row['html_url'], git_blob_sha1=git_hash)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, entries))
    for name in ['LICENSE', 'README.md']:
        with request('https://raw.githubusercontent.com/microsoft/MS-SNSD/master/' + name) as response:
            (folder / ('SOURCE_' + name)).write_bytes(response.read())
    (folder / 'manifest.json').write_text(json.dumps(records, indent=2))
    print(f'MS-SNSD: {len(records)} verified files, {sum(r["bytes"] for r in records)/1e6:.1f} MB, {sum(r["duration_seconds"] for r in records)/60:.1f} minutes', flush=True)


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        super().__init__()
        self.url, self.position, self.transferred = url, 0, 0
        with request(url, {'Range': 'bytes=-65536'}) as response:
            content_range = response.headers.get('Content-Range', '')
            if response.status != 206 or not re.fullmatch(r'bytes \d+-\d+/\d+', content_range):
                raise ValueError('Server does not support byte ranges; refusing an unexpectedly large full download.')
            self.length = int(content_range.split('/')[-1])
            self.tail = response.read(65537)
            if len(self.tail) > 65536:
                raise ValueError('Unexpected range size.')
            self.transferred += len(self.tail)
        self.tail_start = self.length - len(self.tail)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        if self.position < 0:
            raise ValueError('Negative seek.')
        return self.position

    def read(self, size=-1):
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        if size <= 0:
            return b''
        if size > 20_000_000:
            raise ValueError('Unexpectedly large ZIP member read.')
        if self.position >= self.tail_start:
            data = self.tail[self.position - self.tail_start:self.position - self.tail_start + size]
        else:
            start = self.position
            with request(self.url, {'Range': f'bytes={start}-{start + size - 1}'}) as response:
                expected = f'bytes {start}-{start + size - 1}/{self.length}'
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise ValueError('Incorrect HTTP range response.')
                data = response.read(size + 1)
            if len(data) != size:
                raise ValueError('Truncated or oversized HTTP range.')
            self.transferred += len(data)
        self.position += len(data)
        return data


def demand():
    folder = ROOT / 'DEMAND'
    folder.mkdir(parents=True, exist_ok=True)
    def fetch(environment):
        url = f'https://zenodo.org/records/1227121/files/{environment}_16k.zip?download=1'
        path = folder / f'{environment}_ch01.wav'
        info_path = folder / f'{environment}_source.json'
        if path.exists() and info_path.exists():
            old = json.loads(info_path.read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != old['sha256']:
                raise ValueError('Previously downloaded audio checksum changed.')
            return old
        local_zip = folder / f'{environment}_16k.zip'
        remote = None
        if local_zip.exists() and zipfile.is_zipfile(local_zip):
            archive = zipfile.ZipFile(local_zip)
        else:
            remote = RemoteZip(url)
            archive = zipfile.ZipFile(remote)
        with archive:
            matches = [i for i in archive.infolist() if Path(i.filename).name.lower() == 'ch01.wav']
            if len(matches) != 1:
                raise ValueError(f'Unexpected channel names: {[i.filename for i in archive.infolist()][:20]}')
            item = matches[0]
            if item.file_size > 20_000_000:
                raise ValueError('Unexpected selected channel size.')
            data = archive.read(item)  # ZipFile verifies member CRC.
        path.write_bytes(data)
        record = audio_record(path, 'DEMAND', environment, url, original_zip_member=item.filename,
                              zip_member_crc32=f'{item.CRC:08x}', channel_selection='ch01 only',
                              bytes_transferred=remote.transferred if remote else local_zip.stat().st_size,
                              license='CC BY-SA 3.0; see original DEMAND documentation')
        info_path.write_text(json.dumps(record, indent=2))
        print(f'DEMAND {environment}: kept {record["bytes"]/1e6:.1f} MB, {record["duration_seconds"]:.0f} seconds', flush=True)
        return record
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(fetch, DEMAND_ENVIRONMENTS))
    with request('https://zenodo.org/records/1227121/files/DEMAND.pdf?download=1') as response:
        (folder / 'SOURCE_DEMAND.pdf').write_bytes(response.read())
    (folder / 'manifest.json').write_text(json.dumps(records, indent=2))
    print(f'DEMAND: {len(records)} channels, {sum(r["bytes"] for r in records)/1e6:.1f} MB, {sum(r["duration_seconds"] for r in records)/60:.1f} minutes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('dataset', choices=['ms-snsd', 'demand'])
    args = parser.parse_args()
    ms_snsd() if args.dataset == 'ms-snsd' else demand()
