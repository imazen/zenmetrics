#!/usr/bin/env python3
"""One zenfleet native-HDR view; stage AVIF, PQ16 rendition and exact labels.

The normal zenfleet worker owns claims, retries, CAS publication and DONE.
No EVIF code or container is used. Runtime and byte pins come from the frozen
scientific declaration; the executor never chooses sources or admission rules.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

CAPABILITIES = ['hdr-chroma-view-v1', 'ssim2-natural-hdr-b7c2b4b3']
DELTAS = [0, 8, 16, 24, 32, 40, 48, 56, 63]
QUALITIES = [95, 85, 75, 60]
CONTROLS = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate(job):
    cell = job['cell']
    if cell['codec'] != 'hdr-chroma:avif-svt' or job['kind'] != {'kind': 'bake', 'view': 'hdr-chroma-view-v1'}:
        raise ValueError('unsupported native HDR view')
    knobs = json.loads(cell['knob_tuple_json'])
    if set(knobs) != {'control', 'width', 'height', 'source_sha256', 'source_identity', 'binaries', 'executor_sha256'}:
        raise ValueError('unknown/missing HDR controls')
    if type(knobs['control']) is not bool:
        raise ValueError('control selection must be explicit')
    raw = Path(cell['image_path']).read_bytes()
    if raw[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('prepared native PNG required')
    w, h = struct.unpack('>II', raw[16:24])
    if not w*h or w*h >= 1_000_000 or (w, h) != (knobs['width'], knobs['height']):
        raise ValueError('invalid HDR view geometry')
    if raw[24:26] != bytes([16, 2]):
        raise ValueError('native RGB16 required')
    required = {sha(raw), sha(canonical(knobs).encode()), sha(knobs['source_identity'].encode()), knobs['executor_sha256'], *knobs['binaries'].values()}
    if sha(raw) != knobs['source_sha256'] or not required <= set(job['inputs']):
        raise ValueError('source, controls and implementations must be hash-bound')
    if set(knobs['binaries']) != {'chroma-hdr-render', 'chroma-natural-hdr'}:
        raise ValueError('missing renderer/teacher pin')
    if sha(Path(__file__).read_bytes()) != knobs['executor_sha256']:
        raise ValueError('executor source differs from declaration')
    bins = Path(os.environ['ZEN_HDR_BIN_DIR'])
    for name, digest in knobs['binaries'].items():
        if sha((bins/name).read_bytes()) != digest:
            raise ValueError('HDR binary pin mismatch: '+name)
    return knobs, raw, bins


def execute(job):
    knobs, reference_bytes, bins = validate(job)
    scratch = Path(os.environ['TMPDIR'])
    if not scratch.is_absolute() or str(scratch).startswith(('/tmp', '/run', '/dev/shm')):
        raise ValueError('explicit disk-backed scratch required')
    archive = Path(os.environ.get('ZEN_ARCHIVE_BLOBS_DIR', scratch))
    stage_base = archive/'.staging'
    stage_base.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='hdr-view-', dir=stage_base))

    def run(name, args, data=None):
        with (stage/(name+'.stderr')).open('wb') as err:
            p = subprocess.run(list(map(str, args)), input=data, stdout=subprocess.PIPE, stderr=err)
        if p.returncode:
            raise RuntimeError(f'{name} failed; retained {stage}: '+(stage/(name+'.stderr')).read_text(errors='replace'))
        return p.stdout

    reference = Path(job['cell']['image_path'])
    out = stage/'render'
    run('render', [bins/'chroma-hdr-render', 'a3' if knobs['control'] else 'batch', reference, out])
    records = json.loads((out/'encodes.json').read_text())
    expected = {(q, d) for q in QUALITIES for d in DELTAS}
    if knobs['control']:
        expected.update((q, 0) for q in CONTROLS)
    actual = [(int(r['quality']), r['chroma_q'][0]) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('rendered HDR grid differs from declaration')
    requests = []
    for row in records:
        q, delta = int(row['quality']), row['chroma_q'][0]
        if row['chroma_q'] != [delta, delta] or row['bit_depth'] != 10 or row['cicp'] != [9, 16, 6, True]:
            raise ValueError('HDR recipe/signaling differs')
        candidate = (out/row['file']).with_suffix('.pq.png')
        requests.append(dict(reference=str(reference), distorted=str(candidate)))
    results = run('teacher', [bins/'chroma-natural-hdr'], ''.join(canonical(r)+'\n' for r in requests).encode())
    scores = [json.loads(line) for line in results.splitlines()]
    if len(scores) != len(records):
        raise ValueError('teacher lost HDR identities')
    cells = []
    for row, request, score in zip(records, requests, scores):
        if (score['reference'], score['distorted']) != (request['reference'], request['distorted']):
            raise ValueError('HDR teacher pair differs')
        value = score['value']
        if not math.isfinite(value) or struct.pack('>d', value).hex() != score['value_bits']:
            raise ValueError('HDR teacher double transport differs')
        encoded, rendition = (out/row['file']).read_bytes(), Path(request['distorted']).read_bytes()
        cells.append(dict(quality=int(row['quality']), chroma_q=row['chroma_q'], bit_depth=10,
                          cicp=row['cicp'], encoded_sha256=sha(encoded), rendition_sha256=sha(rendition),
                          encoded_bytes=len(encoded), rendition_bytes=len(rendition),
                          natural_ssim2=value, value_bits=score['value_bits'], implementation=score['implementation']))
    (stage/'source.pq.png').write_bytes(reference_bytes)
    manifest = dict(schema='hdr-chroma-view-v1', image_path=str(reference),
                    knob_tuple_json=job['cell']['knob_tuple_json'], source_sha256=sha(reference_bytes),
                    source_identity=knobs['source_identity'], binaries=knobs['binaries'],
                    executor_sha256=knobs['executor_sha256'], control=knobs['control'], cells=cells)
    (stage/'result.json').write_text(canonical(manifest)+'\n')
    objects = []
    # Persist only stable science objects. Failed attempts retain the complete stage.
    for path in [stage/'source.pq.png', *sorted(out.glob('*.avif')), *sorted(out.glob('*.png')), out/'encodes.json']:
        objects.append(dict(path=str(path.relative_to(stage)), sha256=sha(path.read_bytes())))
    envelope = dict(root=str(stage), primary='result.json', objects=objects)
    return b'ZFEVIFSTAGE1\n'+canonical(envelope).encode()


def main():
    if sys.argv[1:] == ['capabilities']:
        print('\n'.join(CAPABILITIES))
    elif sys.argv[1:] == ['--serve']:
        source, target = sys.stdin.buffer, sys.stdout.buffer
        while True:
            length = source.read(4)
            if not length:
                return
            if len(length) != 4:
                raise ValueError('truncated HDR frame')
            count = struct.unpack('<I', length)[0]
            if count > 1024*1024:
                raise ValueError('oversized HDR frame')
            data = source.read(count)
            if len(data) != count:
                raise ValueError('truncated HDR request')
            try:
                payload, status = execute(json.loads(data)), 0
            except Exception as error:
                payload, status = str(error).encode(), 1
            target.write(bytes([status])+struct.pack('<I', len(payload))+payload)
            target.flush()
    else:
        sys.stdout.buffer.write(execute(json.load(sys.stdin)))


if __name__ == '__main__':
    main()
