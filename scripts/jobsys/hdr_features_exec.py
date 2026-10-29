#!/usr/bin/env python3
"""Native PQ feature transport. Zenfleet owns claims, retries and CAS receipts.

The declared extractor owns all decoding and feature arithmetic. No targets
are read. This capability is separate from RGB8 and historical Rev5 banks.
"""
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

CAPABILITY = 'hdr-native-rev7-features-v1'
UPIQ_CAPABILITY = 'hdr-upiq380-rev7-features-v1'
IDS = list(range(228)) + list(range(372, 720))
FEATURE_ID = 'basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate(job):
    cell = job['cell']
    if cell['codec'] != 'hdr:features' or job['kind'] != {'kind': 'bake', 'view': CAPABILITY}:
        raise ValueError('unsupported native HDR feature job')
    controls = json.loads(cell['knob_tuple_json'])
    if set(controls) != {'reference_sha256', 'candidates', 'extractor_sha256', 'extractor_build_commit', 'executor_sha256', 'input_contract', 'candidate_paths'}:
        raise ValueError('unknown/missing native feature controls')
    contract = controls['input_contract']
    if contract not in {'hdr-common-primaries-v2-cicp-pq10000', 'rousselot-registered-k179-as-is', 'rousselot-registered-k100-as-is'}:
        raise ValueError('unregistered native HDR input contract')
    if contract.startswith('rousselot') and not controls['candidate_paths']:
        raise ValueError('RGBE library inputs must be read in place')
    if controls['candidate_paths']:
        library = Path(os.environ['ZEN_HDR_LIBRARY_DIR']).resolve(strict=True)
        for path in [cell['image_path'], *controls['candidate_paths']]:
            resolved = Path(path).resolve(strict=True)
            if contract.startswith('rousselot'):
                allowed = {'.hdr'}
            elif path == cell['image_path']:
                allowed = {'.png'}
            else:
                # one candidate encoding per job: JXL renditions or exact PQ16 PNG ladder renditions
                allowed = {Path(controls['candidate_paths'][0]).suffix.lower()} & {'.jxl', '.png'}
            if not resolved.is_relative_to(library) or resolved.suffix.lower() not in allowed:
                raise ValueError('native input must be read in place under the declared root')
    candidates = controls['candidates']
    if not candidates or len(candidates) > 128 or len(set(candidates)) != len(candidates):
        raise ValueError('invalid ordered candidate selection')
    if controls['candidate_paths'] and len(controls['candidate_paths']) != len(candidates):
        raise ValueError('native library path coverage differs')
    digests = [controls['reference_sha256'], *candidates, controls['extractor_sha256'], controls['executor_sha256']]
    if any(len(d) != 64 or any(c not in '0123456789abcdef' for c in d) for d in digests):
        raise ValueError('invalid feature input digest')
    required = {*digests, sha(canonical(controls).encode()), sha(controls['extractor_build_commit'].encode())}
    if not required <= set(job['inputs']):
        raise ValueError('feature controls and implementations must be hash-bound')
    if sha(Path(__file__).read_bytes()) != controls['executor_sha256']:
        raise ValueError('feature executor changed')
    binary = Path(os.environ['ZEN_HDR_BIN_DIR']) / 'hdr944_extract'
    if sha(binary.read_bytes()) != controls['extractor_sha256']:
        raise ValueError('native feature extractor changed')
    reference = Path(cell['image_path'])
    if sha(reference.read_bytes()) != controls['reference_sha256']:
        raise ValueError('feature reference changed')
    return controls, binary, reference


def verify_native_producer(manifest, controls):
    research = manifest['research']
    if (research['feature_set_id'] != FEATURE_ID or research.get('layout_width') != 1825
            or research['formula_revision'] != 'Rev7'
            or research.get('zensim_build_commit') != controls['extractor_build_commit']):
        raise ValueError('native research producer identity differs')


def execute_upiq(job):
    cell = job['cell']
    if cell['codec'] != 'hdr:upiq380-features' or job['kind'] != {'kind': 'bake', 'view': UPIQ_CAPABILITY}:
        raise ValueError('unsupported UPIQ native feature job')
    controls = json.loads(cell['knob_tuple_json'])
    if set(controls) not in ({'admission_sha256', 'extractor_sha256', 'extractor_build_commit', 'executor_sha256'},
                              {'admission_sha256', 'extractor_sha256', 'extractor_build_commit', 'executor_sha256', 'stored_extraction'}):
        raise ValueError('unknown UPIQ feature controls')
    admission = Path(cell['image_path'])
    raw = admission.read_bytes()
    if sha(raw) != controls['admission_sha256']:
        raise ValueError('UPIQ admission bytes changed')
    a = json.loads(raw)
    if (a['schema'] != 'upiq380-extraction-admission-rev7-v1' or a['role'] != 'train'
            or a['tier'] != 'T2' or a['formula_revision'] != 7 or a['requested_ids'] != IDS
            or len(a['rows']) != 380):
        raise ValueError('unregistered UPIQ feature scope')
    required = {controls['admission_sha256'], controls['extractor_sha256'], controls['executor_sha256'],
                sha(canonical(controls).encode()), sha(controls['extractor_build_commit'].encode())}
    stored = controls.get('stored_extraction')
    if stored:
        if set(stored) != {'root', 'sha256', 'failed_job_id'} or set(stored['sha256']) != {'features.tsv', 'features.tsv.manifest.json', 'extract.log'}:
            raise ValueError('incomplete stored extraction recovery receipt')
        required.update(stored['sha256'].values())
        required.add(stored['failed_job_id'])
    for row in a['rows']:
        required.update([row['reference_sha256'], row['distorted_sha256']])
    if not required <= set(job['inputs']) or sha(Path(__file__).read_bytes()) != controls['executor_sha256']:
        raise ValueError('unbound UPIQ controls or original inputs')
    binary = Path(os.environ['ZEN_HDR_BIN_DIR']) / 'upiq_pu_score'
    if sha(binary.read_bytes()) != controls['extractor_sha256']:
        raise ValueError('UPIQ native extractor changed')
    staging = Path(os.environ['ZEN_ARCHIVE_BLOBS_DIR']) / '.staging'
    staging.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='upiq-native-', dir=staging))
    output = stage / 'features.tsv'
    env = {k: v for k, v in os.environ.items() if k not in {'ZENSIM_FEATCANON', 'ZENSIM_MAX_TIER'}}
    env.update(ZENSIM_FORMULA_REV='7', ZENSIM_ROOT_FORM='sqrt', RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1')
    if stored:
        # Explicit recovery copies verified generated bytes; no dataset payload
        # is reopened and no feature is recomputed or numerically rewritten.
        for name, digest in stored['sha256'].items():
            path = Path(stored['root']) / name
            if sha(path.read_bytes()) != digest:
                raise ValueError('stored native extraction changed')
            shutil.copyfile(path, stage / name)
    else:
        with (stage / 'extract.log').open('wb') as log:
            subprocess.run([str(binary), '--training-allowlist', str(admission), '--allowlist-sha256', controls['admission_sha256'],
                            '--out', str(output)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest = json.loads(Path(str(output)+'.manifest.json').read_text())
    verify_native_producer(manifest, controls)
    if (manifest['formula_revision'] != 7 or manifest['rows'] != 380 or manifest['requested_ids'] != IDS
            or manifest['allowlist_sha256'] != controls['admission_sha256']):
        raise ValueError('UPIQ native producer differs')
    with output.open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    if [r['condition_id'] for r in rows] != [r['condition_id'] for r in a['rows']]:
        raise ValueError('UPIQ native condition coverage/order differs')
    if any(not math.isfinite(float(row['f'+str(i)])) for row in rows for i in IDS):
        raise ValueError('nonfinite native UPIQ feature')
    sidecars = list(stage.iterdir())
    (stage / 'result.json').write_text(canonical(dict(schema=UPIQ_CAPABILITY, controls=controls, image_path=cell['image_path'], knob_tuple_json=cell['knob_tuple_json'],
        feature_set_id=FEATURE_ID, rows=380, labels_read=False, stored_extraction_reused=bool(stored), sidecars={p.name: sha(p.read_bytes()) for p in sidecars})))
    return b'ZFEVIFSTAGE1\n'+canonical(dict(root=str(stage), primary='result.json',
        objects=[dict(path=p.name, sha256=sha(p.read_bytes())) for p in sidecars])).encode()


def execute(job):
    if job.get('kind') == {'kind': 'bake', 'view': UPIQ_CAPABILITY}:
        return execute_upiq(job)
    controls, binary, reference = validate(job)
    archive = Path(os.environ['ZEN_ARCHIVE_BLOBS_DIR'])
    staging = archive / '.staging'
    staging.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='hdr-features-', dir=staging))
    paths = []
    for index, digest in enumerate(controls['candidates']):
        if controls['candidate_paths']:
            path = Path(controls['candidate_paths'][index])
            if sha(path.read_bytes()) != digest:
                raise ValueError('native library candidate hash differs')
        else:
            data = (archive / digest[:2] / digest[2:4] / digest).read_bytes()
            if sha(data) != digest:
                raise ValueError('native PQ candidate CAS hash differs')
            path = stage / (digest + '.png')
            path.write_bytes(data)
        paths.append(path)
    pairs = stage / 'pairs.tsv'
    with pairs.open('w') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(['q', 'ref_path', 'dist_path'])
        for index, path in enumerate(paths):
            writer.writerow([index, reference, path])
    output = stage / 'features.tsv'
    args = [str(binary), '--pairs', str(pairs), '--enc-root', str(stage), '--ref-root', str(reference.parent),
            '--absolute-pairs', '--out', str(output), '--threads', '1', '--requested-ids', ','.join(map(str, IDS)),
            '--input-contract', controls['input_contract'], '--era-label', 'rev7_rowlanes']
    if not controls['candidate_paths'] or paths[0].suffix.lower() == '.png':
        args.append('--pq-png-distortions')
    env = {k: v for k, v in os.environ.items() if k not in {'ZENSIM_FEATCANON', 'ZENSIM_MAX_TIER'}}
    env.update(ZENSIM_FORMULA_REV='7', ZENSIM_ROOT_FORM='sqrt', RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1')
    with (stage / 'extract.log').open('wb') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    manifest = json.loads(output.with_suffix('.manifest.json').read_text())
    if manifest['formula_revision'] != 'Rev7' or manifest['era_label'] != 'rev7_rowlanes' or manifest['requested_ids'] != IDS:
        raise ValueError('native extraction contract differs')
    verify_native_producer(manifest, controls)
    if manifest['feature_count'] != 1825:
        raise ValueError('native feature layout width differs')
    with output.open() as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        rows = list(reader)
    if len(rows) != len(paths) or [int(r['q']) for r in rows] != list(range(len(paths))):
        raise ValueError('native feature coverage/order differs')
    if any(not math.isfinite(float(row['f'+str(i)])) for row in rows for i in IDS):
        raise ValueError('nonfinite measured native HDR feature')
    sidecars = [p for p in stage.iterdir() if p.is_file() and p.suffix != '.png']
    result = dict(schema=CAPABILITY, controls=controls, image_path=job['cell']['image_path'], knob_tuple_json=job['cell']['knob_tuple_json'], feature_set_id=FEATURE_ID, measured_ids=IDS,
                  feature_sha256=sha(output.read_bytes()), rows=len(rows), labels_read=False,
                  sidecars={p.name: sha(p.read_bytes()) for p in sidecars})
    (stage / 'result.json').write_text(canonical(result))
    return b'ZFEVIFSTAGE1\n' + canonical(dict(root=str(stage), primary='result.json',
            objects=[dict(path=p.name, sha256=sha(p.read_bytes())) for p in sidecars])).encode()


def main():
    if sys.argv[1:] == ['capabilities']:
        selected = os.environ.get('ZEN_HDR_CAPABILITIES', CAPABILITY + ',' + UPIQ_CAPABILITY).split(',')
        binaries = {CAPABILITY: 'hdr944_extract', UPIQ_CAPABILITY: 'upiq_pu_score'}
        for capability in selected:
            if capability not in binaries or not (Path(os.environ['ZEN_HDR_BIN_DIR']) / binaries[capability]).is_file():
                raise ValueError('declared native HDR capability has no baked binary')
        print('\n'.join(selected))
    elif sys.argv[1:] == ['--serve']:
        while True:
            length = sys.stdin.buffer.read(4)
            if not length:
                return
            if len(length) != 4:
                raise ValueError('truncated frame')
            count = struct.unpack('<I', length)[0]
            if count > 1024*1024:
                raise ValueError('oversized feature request')
            data = sys.stdin.buffer.read(count)
            if len(data) != count:
                raise ValueError('truncated request')
            try:
                payload, status = execute(json.loads(data)), 0
            except Exception as error:
                payload, status = str(error).encode(), 1
            sys.stdout.buffer.write(bytes([status]) + struct.pack('<I', len(payload)) + payload)
            sys.stdout.buffer.flush()
    else:
        sys.stdout.buffer.write(execute(json.load(sys.stdin)))


if __name__ == '__main__':
    main()
