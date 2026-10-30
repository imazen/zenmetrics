#!/usr/bin/env python3
"""One zenfleet trade view: `evif trade` on a prepared view, staged as content-addressed packs.

The worker owns claims, retries, CAS publication and DONE. This executor renders no policy: the
template spec (families, ramps, kinds, DE targets, controls, persistence) and the pins come from the
hash-bound declaration. The job is a `bake` view (`declare-views`), so no Rust declare path is needed.

Output: `ledger.jsonl`, the run receipts, and deterministic tar packs of the `.evif` files, bitstreams
and native maps (each pack under the worker's per-object limit). `result.json` lists every member with
its SHA-256, so each sample stays individually content-addressed inside its pack.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile

CAPABILITIES = ['evif-trade-view-v1']
VIEW_KIND = {'kind': 'bake', 'view': 'evif-trade-view-v1'}
PACK_BYTES = 30 * 1024 * 1024          # the worker refuses objects over 32 MiB
MAX_OBJECTS = 512
STAGED_MAX_BYTES = 1024 * 1024 * 1024  # the worker refuses a job staging more than this in total
POLL_SECONDS = 2.0                     # how often a running trade is checked against the staging limit
TAR_SLACK = 8 * 1024 * 1024            # tar headers and padding on top of the files in the run directory
KNOBS = {'template_path', 'template_sha256', 'view', 'binaries', 'models', 'executor_sha256'}
VIEW_FIELDS = {'id', 'source', 'source_path', 'source_sha256', 'class', 'content_class', 'split', 'role',
               'path', 'sha256', 'seeds'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate(job):
    cell = job['cell']
    if cell['codec'] != 'evif:trade' or job['kind'] != VIEW_KIND:
        raise ValueError('unsupported trade view job')
    knobs = json.loads(cell['knob_tuple_json'])
    if set(knobs) != KNOBS or set(knobs['view']) != VIEW_FIELDS or knobs['view']['split'] != 'train':
        raise ValueError('unknown/missing trade controls or a non-TRAIN source')
    raw = Path(cell['image_path']).read_bytes()
    if raw[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('prepared view PNG required')
    w, h = struct.unpack('>II', raw[16:24])
    if not w * h or w * h >= 1_000_000:
        raise ValueError('view must be below 1MP')
    view = knobs['view']
    if cell['image_path'] != view['path'] or sha(raw) != view['sha256']:
        raise ValueError('view bytes differ from the declaration')
    template = Path(knobs['template_path']).read_bytes()
    if sha(template) != knobs['template_sha256']:
        raise ValueError('template spec differs from the declaration')
    if sha(Path(__file__).read_bytes()) != knobs['executor_sha256']:
        raise ValueError('executor source differs from the declaration')
    bound = {sha(raw), sha(canonical(knobs).encode()), knobs['template_sha256'], knobs['executor_sha256'],
             *knobs['binaries'].values(), *knobs['models'].values()}
    if not bound <= set(job['inputs']):
        raise ValueError('view, template, binary and model pins must be job inputs')
    if set(knobs['binaries']) != {'evif-trade'}:
        raise ValueError('exactly one pinned binary expected')
    return knobs, json.loads(template)


def pinned_paths(knobs):
    binary = Path(os.environ['ZEN_EVIF_BIN_DIR']) / 'evif-trade'
    if sha(binary.read_bytes()) != knobs['binaries']['evif-trade']:
        raise ValueError('binary pin mismatch: evif-trade')
    models = {}
    for name, digest in knobs['models'].items():
        path = Path(os.environ['ZEN_EVIF_MODEL_DIR']) / name
        if sha(path.read_bytes()) != digest:
            raise ValueError('model pin mismatch: ' + name)
        models[name] = path
    return binary, models


def tree_bytes(path):
    total = 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except FileNotFoundError:           # a file the trade run just renamed or removed
                pass
    return total


def run_bounded(cmd, env, out, limit, stderr_path):
    """Run `cmd`, killing it as soon as the run directory `out` outgrows `limit` bytes.

    A view that cannot fit the worker's staging limit is known to be doomed long before it finishes;
    stopping it saves the rest of its compute. Returns (return code, or None when it was stopped; bytes seen).
    """
    with stderr_path.open('wb') as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err, env=env)
        while True:
            try:
                proc.wait(timeout=POLL_SECONDS)
                return proc.returncode, tree_bytes(out)
            except subprocess.TimeoutExpired:
                seen = tree_bytes(out)
                if seen > limit:
                    proc.kill()
                    proc.wait()
                    return None, seen


def pack(files, prefix, stage):
    """Deterministic uncompressed tars of `files` ((arcname, path) pairs), each under PACK_BYTES."""
    packs, members = [], []
    current, size, handle = None, 0, None

    def finish():
        nonlocal handle
        if handle is not None:
            handle.close()
            handle = None

    for arcname, path in sorted(files):
        data = path.read_bytes()
        cost = len(data) + 1024
        if handle is None or size + cost > PACK_BYTES:
            finish()
            current = stage / f'{prefix}-{len(packs):03d}.tar'
            handle = tarfile.open(current, 'w', format=tarfile.PAX_FORMAT)
            packs.append(current)
            size = 0
        info = tarfile.TarInfo(arcname)
        info.size, info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = len(data), 0, 0, 0, '', '', 0o644
        with path.open('rb') as f:
            handle.addfile(info, f)
        size += cost
        members.append(dict(pack=current.name, name=arcname, sha256=sha(data), bytes=len(data)))
    finish()
    return packs, members


def execute(job):
    knobs, template = validate(job)
    binary, models = pinned_paths(knobs)
    scratch = Path(os.environ['TMPDIR'])
    if not scratch.is_absolute() or str(scratch).startswith(('/tmp', '/run', '/dev/shm')):
        raise ValueError('explicit disk-backed scratch required')
    stage_base = Path(os.environ.get('ZEN_ARCHIVE_BLOBS_DIR', scratch)) / '.staging'
    stage_base.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='evif-trade-', dir=stage_base))

    spec = dict(template)
    for forbidden in ('views',):
        if forbidden in spec and spec[forbidden]:
            raise ValueError('template must not carry views')
    spec['views'] = [knobs['view']]
    for candidate in spec.get('candidates', []):
        candidate['model'] = str(models[candidate['model']])      # templates name files, not paths
    spec_path = stage / 'spec.json'
    spec_path.write_text(canonical(spec) + '\n')
    out = stage / 'run'
    env = dict(os.environ, RAYON_NUM_THREADS='1', TMPDIR=str(scratch),
               EVIF_E_MODEL=str(models['rev7_general_h128_2026-10-27.bin']),
               EVIF_ECODEC_MODEL=str(models['rev7_codec_h128_2026-10-27.bin']))
    limit_total = int(os.environ.get('ZEN_STAGED_MAX_TOTAL_BYTES', STAGED_MAX_BYTES))
    code, seen = run_bounded([str(binary), 'trade', str(spec_path), str(out)], env, out,
                             limit_total - min(TAR_SLACK, limit_total // 8), stage / 'trade.stderr')
    if code is None:
        shutil.rmtree(stage)
        raise RuntimeError(f'view output exceeds the worker limits: {seen} bytes written when the run was stopped')
    if code:
        tail = (stage / 'trade.stderr').read_bytes()[-4000:].decode(errors='replace')
        shutil.rmtree(out, ignore_errors=True)          # keep the stage (spec, stderr) but not the partial output
        raise RuntimeError(f'evif trade failed; retained {stage}: {tail}')
    if not (out / 'COMPLETE.json').exists():
        shutil.rmtree(out, ignore_errors=True)
        raise RuntimeError(f'evif trade did not complete; retained {stage}')

    view = knobs['view']
    ledger = (out / 'ledger.jsonl').read_bytes()
    rows = sum(1 for _ in ledger.splitlines())
    packs, members = [], []
    for prefix, files in (
        ('evif', [(f'{path.parent.name}/{path.name}', path) for path in (out / view['id']).glob('*.evif')]),
        ('bitstreams', [(path.name, path) for path in (out / 'bitstreams').iterdir()]),
        ('maps', [(path.name, path) for path in (out / 'maps').iterdir()]),
    ):
        if files:
            made, listed = pack(files, prefix, stage)
            packs += made
            members += [dict(m, kind=prefix) for m in listed]
    plain = [out / 'ledger.jsonl', out / 'SPEC.json', out / 'JUDGES.json', out / 'SOURCE_IDENTITIES.json',
             out / 'CONTROL.json', out / 'METRIC_CONTROL.json', out / 'BYTE_CONTROL.json',
             out / 'COMPLETE.json', out / view['id'] / 'timing.json']
    staged = []
    for path in plain:
        if path.exists():
            dest = stage / ('timing.json' if path.name == 'timing.json' else path.name)
            dest.write_bytes(path.read_bytes())
            staged.append(dest)
    objects = [dict(path=p.name, sha256=sha(p.read_bytes())) for p in staged + packs]
    total = sum((stage / o['path']).stat().st_size for o in objects)
    if len(objects) + 1 > MAX_OBJECTS or total > limit_total:
        shutil.rmtree(stage)
        raise RuntimeError(f'view output ({len(objects)} objects, {total} bytes) exceeds the worker limits')
    manifest = dict(schema='evif-trade-view-v1', image_path=job['cell']['image_path'],
                    knob_tuple_json=job['cell']['knob_tuple_json'], view=view['id'], rows=rows,
                    ledger_sha256=sha(ledger), members=members, objects=objects,
                    binary_sha256=knobs['binaries']['evif-trade'], models=knobs['models'],
                    template_sha256=knobs['template_sha256'], executor_sha256=knobs['executor_sha256'],
                    controls={n: json.loads((out / n).read_text()) for n in
                              ('CONTROL.json', 'METRIC_CONTROL.json', 'BYTE_CONTROL.json') if (out / n).exists()})
    (stage / 'result.json').write_text(canonical(manifest) + '\n')
    # Everything is now packed or copied: drop the run directory so the stage holds the data once.
    shutil.rmtree(out)
    envelope = dict(root=str(stage), primary='result.json', objects=objects)
    return b'ZFEVIFSTAGE1\n' + canonical(envelope).encode()


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
                raise ValueError('truncated trade frame')
            count = struct.unpack('<I', length)[0]
            if count > 1024 * 1024:
                raise ValueError('oversized trade frame')
            data = source.read(count)
            if len(data) != count:
                raise ValueError('truncated trade request')
            try:
                payload, status = execute(json.loads(data)), 0
            except Exception as error:
                payload, status = str(error).encode(), 1
            target.write(bytes([status]) + struct.pack('<I', len(payload)) + payload)
            target.flush()
    else:
        sys.stdout.buffer.write(execute(json.load(sys.stdin)))


if __name__ == '__main__':
    main()
