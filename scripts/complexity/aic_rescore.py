#!/usr/bin/env python3
"""Local, resumable, full-precision AIC CSV scoring through zenmetrics compare."""
import argparse
from contextlib import contextmanager
import csv
import fnmatch
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
META = {'distorted', 'source', 'codec', 'codec_acronym', 'distortion_level',
        'bpp', 'codec_id', 'enc_setting', 'enc_value', 'view', 'ptc_subset'}
SCHEMA = 'aic-rescore/1'
# Output selectors are explicit, including versioned in-tree scalar columns.
SELECTORS = {
    'CVVDP': 'cvvdp_cpu_imazen_*_standard_fhd', 'SSIMULACRA2': 'ssim2',
    'FSIM': 'fsimy_imazen_*', 'FSIMc': 'fsimc_imazen_*',
    'IW-SSIM': 'iwssim_piq_imazen_*', 'VIF': 'vifvec_imazen_*',
    'GMSD': 'gmsd_cpu_imazen_*', 'HaarPSI': 'haarpsi_imazen_*',
    'VSI': 'vsi_imazen_*', 'mDCT-PSNR': 'mdctpsnr_imazen_*',
    'SSIM-pyiqa': 'ssim_pyiqa', 'MS-SSIM-pyiqa': 'ms_ssim_pyiqa',
    'SSIMc': 'ssimc', 'STLPIPS': 'stlpips_alex', 'FLIP': 'flip_v17',
    'CW-SSIM': 'cw_ssim_pyiqa', 'CAMBI': 'cambi_aic2026',
    'HDR_VDP_2': 'hdrvdp_aic2026_q', 'HDR_VDP_3': 'hdrvdp3_aic2026_q',
    'PSNR-HVS': 'psnrhvs_daala', 'PSNR-HVS-Y': 'psnrhvs_daala_y',
    'PSNR-HVS-Cb': 'psnrhvs_daala_cb', 'PSNR-HVS-Cr': 'psnrhvs_daala_cr',
    'PSNR-HVS-vmaf': 'psnrhvs_daala', 'PSNR-HVS-Y-vmaf': 'psnrhvs_daala_y',
    'VMAF v0': 'vmaf_studio601', 'VMAF-neg v0': 'vmaf_neg_studio601', 'VMAF v1': 'vmaf_v1',
    'Butteraugli-max': 'butteraugli_max', 'Butteraugli-3norm': 'butteraugli_pnorm3',
}
BLOCKED = {'AIMOS': 'excluded: proprietary, unreleased implementation',
           'proposal-IDFIQA': 'private implementation; not shipped in this repository',
           'MSSWD': 'private implementation; not shipped in this repository'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def atomic(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.part')
    created = False
    try:
        with tmp.open('x', encoding='utf8', newline='') as stream:
            created = True
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if created and tmp.exists():
            tmp.unlink()  # Only this call's newly created staging file.


def put_json(path, value):
    atomic(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def image_path(root, name):
    root = root.resolve()
    candidates = []
    for relative in [Path(name), Path('sources') / name, Path('distorted') / name,
                     Path('PTC-crops-all') / name]:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('image name escapes image directory: ' + name)
        if path.is_file() and path not in candidates:
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError('expected exactly one image in the root or standard AIC directories: ' + name)
    return candidates[0]


def prepare(args):
    with args.aic_csv.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames
        rows = list(reader)
    if header and 'view' in header:
        if not args.view:
            raise ValueError('CSV has multiple-view identity; explicitly select --view cropped or fullres')
        rows = [r for r in rows if r['view'] == args.view]
    elif args.view:
        raise ValueError('--view requires a view column in the CSV')
    if not rows or not header or not {'source', 'distorted'} <= set(header):
        raise ValueError('nonempty AIC CSV with source/distorted columns required')
    if len(header) != len(set(header)) or len({r['distorted'] for r in rows}) != len(rows):
        raise ValueError('duplicate columns or distorted names')
    mapping = json.loads((ROOT / 'scripts/complexity/aic2026_columns.json').read_text())
    routes = dict(mapping['map'])
    routes.update({'A-FINE': 'afine', 'proposal-mDCTPSNR': 'mdctpsnr',
                   'PSNR-HVS-vmaf': 'psnrhvs-daala', 'PSNR-HVS-Y-vmaf': 'psnrhvs-daala',
                   'VMAF v0': 'vmaf-studio601', 'VMAF-neg v0': 'vmaf-neg-studio601', 'VMAF v1': 'vmaf-v1'})
    for k in range(1, 6):
        routes[f'WD_s{k}'] = 'wd'
    for suffix in ['Y', 'Cb', 'Cr']:
        routes['PSNR-HVS-' + suffix] = 'psnrhvs-daala'
    selected = args.columns.split(',') if args.columns else [h for h in header if h not in META]
    if len(selected) != len(set(selected)) or any(c not in header for c in selected):
        raise ValueError('selected columns must occur once in the AIC header')
    columns = {}
    for column in selected:
        if column.startswith('JND_'):
            columns[column] = {'derived': column[4:], 'reason': 'explicit JND mapping required'}
        elif column in ('ADM2', 'VIF_vmaf') or column.startswith('VIF-vmaf-scale') and column[-1:] in '0123':
            selector = 'vif_s' + column[-1] if column.startswith('VIF-vmaf-scale') else column.lower()
            columns[column] = {'route': '@vmaf-features', 'selector': selector}
        elif column in routes:
            route = routes[column]
            columns[column] = {'route': route, 'selector': SELECTORS.get(column, route.replace('-', '_'))}
            if column.startswith('WD_s'):
                columns[column]['selector'] = column.lower()
            if column == 'proposal-mDCTPSNR':
                columns[column].update(selector='mdctpsnr_imazen_*', transform='mdct-proposal')
        else:
            columns[column] = {'blocked': BLOCKED.get(column, 'no qualified public route for this published column')}
    for route in args.extra_metric:
        name = 'WS-HV' if route == 'ws-hv' else 'extra:' + route
        if name in columns or name in header:
            raise ValueError('duplicate extra metric: ' + route)
        columns[name] = {'route': route, 'selector': route.replace('-', '_'), 'extra': True}
    jnd = {}
    if args.jnd_config:
        config = json.loads(args.jnd_config.read_text())['mappings']
        for column, spec in columns.items():
            if 'derived' not in spec:
                continue
            base = spec['derived']
            fit = config.get('PSNR-HVS-vmaf' if base == 'PSNR-HVS' else base)
            if fit is None or not {'a', 'b', 'c', 'sign'} <= set(fit):
                raise ValueError('no supported explicit JND fit for ' + column)
            jnd[column] = {k: fit[k] for k in ['a', 'b', 'c', 'sign']}
            if base not in columns:
                raise ValueError('select JND source column too: ' + base)
    pairs, files = [], {}
    for row in rows:
        for role in ['source', 'distorted']:
            name = row[role]
            if name not in files:
                files[name] = sha(image_path(args.images, name))
        pairs.append({k: row[k] for k in header if k in META})
    plan = dict(schema=SCHEMA, csv_sha256=sha(args.aic_csv), header=header,
                columns=columns, pairs=pairs, files=files, jnd=jnd,
                mapping_sha256=sha(ROOT / 'scripts/complexity/aic2026_columns.json'),
                qualifications={k: mapping['notes'].get(k, '') for k in columns},
                images_relative=os.path.relpath(args.images.resolve(), args.output.parent.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('plan already exists; preserve it or choose a new output path')
    put_json(args.output, plan)
    print(json.dumps({'pairs': len(pairs), 'selected_columns': len(columns),
                      'blocked_columns': {k: v['blocked'] for k, v in columns.items() if 'blocked' in v}}, indent=2))


@contextmanager
def writer_lock(root):
    # OS locks release on process death; there is no stale lease to guess away.
    with (root / '.writer.lock').open('a+b') as stream:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            if not stream.read(1):
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def execute(command, attempt, env, timeout):
    put_json(attempt / 'started.json', {'command': command, 'time': time.time()})
    with (attempt / 'stdout').open('wb') as stdout, (attempt / 'stderr').open('wb') as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                 start_new_session=os.name != 'nt')
        put_json(attempt / 'child.json', {'pid': child.pid})
        try:
            code = child.wait(timeout=timeout)
        except BaseException as error:
            if os.name != 'nt':
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
            child.wait()
            put_json(attempt / 'exit.json', {'returncode': child.returncode, 'interrupted_by': type(error).__name__})
            raise
    put_json(attempt / 'exit.json', {'returncode': code})
    return code


def select(scores, pattern):
    matches = [v for k, v in scores.items() if fnmatch.fnmatchcase(k, pattern)]
    if len(matches) != 1 or isinstance(matches[0], bool) or not isinstance(matches[0], (float, int)):
        raise ValueError('missing, null or ambiguous scalar column: ' + pattern)
    value = float(matches[0])
    if not math.isfinite(value):
        raise ValueError('nonfinite scalar column: ' + pattern)
    return value


def read_cell(path):
    if not path.exists():
        return {}
    cell = json.loads(path.read_text())
    identity = json.loads((path.parent.parent / 'identity.json').read_text())
    if (cell.get('schema') != SCHEMA or cell.get('digest') != digest(cell['payload'])
            or cell.get('pair_id') != path.stem or cell.get('run_digest') != digest(identity)):
        raise ValueError('corrupt checkpoint: ' + str(path))
    return cell['payload']


def save_cell(path, payload):
    identity = json.loads((path.parent.parent / 'identity.json').read_text())
    put_json(path, {'schema': SCHEMA, 'digest': digest(payload), 'payload': payload,
                    'pair_id': path.stem, 'run_digest': digest(identity)})


def project(plan, root):
    fields = plan['header'] + [k for k, v in plan['columns'].items() if v.get('extra')]
    csv_out, states = io.StringIO(newline=''), io.StringIO(newline='')
    writer = csv.DictWriter(csv_out, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    sw = csv.writer(states, lineterminator='\n')
    sw.writerow(['distorted', 'column', 'status', 'reason'])
    counts = {}
    for row in plan['pairs']:
        pair_id = digest([row['source'], row['distorted']])
        cell = read_cell(root / 'cells' / (pair_id + '.json'))
        output = dict(row)
        for col, spec in plan['columns'].items():
            result = cell.get(col, {})
            if 'blocked' in spec:
                status, reason = 'blocked', spec['blocked']
            elif 'derived' in spec:
                source = cell.get(spec['derived'], {})
                if col in plan['jnd'] and source.get('status') == 'scored':
                    fit = plan['jnd'][col]
                    value = fit['a'] * abs(fit['sign'] * (source['value'] - fit['b'])) ** fit['c']
                    if not math.isfinite(value):
                        raise ValueError('nonfinite derived JND: ' + col)
                    output[col] = format(value, '.17g')
                    status, reason = 'derived', 'explicit caller-supplied fit'
                else:
                    status, reason = 'blocked' if col not in plan['jnd'] else 'pending', spec['reason']
            else:
                status, reason = result.get('status', 'pending'), result.get('reason', '')
                if status == 'scored':
                    output[col] = format(result['value'], '.17g')
            sw.writerow([row['distorted'], col, status, reason])
            counts[status] = counts.get(status, 0) + 1
        writer.writerow(output)
    atomic(root / 'scores.csv', csv_out.getvalue())
    atomic(root / 'status.csv', states.getvalue())
    validation_path = root / 'validation.json'
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
    verified = validation.get('status') == 'verified'
    summary = {'schema': SCHEMA, 'pairs': len(plan['pairs']), 'counts': counts,
               'hashes_verified': verified,
               'complete': verified and all(k in ('scored', 'derived') for k in counts)}
    put_json(root / 'summary.json', summary)
    return summary


def model_identity(path):
    if not path:
        return {}
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*')) if p.is_file()}


def run(args):
    plan = json.loads(args.plan.read_text())
    if plan.get('schema') != SCHEMA:
        raise ValueError('unsupported plan schema')
    images = args.images or (args.plan.parent / plan['images_relative'])
    args.output.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output):
        for name, expected in plan['files'].items():
            if sha(image_path(images, name)) != expected:
                raise ValueError('input hash mismatch: ' + name)
        if (args.output / 'invalid.json').exists():
            raise ValueError('previous integrity refusal; preserve this campaign and use a new output directory')
        for child_file in (args.output / 'attempts').glob('*/child.json'):
            if not (child_file.parent / 'exit.json').exists():
                pid = json.loads(child_file.read_text())['pid']
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError('unfinished child may still be alive; refusing overlapping scoring: ' + str(pid))
        binary = args.binary.resolve(strict=True)
        vmaf = args.vmaf_features.resolve(strict=True) if args.vmaf_features else None
        env = os.environ.copy()
        env.pop('ZENMETRICS_NN_MODEL_DIR', None)
        for key in ['RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RTEN_NUM_THREADS']:
            env[key] = str(args.threads)
        if args.models:
            env['ZENMETRICS_NN_MODEL_DIR'] = str(args.models.resolve(strict=True))
        env['ZENMETRICS_NN_ENGINE'] = args.engine
        env['ZENMETRICS_NN_ACCEPT_WEIGHTS'] = args.accept_weights
        identity = dict(schema=SCHEMA, plan_digest=digest(plan), binary_sha256=sha(binary),
                        runner_sha256=sha(__file__), source_commit=args.source_commit,
                        vmaf_features_sha256=sha(vmaf) if vmaf else None,
                        model_files=model_identity(args.models), threads=args.threads,
                        engine=args.engine, accepted_weights=args.accept_weights,
                        environment={k: env.get(k) for k in sorted(env) if k.startswith(('ZENMETRICS_', 'RTEN_', 'ORT_', 'CUDA_', 'CUDNN_', 'RAYON_', 'OMP_', 'MKL_', 'OPENBLAS_')) and k != 'ZENMETRICS_NN_MODEL_DIR'})
        binding = args.output / 'identity.json'
        if binding.exists() and json.loads(binding.read_text()) != identity:
            raise ValueError('checkpoint identity changed; choose a new output directory (old results preserved)')
        if not binding.exists():
            reserved = ['scores.csv', 'status.csv', 'summary.json', 'cells', 'attempts', 'validation.json']
            if any((args.output / name).exists() for name in reserved):
                raise ValueError('output contains unowned score/checkpoint files; choose a new output directory')
            put_json(binding, identity)
        put_json(args.output / 'validation.json', {'status': 'running'})
        (args.output / 'cells').mkdir(exist_ok=True)
        (args.output / 'attempts').mkdir(exist_ok=True)
        inventory = subprocess.check_output([str(binary), 'list-metrics'], text=True, env=env)
        atomic(args.output / 'inventory.txt', inventory)
        available = {words[0] for line in inventory.splitlines()[1:]
                     if len(words := line.split()) == 3 and words[-1] in ('yes', 'no')}
        if args.only:
            allowed = set(plan['columns']) | {s['route'] for s in plan['columns'].values() if 'route' in s}
            if not set(args.only.split(',')) <= allowed:
                raise ValueError('--only contains a column/route not present in the plan')
        summary = project(plan, args.output)
        attempted_pairs = 0
        for row in plan['pairs']:
            pair_id = digest([row['source'], row['distorted']])
            path = args.output / 'cells' / (pair_id + '.json')
            cell = read_cell(path)
            todo = {}
            for col, spec in plan['columns'].items():
                if args.only and col not in args.only.split(',') and spec.get('route') not in args.only.split(','):
                    continue
                if 'route' not in spec or cell.get(col, {}).get('status') == 'scored':
                    continue
                count = cell.get(col, {}).get('attempts', 0)
                if count >= args.max_attempts:
                    continue
                route = spec['route']
                if route not in available and route != '@vmaf-features':
                    cell[col] = {'status': 'unavailable', 'reason': 'route absent from executable inventory: ' + route, 'attempts': count}
                elif route == '@vmaf-features' and vmaf is None:
                    cell[col] = {'status': 'unavailable', 'reason': '--vmaf-features executable required', 'attempts': count}
                else:
                    todo[col] = spec
            save_cell(path, cell)
            if not todo or args.limit_pairs is not None and attempted_pairs >= args.limit_pairs:
                continue
            attempted_pairs += 1
            ref = image_path(images, row['source'])
            dist = image_path(images, row['distorted'])
            # Commit each adapter batch separately; an optional feature adapter
            # must not erase already completed compare results on interruption.
            for adapter in ['compare', 'features']:
                group = {c: s for c, s in todo.items() if (s['route'] == '@vmaf-features') == (adapter == 'features')}
                if not group:
                    continue
                attempt = args.output / 'attempts' / (pair_id + '-' + uuid.uuid4().hex)
                attempt.mkdir()
                for col in group:
                    cell[col] = {'status': 'interrupted', 'reason': 'attempt started; no completed scalar yet',
                                 'attempts': cell.get(col, {}).get('attempts', 0) + 1,
                                 'attempt': str(attempt.relative_to(args.output))}
                save_cell(path, cell)
                if adapter == 'compare':
                    routes = sorted({s['route'] for s in group.values()}, key=lambda s: (not s.startswith('hdrvdp'), s))
                    command = [str(binary), 'compare', '--reference', str(ref), '--variant', str(dist), '--output', 'json']
                    for route in routes:
                        command.extend(['--metric', route])
                    if 'cvvdp' in routes:
                        command.extend(['--display-model', 'standard_fhd'])
                else:
                    pair_tsv = attempt / 'pair.tsv'
                    with pair_tsv.open('w', newline='') as stream:
                        w = csv.writer(stream, delimiter='\t', lineterminator='\n')
                        w.writerow(['ref_path', 'dist_path', 'distorted'])
                        w.writerow([str(ref), str(dist), row['distorted']])
                    command = [str(vmaf), str(pair_tsv), str(attempt / 'features.tsv'), str(args.threads)]
                scores, failure = {}, ''
                try:
                    code = execute(command, attempt, env, args.timeout)
                    stderr = (attempt / 'stderr').read_text(errors='replace')
                    if code not in (0, 1) or 'panicked at' in stderr or 'cubecl-wgpu:' in stderr:
                        raise ValueError('unhealthy child execution; no scores accepted; see attempt stderr')
                    if adapter == 'compare':
                        doc = json.loads((attempt / 'stdout').read_text())
                        if doc.get('metrics') != routes or len(doc['results']) != 1:
                            raise ValueError('unexpected compare response identity')
                        result = doc['results'][0]
                        if result['reference'] != str(ref) or result['variant'] != str(dist):
                            raise ValueError('unexpected compare input identity')
                        scores = result['scores']
                    else:
                        with (attempt / 'features.tsv').open() as stream:
                            result = list(csv.DictReader(stream, delimiter='\t'))
                        if code or len(result) != 1 or result[0]['distorted'] != row['distorted']:
                            raise ValueError('unexpected feature adapter response')
                        scores = {k: float(v) for k, v in result[0].items() if k not in ('distorted', 'ref_path', 'dist_path')}
                    if code:
                        failure = 'child exit ' + str(code) + '; see attempt stderr'
                except (ValueError, KeyError, OSError, subprocess.TimeoutExpired) as error:
                    scores = {}
                    failure = str(error)
                for col, spec in group.items():
                    try:
                        value = select(scores, spec['selector'])
                        if spec.get('transform') == 'mdct-proposal':
                            value = 0.000475997802 * max(0, 80 - value) ** 2.3182
                        cell[col].update(status='scored', value=value, reason='')
                    except ValueError as error:
                        cell[col].update(status='failed', reason=failure or str(error))
                save_cell(path, cell)
            if attempted_pairs % args.export_every == 0:
                summary = project(plan, args.output)
            print(json.dumps({'pair': row['distorted'], 'attempted_pairs': attempted_pairs,
                              'checkpointed_columns': sum(c.get('status') == 'scored' for c in cell.values())}), flush=True)
        # Rehash all inputs/assets/executables before advertising completed output.
        invalid = []
        try:
            if sha(binary) != identity['binary_sha256'] or (vmaf and sha(vmaf) != identity['vmaf_features_sha256']):
                invalid.append('executable changed during run')
            if model_identity(args.models) != identity['model_files']:
                invalid.append('model assets changed during run')
            for name, expected in plan['files'].items():
                if sha(image_path(images, name)) != expected:
                    invalid.append('input changed during run: ' + name)
        except (OSError, ValueError) as error:
            invalid.append('post-run identity check failed: ' + str(error))
        if invalid:
            put_json(args.output / 'invalid.json', {'reasons': invalid})
            raise ValueError('; '.join(invalid))
        put_json(args.output / 'validation.json', {'status': 'verified'})
        summary = project(plan, args.output)
        print(json.dumps(summary, indent=2))
        return 0 if summary['complete'] else 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='bind an AIC CSV, relative image names and hashes')
    prep.add_argument('--aic-csv', required=True, type=Path)
    prep.add_argument('--images', required=True, type=Path)
    prep.add_argument('--output', required=True, type=Path)
    prep.add_argument('--view', choices=['cropped', 'fullres'])
    prep.add_argument('--columns', help='comma-separated original CSV columns; default all')
    prep.add_argument('--extra-metric', action='append', default=[])
    prep.add_argument('--jnd-config', type=Path)
    r = sub.add_parser('run', help='resume missing scalar cells and atomically refresh CSV projections')
    r.add_argument('--plan', required=True, type=Path)
    r.add_argument('--images', type=Path, help='override plan image root on another computer')
    r.add_argument('--binary', required=True, type=Path)
    r.add_argument('--source-commit', required=True, help='source revision from the binary build receipt')
    r.add_argument('--output', required=True, type=Path)
    r.add_argument('--vmaf-features', type=Path)
    r.add_argument('--threads', type=int, default=1)
    r.add_argument('--max-attempts', type=int, default=2)
    r.add_argument('--limit-pairs', type=int)
    r.add_argument('--export-every', type=int, default=100, help='refresh CSV projections every N attempted pairs and on clean exit')
    r.add_argument('--only', help='schedule only these comma-separated columns/routes; resume the rest later')
    r.add_argument('--timeout', type=float)
    r.add_argument('--models', type=Path)
    r.add_argument('--engine', choices=['rten', 'ort-cpu', 'ort-cuda', 'ort-webgpu'], default='rten')
    r.add_argument('--accept-weights', default='')
    a = p.parse_args(argv)
    if a.command == 'prepare':
        prepare(a)
        return 0
    if a.threads < 1 or a.export_every < 1 or a.max_attempts < 1 or a.limit_pairs is not None and a.limit_pairs < 0 or a.timeout is not None and a.timeout <= 0:
        p.error('positive threads, attempt budget and timeout; nonnegative pair limit required')
    return run(a)


if __name__ == '__main__':
    try:
        def terminate(signum, frame):
            raise KeyboardInterrupt('termination requested')
        signal.signal(signal.SIGTERM, terminate)
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('aic-rescore: interrupted; checkpoints and attempt logs retained', file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print('aic-rescore: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
