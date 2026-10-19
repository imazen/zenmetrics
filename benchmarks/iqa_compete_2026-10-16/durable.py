#!/usr/bin/env python3
"""Crash-resumable Unix IQA controller. Frozen manifest schema 1, stdlib only.

run MANIFEST OUTPUT [--run-heavy PATH --memory 12G --jobs N --lock PATH]
status MANIFEST OUTPUT verifies retained evidence without starting measurements.
Exit 0 means all cells terminal (inspect report for failures), 2 is permanent
configuration/integrity refusal; signals and supervisor crashes are restartable.
The advisory lock is scheduling, not proof of exclusive hardware access.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import select
import signal
import socket
import struct
import subprocess
import sys
import time
import uuid
from admission_recovery import recover

SELF = str(Path(__file__).resolve())
HEARTBEAT = 2.0
GRACE = 1.0
STOP = 0


class Refusal(ValueError):
    """Permanent configuration or retained-evidence integrity failure."""


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with tmp.open('xb') as stream:
        stream.write(encoded(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load(path):
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as error:
        raise Refusal(f'invalid JSON {path}: {error}') from error


def save_result(path, record):
    atomic(path, dict(record=record, sha256=digest(encoded(record))))


def read_result(path):
    value = load(path)
    if not isinstance(value, dict) or value.get('sha256') != digest(encoded(value.get('record'))):
        raise Refusal(f'result checksum mismatch: {path}')
    return value['record']


def relative(root, name):
    if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
        raise Refusal('manifest paths must be relative strings')
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root):
        raise Refusal('manifest path escapes bundle')
    return path


def integer(value, minimum=1):
    return type(value) is int and value >= minimum


def manifest(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get('schema') != 1 or not isinstance(value.get('metadata'), dict):
        raise Refusal('manifest requires schema=1 and metadata object')
    root = path.parent
    runners = value['metadata'].get('runner_sha256', {})
    for name, expected in runners.items():
        if file_hash(relative(root, name)) != expected:
            raise Refusal('runner content changed: ' + name)
    if 'durable.py' in runners and file_hash(SELF) != runners['durable.py']:
        raise Refusal('executed controller differs from frozen runner')
    binary = relative(root, value.get('binary'))
    binary_sha = file_hash(binary)
    if binary_sha != value.get('binary_sha256'):
        raise Refusal('binary_sha256 mismatch')
    hashes = value.get('input_sha256')
    if not isinstance(hashes, dict):
        raise Refusal('manifest requires frozen input_sha256 mapping')
    observed = {}
    for name, expected in hashes.items():
        observed[name] = file_hash(relative(root, name))
        if observed[name] != expected:
            raise Refusal(f'input_sha256 mismatch: {name}')
    if not isinstance(value.get('cells'), list) or not value['cells']:
        raise Refusal('manifest requires nonempty cells')
    configs = []
    for index, cell in enumerate(value['cells']):
        if not isinstance(cell, dict):
            raise Refusal('cell must be an object')
        config = dict(cell)
        check_score_policy(config)
        placement = value['metadata'].get('cpu_placement')
        if placement:
            if (placement.get('policy') != 'performance' or not placement.get('cpus')
                    or any(type(cpu) is not int or cpu < 0 for cpu in placement['cpus'])
                    or placement.get('zenbench') not in value.get('input_sha256', {})):
                raise Refusal('invalid checked CPU placement')
            config['cpu_placement'] = dict(placement,
                zenbench=str(relative(root, placement['zenbench'])))
        if cell.get('timing_phase') not in ('cold', 'warm') or cell.get('only_mode') not in ('score', 'score+map'):
            raise Refusal('each cell requires timing_phase and only_mode')
        if cell['timing_phase'] == 'cold' and not integer(cell.get('cold_repeat'), 0):
            raise Refusal('cold cells require nonnegative integer cold_repeat')
        if not isinstance(cell.get('metric_name'), str) or not cell['metric_name']:
            raise Refusal('metric_name required')
        if not all(integer(cell.get(k)) for k in ('width', 'height', 'threads')):
            raise Refusal('positive width, height, threads required')
        required = cell.get('required_samples', 10)
        if not integer(required, 1 if cell['timing_phase'] == 'cold' else 10):
            raise Refusal('required_samples must be at least ten')
        for role in ('reference', 'distorted'):
            name = cell.get(role)
            if name not in hashes:
                raise Refusal(f'missing frozen input hash: {name}')
            config[role] = str(relative(root, name))
            config[role + '_sha256'] = hashes[name]
        assets = cell.get('asset_paths', [])
        if not isinstance(assets, list) or any(name not in hashes for name in assets):
            raise Refusal('every worker asset requires a frozen input hash')
        allowed = {'ZENMETRICS_NN_MODEL_DIR', 'ZENMETRICS_NN_ENGINE', 'ZENMETRICS_NN_ACCEPT_WEIGHTS',
                   'ORT_DYLIB_PATH', 'ZENMETRICS_NN_WEBGPU_EP_LIB', 'WGPU_BACKEND', 'LD_LIBRARY_PATH'}
        worker_env = cell.get('worker_environment', {})
        if not isinstance(worker_env, dict) or set(worker_env) - allowed:
            raise Refusal('unknown worker environment key')
        resolved_env = {}
        for key, val in worker_env.items():
            if not isinstance(val, str):
                raise Refusal('worker environment values must be strings')
            if key in ('ZENMETRICS_NN_MODEL_DIR', 'ORT_DYLIB_PATH', 'ZENMETRICS_NN_WEBGPU_EP_LIB', 'LD_LIBRARY_PATH'):
                target = relative(root, val)
                files = [p for p in target.rglob('*') if p.is_file()] if target.is_dir() else [target]
                if not files or any(str(p.relative_to(root)) not in assets for p in files):
                    raise Refusal('worker environment path lacks active asset hashes')
                val = str(target)
            resolved_env[key] = val
        if worker_env:
            config['worker_environment'] = resolved_env
        if assets:
            config['asset_sha256'] = {str(relative(root, name)): hashes[name] for name in assets}
        config['impl_binary_sha'] = binary_sha
        config['durable_cell_id'] = f'{index:06d}-' + digest(encoded(cell))[:16]
        configs.append(config)
    identity = dict(schema=1, manifest_sha256=digest(raw), manifest=value,
                    binary_sha256=binary_sha, input_sha256=observed)
    return binary, configs, identity


def environment(config, scratch):
    env = os.environ.copy()
    for key in ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'CARGO_BUILD_JOBS',
                'RTEN_NUM_THREADS', 'ORT_NUM_THREADS', 'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = str(config['threads'])
    placement = config.get('cpu_placement')
    env.update(TMPDIR=str(scratch), ZENBENCH_NO_SAVE='1',
               ZENBENCH_CPU_POLICY=placement['policy'] if placement else 'unrestricted')
    if placement:
        env['ZENBENCH_CPU_SET'] = ','.join(map(str, placement['cpus']))
    if config.get('worker_environment'):
        for key in ('ZENMETRICS_NN_MODEL_DIR', 'ZENMETRICS_NN_ENGINE', 'ZENMETRICS_NN_ACCEPT_WEIGHTS', 'ORT_DYLIB_PATH', 'ZENMETRICS_NN_WEBGPU_EP_LIB', 'WGPU_BACKEND', 'LD_LIBRARY_PATH'):
            env.pop(key, None)
        env.update(config['worker_environment'])
    return env


def verify_assets(config):
    for path, expected in config.get('asset_sha256', {}).items():
        if file_hash(path) != expected:
            raise Refusal('active model/runtime asset changed: ' + path)


def observation():
    cpu_model = platform.processor()
    if sys.platform == 'darwin':
        cpu_model = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    elif Path('/proc/cpuinfo').is_file():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu_model = line.split(':', 1)[1].strip()
                break
    return dict(cpu_model=cpu_model, logical_cpus=os.cpu_count(), host=platform.node(), os=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), python=sys.version, load=os.getloadavg(),
                cpu_ticks=[line for line in Path('/proc/stat').read_text().splitlines() if line.startswith('cpu')]
                    if Path('/proc/stat').exists() else None,
                unix_ns=time.time_ns())


FINITE_GPU_POLICY = 'mac-metal-finite-logged-v1'


def check_score_policy(config):
    policy = config.get('gpu_score_repeatability_policy')
    if policy is None:
        return False
    env = config.get('worker_environment', {})
    if (policy != FINITE_GPU_POLICY or config.get('device') != 'metal'
            or config.get('only_mode') != 'score' or config.get('timing_phase') != 'warm'
            or env.get('WGPU_BACKEND') != 'metal' or env.get('ZENMETRICS_NN_ENGINE') != 'ort-webgpu'):
        raise Refusal('finite score policy requires warm scalar Metal ORT WebGPU')
    return True


def finite_gpu_observations(row):
    """Accept timing independently of score repeatability, never oracle parity."""
    def bits(key):
        text = row.get(key)
        if not isinstance(text, str) or len(text) != 16:
            raise Refusal('missing observation bits: ' + key)
        value = struct.unpack('>d', bytes.fromhex(text))[0]
        if not math.isfinite(value):
            raise Refusal('nonfinite observation: ' + key)
        return value
    total, finite, nonfinite, varied = [row.get(k) for k in (
        'gpu_call_observation_count', 'gpu_finite_observation_count',
        'gpu_nonfinite_observation_count', 'gpu_call_bit_variation_count')]
    if (any(type(v) is not int or v < 0 for v in (total, finite, nonfinite, varied))
            or total < row['n'] or finite != total or nonfinite != 0 or varied > total):
        raise Refusal('missing or nonfinite full-call observations')
    samples = row.get('zenbench_samples')
    if (not isinstance(samples, list) or len(samples) != row['n']
            or any(s.get('gate_clean') is not True or not integer(s.get('iterations')) for s in samples)
            or total < sum(s['iterations'] for s in samples)):
        raise Refusal('finite policy requires complete clean timed rounds')
    baseline = bits('gpu_reference_score_bits')
    low, high, last = bits('gpu_min_score_bits'), bits('gpu_max_score_bits'), bits('score_bits')
    if not low <= baseline <= high or not low <= last <= high or row.get('gpu_score_range_includes_baseline') is not True:
        raise Refusal('incoherent finite score range')
    if varied:
        if not low <= bits('gpu_one_different_score_bits') <= high:
            raise Refusal('alternate score outside range')
    elif low != baseline or high != baseline or last != baseline:
        raise Refusal('unrecorded score variation')
    delta = max(abs(low - baseline), abs(high - baseline))
    if (row.get('gpu_max_abs_delta_bits') != struct.pack('>d', delta).hex()
            or row.get('gpu_delta_overflow') is not (not math.isfinite(delta))
            or row.get('gpu_max_abs_delta') != (delta if math.isfinite(delta) else None)):
        raise Refusal('incoherent absolute score delta')
    relative = delta / abs(baseline) if baseline else math.inf
    if row.get('gpu_max_relative_delta') != (relative if math.isfinite(relative) else None):
        raise Refusal('incoherent relative score delta')
    row.update(raw_status=row['status'], status='ok', gpu_score_repeatability_policy=FINITE_GPU_POLICY,
               numerical_variation_accepted=bool(varied), score_repeatability_qualified=not bool(varied),
               numerical_instability_flag=bool(varied),
               numerical_qualification='Finite observations only; no native-oracle parity qualification')


def validate_row(log, config):
    rows = []
    with Path(log).open(errors='replace') as stream:
        for line in stream:
            if line.startswith('IQA_ROW '):
                if not line.endswith('\n'):
                    raise Refusal('unterminated IQA_ROW')
                rows.append(json.loads(line[8:]))
    if len(rows) != 1:
        raise Refusal(f'expected one IQA_ROW, observed {len(rows)}')
    row = rows[0]
    for key in ('metric_name', 'reference', 'distorted', 'width', 'height', 'threads',
                'timing_phase', 'only_mode', 'cold_repeat', 'durable_cell_id',
                'reference_sha256', 'distorted_sha256', 'impl_binary_sha', 'worker_environment', 'asset_sha256'):
        if row.get(key) != config.get(key):
            raise Refusal('IQA_ROW identity mismatch: ' + key)
    if row.get('cpu_placement') != config.get('cpu_placement'):
        raise Refusal('IQA_ROW identity mismatch: cpu_placement')
    if row.get('mode') != config['only_mode']:
        raise Refusal('IQA_ROW mode mismatch')
    finite_policy = check_score_policy(config)
    status = row.get('status')
    if status in ('unsupported', 'failed'):
        if type(row.get('n')) is not int or row['n'] != 0 or any(key in row for key in ('median_ns', 'cold_elapsed_ns')):
            raise Refusal('refusal row contains timing')
        return row
    if status != 'ok' and not (finite_policy and status == 'variant_mismatch'):
        raise Refusal('unqualified IQA_ROW status: ' + str(status))
    if not integer(row.get('n')):
        raise Refusal('invalid sample count')
    timing_keys = ('median_ns', 'input_load_ns') if config['timing_phase'] == 'cold' else ('median_ns',)
    for key in timing_keys:
        number = row.get(key)
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0 or (key == 'median_ns' and number == 0):
            raise Refusal('invalid timing: ' + key)
    for key in ('mean_ns', 'p10_ns', 'p90_ns', 'input_load_ns', 'cold_elapsed_ns'):
        if key in row and (type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0):
            raise Refusal('invalid timing statistic: ' + key)
    if config['timing_phase'] == 'cold':
        if row['n'] != 1 or row.get('cold_elapsed_ns') != row['median_ns']:
            raise Refusal('cold sample must have n=1 and actual cold_elapsed_ns')
    elif row['n'] < config.get('required_samples', 10):
        raise Refusal('insufficient warm samples')
    elif 'round_means_ns' not in row:
        raise Refusal('missing actual warm round samples')
    if 'round_means_ns' in row:
        samples = row['round_means_ns']
        if not isinstance(samples, list) or len(samples) != row['n'] or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in samples):
            raise Refusal('invalid actual round samples')
    bits = row.get('score_bits')
    if not isinstance(bits, str) or len(bits) != 16 or not math.isfinite(struct.unpack('>d', bytes.fromhex(bits))[0]):
        raise Refusal('invalid finite score_bits')
    if 'score' in row and (type(row['score']) not in (int, float) or not math.isfinite(row['score'])
                           or struct.pack('>d', row['score']).hex() != bits.lower()):
        raise Refusal('score disagrees with score_bits')
    if finite_policy:
        finite_gpu_observations(row)
    return row


def group_members(pgid):
    # ps runs outside the inspected group, so the observation cannot count itself.
    text = subprocess.check_output(['ps', '-axo', 'pid=,pgid=,stat='], text=True, start_new_session=True)
    return [int(pid) for pid, group, state in (line.split()[:3] for line in text.splitlines())
            if int(group) == pgid and not state.startswith('Z')]


def worker_exec(attempt):
    """Persist the worker PID/group before exec can create an admission ticket."""
    intent = load(attempt / 'intent.json')
    atomic(attempt / 'worker-process.json', dict(pid=os.getpid(), pgid=os.getpgrp(), unix_s=int(time.time())))
    config = load(attempt / 'config.json')
    placement = config.get('cpu_placement')
    if placement:
        enforce_placement(placement, attempt)
    os.execv(intent['binary'], [intent['binary'], str(attempt / 'config.json')])


def enforce_placement(placement, attempt):
    """Constrain the actual resource scope, then ask zenbench to certify it."""
    group = Path('/proc/self/cgroup').read_text().strip().split('0::')[-1]
    scope = Path('/sys/fs/cgroup') / group.lstrip('/')
    if ('..' in scope.parts or scope.suffix != '.scope' or 'app.slice' not in scope.parts
            or f'user@{os.getuid()}.service' not in scope.parts):
        raise Refusal('CPU placement requires an owned run-heavy user scope')
    target = scope / 'cpuset.cpus'
    cpus = ','.join(map(str, placement['cpus']))
    # Some user managers delegate memory but leave cpuset files root-owned.
    # The path comes from this process's kernel membership, never the manifest.
    subprocess.run(['sudo', '-n', 'tee', str(target)], input=cpus + '\n',
                   text=True, stdout=subprocess.DEVNULL, check=True)
    os.sched_setaffinity(0, placement['cpus'])
    observed = json.loads(subprocess.check_output([placement['zenbench'], 'cpu-info'], text=True))
    proof = observed.get('placement', {}).get('Ok', {})
    if (proof.get('status') != 'enforced_uniform' or proof.get('policy') != 'performance'
            or proof.get('effective_cpus') != placement['cpus']):
        raise Refusal('zenbench refused the requested P-core boundary')
    atomic(attempt / 'cpu-placement.json', dict(zenbench=observed,
        affinity=sorted(os.sched_getaffinity(0)), cgroup=str(scope),
        memory_max=(scope / 'memory.max').read_text().strip(),
        memory_swap_max=(scope / 'memory.swap.max').read_text().strip()))


def payload(attempt, address, seconds):
    """Inner guard survives resource wrappers that establish their own session."""
    if os.getpgrp() != os.getpid():
        os.setsid()
    pgid = os.getpid()  # This live process anchors the ID until its own final kill.
    interrupted = [False]
    signal.signal(signal.SIGTERM, lambda *_: interrupted.__setitem__(0, True))
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(address)
        connection.sendall(encoded(dict(pgid=pgid)) + b'\n')
        connection.setblocking(False)
        config = load(attempt / 'config.json')
        intent = load(attempt / 'intent.json')
        worker = subprocess.Popen([sys.executable, SELF, '_worker', str(attempt)],
                                  env=environment(config, Path(intent['scratch'])))
        deadline = time.monotonic() + seconds
        last = time.monotonic()
        reason = None
        while worker.poll() is None:
            if select.select([connection], [], [], .05)[0]:
                data = connection.recv(4096)
                if not data:
                    reason = 'supervisor_disconnected'
                    break
                last = time.monotonic()
            if interrupted[0] or time.monotonic() - last > HEARTBEAT or time.monotonic() >= deadline:
                reason = 'inner_deadline_or_supervisor_loss'
                break
        if reason is None:
            atomic(attempt / 'payload.json', dict(returncode=worker.returncode))
            if group_members(pgid) == [pgid]:
                return 0
        else:
            atomic(attempt / 'payload.json', dict(reason=reason))
        # Never signal a remembered PID/PGID: we are still its live group anchor.
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(GRACE)
        os.killpg(pgid, signal.SIGKILL)
    return 1


def anchor(attempt, address, seconds, supervisor):
    # A handler (not SIG_IGN) resets to default in exec'd children.
    signal.signal(signal.SIGTERM, lambda *_: None)
    intent = load(attempt / 'intent.json')
    command = [sys.executable, SELF, '_payload', str(attempt), address, str(seconds)]
    wrapper = intent['wrapper']
    if wrapper:
        command = [wrapper, '--mem', intent['memory'], '--jobs', str(intent['jobs']), '--', *command]
    atomic(attempt / 'command.json', dict(wrapper_command=command, worker_command=intent['command']))
    with (attempt / 'worker.log').open('xb', buffering=0) as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=environment(load(attempt / 'config.json'), Path(intent['scratch'])))
        while child.poll() is None:
            if os.getppid() != supervisor:
                os.killpg(os.getpgrp(), signal.SIGTERM)
                time.sleep(GRACE)
                os.killpg(os.getpgrp(), signal.SIGKILL)
            time.sleep(.05)
        rc = child.returncode
        os.fsync(log.fileno())
    atomic(attempt / 'exit.json', dict(returncode=rc))
    while True:
        if os.getppid() != supervisor:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        time.sleep(.05)


def watchdog(attempt, parent_fd, lock_fds, seconds):
    # lock_fds deliberately remain open until all descendants are contained.
    intent = load(attempt / 'intent.json')
    atomic(attempt / 'watchdog.json', dict(pid=os.getpid()))
    scratch = Path(intent['socket_root'])
    address = str(scratch / ('s-' + uuid.uuid4().hex[:16]))
    started = observation()
    signal.signal(signal.SIGTERM, lambda *_: None)
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(address)
        server.listen(1)
        server.setblocking(False)
        group = subprocess.Popen([sys.executable, SELF, '_anchor', str(attempt), address, str(seconds), str(os.getpid())],
                                 start_new_session=True)
        atomic(attempt / 'groups.json', dict(anchor=group.pid))
        connection = None
        payload_group = None
        hello = b''
        reason = None
        deadline = time.monotonic() + seconds
        heartbeat = time.monotonic()
        try:
            while not (attempt / 'exit.json').exists():
                if connection is None and select.select([server], [], [], 0)[0]:
                    connection, _ = server.accept()
                    connection.setblocking(False)
                if connection is not None:
                    if payload_group is None and select.select([connection], [], [], 0)[0]:
                        hello += connection.recv(4096)
                        if b'\n' in hello:
                            payload_group = json.loads(hello.split(b'\n')[0])['pgid']
                            atomic(attempt / 'groups.json', dict(anchor=group.pid, payload=payload_group))
                    try:
                        connection.send(b'.')
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                if select.select([parent_fd], [], [], .05)[0]:
                    data = os.read(parent_fd, 4096)
                    if not data:
                        reason = 'controller_lost'
                        break
                    heartbeat = time.monotonic()
                now = time.monotonic()
                if now >= deadline:
                    reason = 'timeout'
                    break
                if now - heartbeat > HEARTBEAT:
                    reason = 'controller_stalled'
                    break
        finally:
            if connection is not None:
                connection.close()  # Inner guard cleans its own anchored process group.
            os.killpg(group.pid, signal.SIGTERM)
            until = time.monotonic() + GRACE + HEARTBEAT + 1
            while payload_group and group_members(payload_group) and time.monotonic() < until:
                time.sleep(.05)
            os.killpg(group.pid, signal.SIGKILL)  # Unreaped child anchors PGID, no reuse.
            group.wait()
            os.unlink(address)
        clean = not group_members(group.pid) and (not payload_group or not group_members(payload_group))
    record = dict(identity=intent['identity'], cell_id=intent['cell_id'], attempt=intent['attempt'],
                  status='failed', reason=reason, row=None, cleanup_verified=clean,
                  before=started, after=observation(), command=intent['command'],
                  intent_sha256=file_hash(attempt / 'intent.json'), config_sha256=file_hash(attempt / 'config.json'),
                  resource_policy=intent['resource_policy'])
    if not clean:
        record['reason'] = 'cleanup_failed'
    elif reason is None:
        exit_path = attempt / 'exit.json'
        rc = load(exit_path)['returncode'] if exit_path.exists() else None
        record['returncode'] = rc
        try:
            config = load(attempt / 'config.json')
            verify_assets(config)
            if file_hash(intent['binary']) != config['impl_binary_sha'] or any(
                    file_hash(config[role]) != config[role + '_sha256'] for role in ('reference', 'distorted')):
                raise Refusal('binary/input changed during attempt')
            if rc != 0:
                raise Refusal(f'worker/wrapper exited {rc}')
            row = validate_row(attempt / 'worker.log', load(attempt / 'config.json'))
            record.update(row=row, status=row['status'], reason=None)
        except (ValueError, OSError, struct.error) as error:
            record['reason'] = str(error)
    if clean and (attempt / 'worker-process.json').exists():
        record['admission_recovery'] = recover(load(attempt / 'worker-process.json'), attempt / 'admission-recovery')
    log = attempt / 'worker.log'
    record['executed_command'] = load(attempt / 'command.json') if (attempt / 'command.json').exists() else None
    record['log_sha256'] = file_hash(log) if log.exists() else None
    save_result(attempt / 'result.json', record)
    return 0


@contextlib.contextmanager
def locked(path, wait=False, heartbeat=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a+b') as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not wait:
                    raise Refusal('coordinator output is locked')
                if heartbeat:
                    heartbeat()
                if STOP:
                    raise InterruptedError('stopped while waiting for admission lock')
                time.sleep(.1)
        yield stream.fileno()


def attempts(output, config, identity):
    directory = output / 'cells' / config['durable_cell_id']
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for number, path in enumerate(sorted(directory.glob('attempt-*')), 1):
        if path.name != f'attempt-{number:04d}':
            raise Refusal('noncontiguous attempt history')
        intent_path = path / 'intent.json'
        if not intent_path.exists():
            # Intent is committed before any child is launched; a crash here consumed
            # an attempt, but can never leave a measurement to adopt or kill.
            if any(p.name in ('worker.log', 'exit.json', 'payload.json', 'result.json') for p in path.iterdir()):
                raise Refusal('attempt has evidence without intent')
            atomic(path / 'config.json', config)
            atomic(intent_path, dict(identity=identity, cell_id=config['durable_cell_id'], attempt=number,
                                     preparation_interrupted=True))
        intent = load(intent_path)
        if (intent.get('identity') != identity or intent.get('cell_id') != config['durable_cell_id']
                or intent.get('attempt') != number or load(path / 'config.json') != config):
            raise Refusal('attempt identity/config mismatch')
        result_path = path / 'result.json'
        if not result_path.exists():
            # Allow autonomous guards to finish after a watchdog was killed.
            time.sleep(HEARTBEAT + GRACE + .2)
            groups = load(path / 'groups.json') if (path / 'groups.json').exists() else {}
            if any(group_members(pgid) for pgid in groups.values()):
                raise RuntimeError('interrupted supervisor groups still live; refusing admission')
            # The inherited exclusive lock proves no earlier watchdog is still alive.
            # An exit record + full log may have been fsynced before watchdog death.
            record = dict(identity=identity, cell_id=config['durable_cell_id'], attempt=number,
                          status='failed', reason='interrupted_attempt', row=None)
            if (path / 'exit.json').exists() and load(path / 'exit.json').get('returncode') == 0:
                try:
                    row = validate_row(path / 'worker.log', config)
                    record.update(status=row['status'], row=row, reason='adopted_finished_output')
                except (ValueError, OSError, struct.error):
                    pass
            record['intent_sha256'] = file_hash(path / 'intent.json')
            record['config_sha256'] = file_hash(path / 'config.json')
            record['log_sha256'] = file_hash(path / 'worker.log') if (path / 'worker.log').exists() else None
            save_result(result_path, record)
        record = read_result(result_path)
        if record.get('cleanup_verified') is False:
            groups = load(path / 'groups.json') if (path / 'groups.json').exists() else {}
            if not groups or any(group_members(pgid) for pgid in groups.values()):
                raise RuntimeError('prior cleanup unverified; refusing admission while groups live')
        if record.get('intent_sha256') != file_hash(intent_path) or record.get('config_sha256') != file_hash(path / 'config.json'):
            raise Refusal('retained intent/config changed')
        if any(record.get(k) != intent.get(k) for k in ('identity', 'cell_id', 'attempt')):
            raise Refusal('result identity mismatch')
        log = path / 'worker.log'
        if record.get('log_sha256') != (file_hash(log) if log.exists() else None):
            raise Refusal('retained log changed')
        if record['status'] in ('ok', 'unsupported') and validate_row(log, config) != record.get('row'):
            raise Refusal('retained row changed')
        records.append(record)
    return directory, records


def report(output, configs, identity, cap, heartbeat=None):
    cells = []
    for config in configs:
        if heartbeat:
            heartbeat()
        _, records = attempts(output, config, identity)
        final = next((r for r in records if r['status'] in ('ok', 'unsupported')), None)
        status = final['status'] if final else ('failed' if len(records) >= cap else 'pending')
        cells.append(dict(cell_id=config['durable_cell_id'], config=config, status=status,
                          attempts=records, row=final['row'] if final else None))
    value = dict(schema=1, identity=identity, cells=cells,
                 complete=all(c['status'] != 'pending' for c in cells),
                 qualified=all(c['status'] in ('ok', 'unsupported') for c in cells),
                 cold=[c['row'] for c in cells if c['status'] == 'ok' and c['config']['timing_phase'] == 'cold'],
                 warm=[c['row'] for c in cells if c['status'] == 'ok' and c['config']['timing_phase'] == 'warm'],
                 numerical_instability_cells=sum(bool(c.get('row') and c['row'].get('numerical_instability_flag')) for c in cells),
                 qualification_note='Protocol checks only; advisory locks and load observations do not qualify CPU contention or same-score parity')
    atomic(output / 'report.json', value)
    return value


def run(args):
    binary, configs, frozen = manifest(args.manifest)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with locked(output / 'coordinator.lock') as output_fd:
        identity_path = output / 'identity.json'
        policy = dict(timeout_seconds=args.timeout_seconds, max_attempts=args.max_attempts,
                      wrapper=str(Path(args.run_heavy).resolve(strict=True)) if args.run_heavy else None,
                      memory=args.memory, jobs=args.jobs,
                      lock=str(Path(args.lock).resolve()) if args.lock else None)
        if policy['wrapper']:
            policy['wrapper_sha256'] = file_hash(policy['wrapper'])
            backend = Path(policy['wrapper']).parent / 'run-heavy-macos.py'
            if backend.exists():
                policy['wrapper_backend_sha256'] = file_hash(backend)
        if identity_path.exists():
            saved = load(identity_path)
            if saved.get('frozen') != frozen:
                raise Refusal('frozen manifest/binary/input identity changed')
            if args.action == 'run' and saved.get('policy') != policy:
                raise Refusal('attempt policy changed')
            policy = saved['policy']
        else:
            if args.action != 'run':
                raise Refusal('output has no frozen identity')
            if any(p.name != 'coordinator.lock' for p in output.iterdir()):
                raise Refusal('refusing to initialize nonempty output')
            atomic(identity_path, dict(frozen=frozen, policy=policy))
        identity = digest(encoded(dict(frozen=frozen, policy=policy)))
        scratch = output / 'scratch'
        scratch.mkdir(exist_ok=True)
        last_beat = [0.]
        def heartbeat():
            now = time.monotonic()
            if now - last_beat[0] >= 1:
                (output / 'controller-heartbeat').touch()
                last_beat[0] = now
        heartbeat()
        # Short Unix-socket address: Darwin sockaddr_un has a small fixed limit.
        socket_root = Path.home() / 'tmp' / ('iqa-' + uuid.uuid4().hex[:12])
        socket_root.mkdir(parents=True)
        try:
            current = report(output, configs, identity, policy['max_attempts'], heartbeat)
            if args.action == 'run':
                while not current['complete'] and not STOP:
                    # One pass visits every pending cell before any retry.
                    for cell in sorted(current['cells'], key=lambda c: len(c['attempts'])):
                        heartbeat()
                        if cell['status'] != 'pending' or STOP:
                            continue
                        config = cell['config']
                        admission = locked(policy['lock'], wait=True, heartbeat=heartbeat) if policy['lock'] else contextlib.nullcontext(None)
                        with admission as heavy_fd:
                            # Hashing model assets is substantial CPU work. Serialize it
                            # with the worker, outside every cold/warm measurement clock.
                            verify_assets(config)
                            if file_hash(args.manifest) != frozen['manifest_sha256'] or file_hash(binary) != frozen['binary_sha256']:
                                raise Refusal('manifest/binary content changed during run')
                            for role in ('reference', 'distorted'):
                                if file_hash(config[role]) != config[role + '_sha256']:
                                    raise Refusal('active input changed during run')
                            directory, records = attempts(output, config, identity)
                            number = len(records) + 1
                            attempt = directory / f'attempt-{number:04d}'
                            attempt.mkdir()
                            atomic(attempt / 'config.json', config)
                            command = [str(binary), str(attempt / 'config.json')]
                            if policy['wrapper']:
                                command = [policy['wrapper'], '--mem', policy['memory'], '--jobs', str(policy['jobs']), '--', *command]
                            intent = dict(identity=identity, cell_id=config['durable_cell_id'], attempt=number,
                                          binary=str(binary), scratch=str(scratch), socket_root=str(socket_root), command=command,
                                          wrapper=policy['wrapper'], memory=policy['memory'], jobs=policy['jobs'],
                                          resource_policy='caller-supplied wrapper; inspect log for actual guard' if policy['wrapper'] else 'no resource wrapper or memory cap',
                                          controller_before=observation(), input_scratch=str(scratch))
                            atomic(attempt / 'intent.json', intent)
                            read_fd, write_fd = os.pipe()
                            fds = [output_fd, read_fd] + ([heavy_fd] if heavy_fd is not None else [])
                            with (attempt / 'supervisor.log').open('xb') as log:
                                child = subprocess.Popen([sys.executable, SELF, '_watchdog', str(attempt),
                                                          str(read_fd), ','.join(map(str, fds)), str(policy['timeout_seconds'])],
                                                         pass_fds=fds, stdout=log, stderr=subprocess.STDOUT,
                                                         start_new_session=True)
                                os.close(read_fd)
                                try:
                                    while child.poll() is None:
                                        heartbeat()
                                        if STOP:
                                            break
                                        try:
                                            os.write(write_fd, b'.')
                                        except BrokenPipeError:
                                            break
                                        time.sleep(.1)
                                finally:
                                    os.close(write_fd)
                                    child.wait(timeout=policy['timeout_seconds'] + 15)
                            if not (attempt / 'result.json').exists():
                                # Watchdog unexpectedly died; inner guard has its own
                                # heartbeat wall bound. Keep the advisory lock until gone.
                                time.sleep(HEARTBEAT + GRACE + 1)
                                raise RuntimeError('watchdog failed; diagnostics retained for restart')
                            if read_result(attempt / 'result.json').get('cleanup_verified') is False:
                                raise RuntimeError('process-group cleanup unverified')
                        _, history = attempts(output, config, identity)
                        final = next((r for r in history if r['status'] in ('ok', 'unsupported')), None)
                        cell.update(attempts=history, row=final['row'] if final else None,
                                    status=final['status'] if final else ('failed' if len(history) >= policy['max_attempts'] else 'pending'))
                        current['complete'] = all(c['status'] != 'pending' for c in current['cells'])
                        atomic(output / 'progress.json', dict(identity=identity, complete=current['complete'],
                               latest_cell=cell['cell_id'], statuses={s:sum(c['status']==s for c in current['cells'])
                               for s in ('ok','unsupported','failed','pending')}))
            current = report(output, configs, identity, policy['max_attempts'], heartbeat)
            print(json.dumps(dict(complete=current['complete'], qualified=current['qualified'],
                                  statuses={s: sum(c['status'] == s for c in current['cells'])
                                            for s in ('ok', 'unsupported', 'failed', 'pending')},
                                  report=str(output / 'report.json'))))
        finally:
            # Only the socket belongs here. All substantive evidence lives in OUTPUT.
            for path in socket_root.iterdir():
                path.unlink()
            socket_root.rmdir()
    return 128 + STOP if STOP else 0


def main():
    global STOP
    if len(sys.argv) > 1 and sys.argv[1].startswith('_'):
        action, attempt = sys.argv[1], Path(sys.argv[2])
        if action == '_worker':
            return worker_exec(attempt)
        if action == '_payload':
            return payload(attempt, sys.argv[3], float(sys.argv[4]))
        if action == '_anchor':
            return anchor(attempt, sys.argv[3], float(sys.argv[4]), int(sys.argv[5]))
        if action == '_watchdog':
            return watchdog(attempt, int(sys.argv[3]), sys.argv[4], float(sys.argv[5]))
        raise Refusal('unknown internal command')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'status'))
    parser.add_argument('manifest')
    parser.add_argument('output')
    parser.add_argument('--timeout-seconds', type=float, default=1100)
    parser.add_argument('--max-attempts', type=int, default=2)
    parser.add_argument('--lock')
    parser.add_argument('--run-heavy')
    parser.add_argument('--memory', default='12G')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if not 0 < args.timeout_seconds <= 1100 or not integer(args.max_attempts) or not integer(args.jobs) or args.jobs > (os.cpu_count() or 1):
        raise Refusal('positive jobs/attempts and timeout <=1100 required')
    def stop(number, _frame):
        global STOP
        STOP = number
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, stop)
    return run(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except InterruptedError as error:
        print(f'durable: interrupted: {error}', file=sys.stderr)
        sys.exit(128 + STOP if STOP else 1)
    except (Refusal, ValueError, OSError, KeyError, TypeError) as error:
        print(f'durable: permanent refusal: {error}', file=sys.stderr)
        sys.exit(2)
