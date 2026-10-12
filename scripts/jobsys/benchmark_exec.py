#!/usr/bin/env python3
"""ZEN_EXEC adapter for one zenbench observation; zenfleet owns queue/leases.

All outcome receipts (including wrong-output, busy and timeout) are durable
observations. Exit zero means the observation was stored, NOT that it is valid
performance evidence. The packet's status is the measurement verdict.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

FIELDS = ("kind", "bundle_sha", "runner_sha", "experiment", "machine", "domain",
          "replicate", "backend", "timeout_seconds", "admission_seconds")
BACKENDS = ("wall", "os_peak", "heaptrack", "driver")

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def sha(value):
    if not isinstance(value, str) or not re.fullmatch("[a-f0-9]{64}", value):
        raise ValueError("invalid SHA-256")
    return value

def token(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,256}", value):
        raise ValueError("invalid benchmark identity token")
    return value

def canonical_job_id(kind, inputs):
    wire = {"inputs": sorted(set(inputs)), "kind": {k: kind[k] for k in FIELDS}}
    return hashlib.sha256(json.dumps(wire, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def validate_job(job):
    kind = job["kind"]
    if set(kind) != set(FIELDS) or kind["kind"] != "benchmark":
        raise ValueError("expected benchmark-v1 job fields")
    for key in ("bundle_sha", "runner_sha"):
        sha(kind[key])
    for key in ("experiment", "machine", "domain"):
        token(kind[key])
    if kind["backend"] not in BACKENDS:
        raise ValueError("unsupported measurement backend")
    if type(kind["replicate"]) is not int or not 0 <= kind["replicate"] <= 0xffffffff:
        raise ValueError("invalid replicate")
    if any(type(kind[k]) is not int or not 0 <= kind[k] <= 86400 for k in ("timeout_seconds", "admission_seconds")) or kind["timeout_seconds"] == 0:
        raise ValueError("invalid timeout/admission budget")
    if sorted(set(job["inputs"])) != sorted({kind["bundle_sha"], kind["runner_sha"]}):
        raise ValueError("benchmark inputs must bind bundle and runner")
    if "job_id" in job and job["job_id"] != canonical_job_id(kind, job["inputs"]):
        raise ValueError("job ID does not bind the declared observation")
    return kind

class Store:
    def __init__(self, root):
        self.root = Path(os.environ.get("ZEN_BLOBS_DIR", root / "blobs")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.uri = os.environ.get("ZENBENCH_BLOB_URI")
        if self.uri and not self.uri.startswith("s3://"):
            raise ValueError("ZENBENCH_BLOB_URI must be the fleet s3://.../blobs prefix")

    def transfer(self, src, dest):
        cmd = ["s5cmd"]
        endpoint = os.environ.get("ZEN_S3_ENDPOINT") or os.environ.get("ZEN_R2_ENDPOINT")
        if endpoint:
            cmd += ["--endpoint-url", endpoint]
        subprocess.run(cmd + ["cp", str(src), str(dest)], check=True, timeout=120, stdout=subprocess.DEVNULL)

    def get(self, value):
        value = sha(value)
        path = self.root / value
        if not path.exists():
            if not self.uri:
                raise FileNotFoundError(f"missing blob {value}; stage it or configure ZENBENCH_BLOB_URI")
            temp = self.root / f"download-{uuid.uuid4().hex}"
            self.transfer(f"{self.uri.rstrip('/')}/{value}", temp)
            if digest(temp) != value:
                raise ValueError("download hash mismatch; partial file retained")
            # Identical concurrent content is safe; no source/attempt is removed.
            temp.replace(path)
        if path.is_symlink() or digest(path) != value:
            raise ValueError(f"corrupt blob {value}")
        return path

    def put(self, source):
        value = digest(source)
        dest = self.root / value
        if dest.exists():
            if dest.is_symlink() or digest(dest) != value:
                raise ValueError("existing content-addressed blob differs")
        else:
            temp = self.root / f"upload-{uuid.uuid4().hex}"
            shutil.copyfile(source, temp)
            if digest(temp) != value:
                raise ValueError("artifact changed during copy")
            temp.replace(dest)
        if self.uri:
            self.transfer(dest, f"{self.uri.rstrip('/')}/{value}")
        return value

def materialize(store, manifest_sha, destination):
    source = store.get(manifest_sha)
    if source.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("bundle manifest exceeds 8 MiB")
    manifest = json.loads(source.read_bytes())
    if manifest["schema"] != 1 or not manifest["files"]:
        raise ValueError("unsupported bundle manifest")
    destination.mkdir(parents=True, exist_ok=False)
    seen = set()
    for name, artifact in manifest["files"].items():
        parts = PurePosixPath(name).parts
        if not parts or name.startswith('/') or '\\' in name or ':' in name or any(p in ('..', '.') for p in name.split('/')) or name.lower() in seen or name.lower() == 'bundle.json':
            raise ValueError("unsafe/colliding bundle path")
        seen.add(name.lower())
        src = store.get(artifact["sha256"])
        if src.stat().st_size != artifact["bytes"]:
            raise ValueError("bundle file length mismatch")
        dest = destination / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        dest.chmod(0o755 if artifact["executable"] else 0o644)
    shutil.copyfile(source, destination / 'bundle.json')
    return manifest

def capabilities(runner):
    result = subprocess.run([runner, "capabilities"], capture_output=True, check=True, timeout=15)
    caps = json.loads(result.stdout)
    if caps["schema"] != 1 or caps["protocols"].get("fleet_attempt") != 1:
        raise ValueError("runner lacks fleet-attempt-v1")
    machine = token(os.environ["ZENBENCH_MACHINE"])
    domain = token(os.environ["ZENBENCH_DOMAIN"])
    placement = caps["placement"].get("Ok")
    if placement is None:
        raise ValueError(f"CPU placement refused: {caps['placement']}")
    actual = placement.get("core_class")
    if domain == "unrestricted":
        if placement["policy"] != "unrestricted":
            raise ValueError("unrestricted domain needs explicit unrestricted policy")
    elif actual != domain and not (domain == 'uniform' and placement['status'] == 'uniform_hardware'):
        raise ValueError("configured domain differs from verified CPU class")
    tokens = {"benchmark-v1", f"benchmark-runner:{caps['runner']['sha256']}",
              f"benchmark-machine:{machine}", f"benchmark-domain:{domain}",
              "benchmark-backend:wall", "benchmark-backend:driver"}
    if caps["memory"]["heaptrack"]["implemented"] and caps["memory"]["heaptrack"]["recorder"] and caps["memory"]["heaptrack"]["analyzer"]:
        tokens.add("benchmark-backend:heaptrack")
    if caps["memory"]["os_peak"]["implemented"] and (caps["runner"]["os"] == "windows" or caps["memory"]["os_peak"]["tool"]):
        tokens.add("benchmark-backend:os_peak")
    return caps, tokens

class ParentWatch:
    """A Windows process handle survives exit/PID reuse; getppid alone does not."""
    def __enter__(self):
        self.pid = os.getppid()
        self.handle = None
        if os.name == 'nt':
            import _winapi
            self.handle = _winapi.OpenProcess(_winapi.SYNCHRONIZE, False, self.pid)
        return self

    def alive(self):
        if self.handle is not None:
            import _winapi
            return _winapi.WaitForSingleObject(self.handle, 0) == _winapi.WAIT_TIMEOUT
        return os.getppid() == self.pid

    def __exit__(self, *exc):
        if self.handle is not None:
            import _winapi
            _winapi.CloseHandle(self.handle)

def execute(job):
    with ParentWatch() as parent:
        return _execute(job, parent)

def _execute(job, parent):
    kind = validate_job(job)
    runner = str(Path(os.environ["ZENBENCH_BIN"]).resolve())
    caps, supported = capabilities(runner)
    needed = {"benchmark-v1", f"benchmark-runner:{kind['runner_sha']}",
              f"benchmark-machine:{kind['machine']}", f"benchmark-domain:{kind['domain']}",
              f"benchmark-backend:{kind['backend']}"}
    if not needed <= supported:
        raise ValueError(f"worker does not satisfy benchmark contract: {sorted(needed-supported)}")
    root = Path(os.environ["ZENBENCH_WORK_DIR"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    store = Store(root)
    attempt = uuid.uuid4().hex
    bundle = root / "bundles" / attempt
    manifest = materialize(store, kind["bundle_sha"], bundle)
    if manifest['plan']['backend'] != kind['backend'] or manifest['plan']['timeout_seconds'] != kind['timeout_seconds']:
        raise ValueError("bundle and job measurement contracts differ")
    output = root / "attempts" / attempt
    cancel = root / f"cancel-{attempt}"
    output.parent.mkdir(parents=True, exist_ok=True)
    interrupted = False
    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        cancel.touch(exist_ok=True)
    old_handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    args = [runner,"fleet-run",str(bundle),"--output",str(output),"--experiment",kind['experiment'],
            "--machine",kind['machine'],"--domain",kind['domain'],"--replicate",str(kind['replicate']),
            "--attempt",attempt,"--wait",str(kind['admission_seconds']),"--cancel",str(cancel)]
    try:
        with (root/f"runner-{attempt}.stdout").open('wb') as stdout, (root/f"runner-{attempt}.stderr").open('wb') as stderr:
            process = subprocess.Popen(args, stdout=stdout, stderr=stderr)
            began = time.monotonic()
            deadline = kind['admission_seconds'] + 2*kind['timeout_seconds'] + 60
            while process.poll() is None:
                if not parent.alive():
                    interrupted = True
                    cancel.touch(exist_ok=True)
                if time.monotonic()-began > deadline:
                    cancel.touch(exist_ok=True)
                # Leave a cleanup grace interval before the worker watchdog.
                if time.monotonic()-began > deadline+15:
                    raise TimeoutError("runner did not finish bounded cancellation; reservation and attempt retained")
                time.sleep(0.1)
        packet = json.loads((output/'packet.json').read_bytes())
        if packet['attempt'] != attempt or packet.get('bundle_sha256',kind['bundle_sha']) != kind['bundle_sha']:
            raise ValueError("attempt receipt identity mismatch")
        artifacts = {}
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != '.gitignore':
                artifacts[path.name] = {"sha256":store.put(path),"bytes":path.stat().st_size}
        for stream in ('stdout', 'stderr'):
            path = root / f'runner-{attempt}.{stream}'
            artifacts[f'runner.{stream}'] = {"sha256":store.put(path),"bytes":path.stat().st_size}
        receipt = {"kind":"zenfleet_benchmark_observation","schema":1,"job_id":canonical_job_id(kind,job["inputs"]),
                   "attempt":attempt,"status":packet['status'],"packet":artifacts['packet.json'],
                   "artifacts":artifacts,"interrupted":interrupted,"runner_exit":process.returncode,
                   "retention":{"regenerability":"not_regenerable","references":[v['sha256'] for v in artifacts.values()]},
                   "commands":{"decode":"fetch packet blob, then read its JSON; raw evidence is hash-addressed",
                               "validity":"require packet.status == complete before comparison"}}
        encoded = json.dumps(receipt,separators=(',',':')).encode()
        receipts = root / 'receipts'
        receipts.mkdir(exist_ok=True)
        receipt_file = receipts / f'{attempt}.json'
        receipt_file.write_bytes(encoded)
        receipt_sha = store.put(receipt_file)
        print(f'benchmark observation receipt sha256={receipt_sha} status={packet["status"]}', file=sys.stderr)
        return encoded
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)

def stage(bundle):
    runner = Path(os.environ['ZENBENCH_BIN']).resolve()
    subprocess.run([str(runner), 'bundle-check', str(bundle)], check=True, stdout=subprocess.DEVNULL)
    store = Store(Path(os.environ['ZENBENCH_WORK_DIR']))
    manifest = json.loads((bundle / 'bundle.json').read_bytes())
    for name, artifact in manifest['files'].items():
        if store.put(bundle / name) != artifact['sha256']:
            raise ValueError('bundle changed while staging')
    return {'bundle_sha': store.put(bundle / 'bundle.json'), 'runner_sha': store.put(runner)}

def collect(receipt, output):
    store = Store(Path(os.environ['ZENBENCH_WORK_DIR']))
    source = store.get(receipt) if re.fullmatch('[a-f0-9]{64}', receipt) else Path(receipt)
    value = json.loads(source.read_bytes())
    if value['kind'] != 'zenfleet_benchmark_observation' or value['schema'] != 1:
        raise ValueError('expected benchmark observation receipt')
    output.mkdir(parents=True, exist_ok=False)
    for name, artifact in value['artifacts'].items():
        if not re.fullmatch(r'[A-Za-z0-9._-]+', name) or name in ('.', '..'):
            raise ValueError('invalid observation artifact name')
        src = store.get(artifact['sha256'])
        if src.stat().st_size != artifact['bytes']:
            raise ValueError('observation artifact length mismatch')
        shutil.copyfile(src, output / name)
    packet = json.loads((output / 'packet.json').read_bytes())
    if packet['attempt'] != value['attempt'] or packet['status'] != value['status']:
        raise ValueError('receipt differs from packet')
    return value

def main():
    if sys.argv[1:] == ['capabilities']:
        _, tokens = capabilities(os.environ['ZENBENCH_BIN'])
        print('\n'.join(sorted(tokens)))
    elif len(sys.argv) == 1:
        sys.stdout.buffer.write(execute(json.load(sys.stdin)))
    elif len(sys.argv) == 3 and sys.argv[1] == 'cancel':
        attempt = token(sys.argv[2])
        (Path(os.environ['ZENBENCH_WORK_DIR']) / f'cancel-{attempt}').touch(exist_ok=True)
    elif len(sys.argv) == 3 and sys.argv[1] == 'stage':
        print(json.dumps(stage(Path(sys.argv[2]).resolve()), separators=(',', ':')))
    elif len(sys.argv) == 4 and sys.argv[1] == 'collect':
        print(json.dumps(collect(sys.argv[2], Path(sys.argv[3])), separators=(',', ':')))
    else:
        raise ValueError('usage: benchmark_exec.py [capabilities | stage BUNDLE | collect RECEIPT_OR_SHA NEW_DIRECTORY | cancel ATTEMPT]; no arguments executes a stdin job')

if __name__ == '__main__':
    try:
        main()
    except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
        print(f'ZEN_ERROR_CLASS: worker_lost {exc}', file=sys.stderr)
        sys.exit(2)
    except (ValueError, KeyError) as exc:
        print(f'ZEN_ERROR_CLASS: unknown {exc}', file=sys.stderr)
        sys.exit(2)
