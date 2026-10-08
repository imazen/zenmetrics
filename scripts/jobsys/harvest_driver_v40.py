#!/usr/bin/env python3
"""V40 copy of the budget-bound D1 incremental driver; no existing driver edits.

Never schedules work. --dry-run-blob exercises the same verifier without S3
or installation. No local-smoke override is exposed.
"""
import argparse
import collections
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

A = None
R = Path('/var/tmp/fitv2')
POT = Path('/var/tmp/rev4-featpot')
PROGRAM = None
MANIFESTS = {}
INSPECTOR = None
hfc = None


def sha(p):
    return hfc.digest(Path(p))


def configuration(js, manifest):
    if js not in MANIFESTS or sha(manifest) != MANIFESTS[js]:
        raise ValueError('unregistered V40 manifest; local-smoke cannot install')
    if sha(A / 'program.tar.gz') != PROGRAM or sha(A / 'bin/inspect_qualified_checkpoint') != INSPECTOR:
        raise ValueError('V40 program/inspector pin changed')
    import tarfile
    with tarfile.open(A / 'program.tar.gz') as archive:
        for name in ('harvest_fit_cells.py', 'qualified_fit_contract.py', 'fit_paths.py'):
            expected = archive.extractfile(name).read()
            if (A / 'committed-tools' / name).read_bytes() != expected:
                raise ValueError('committed verifier differs from pinned program: ' + name)
    jobs = json.loads(Path(manifest).read_text())
    for c in jobs:
        k, name = c['kind'], c['cell']['image_path']
        argv = k['argv']
        if (k['program_sha'] != PROGRAM or k['data_sha'] not in DATA or '--local-smoke-budget' in argv
                or '--strict-admission' not in argv or '--train-only' not in argv):
            raise ValueError('requires registered full-budget strict D1 cell')
        dest = POT / hfc.blob_root(k) / name
        if dest != Path(argv[argv.index('--dest') + 1]):
            raise ValueError('manifest root/dest and harvest install root differ')
    return jobs


def installed(c):
    k = c['kind']
    d = POT / hfc.blob_root(k) / c['cell']['image_path']
    try:
        receipt = json.loads((d / 'fleet_receipt.json').read_text())
        result = json.loads((d / 'result.json').read_text())
        return (all(receipt[x] == k[x] for x in ('program_sha', 'data_sha', 'argv_sha'))
                and result.get('execution_contract') != 'local-smoke'
                and receipt['files']['result.json'] == sha(d / 'result.json')
                and result['selected_bake_sha256'] == sha(d / 'refit' / Path(result['selected_bake']).name))
    except (OSError, ValueError, KeyError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('jobset')
    p.add_argument('manifest', type=Path)
    p.add_argument('--install', action='store_true')
    p.add_argument('--require-all', action='store_true')
    p.add_argument('--status', action='store_true')
    p.add_argument('--check-config', action='store_true')
    p.add_argument('--dry-run-blob', type=Path)
    p.add_argument('--scratch', type=Path)
    args = p.parse_args()
    global A, PROGRAM, DATA, MANIFESTS, INSPECTOR, hfc
    A = args.bundle
    identities = json.loads((A/'PACKAGE_PINNED.json').read_text())
    PROGRAM, DATA, INSPECTOR = identities['program_sha'], identities['data_shas'], identities['inspector_sha']
    MANIFESTS = identities['manifests']
    sys.path.insert(0, str(A/'committed-tools'))
    import harvest_fit_cells
    hfc = harvest_fit_cells

    if args.dry_run_blob:
        # Exercise the canonical verifier on actual image-produced smoke bytes.
        # Installation is unreachable, even when --install was requested.
        jobs = json.loads(args.manifest.read_text())
        if len(jobs) != 1:
            raise ValueError('offline blob regression requires one exact cell')
        c = jobs[0]
        stage = args.scratch
        if stage is None or stage.exists():
            raise ValueError('offline regression requires a fresh explicit --scratch')
        hfc.verify_blob(args.dry_run_blob, stage, c['cell']['image_path'], c['kind'],
                        program_archive=A / 'program.tar.gz',
                        checkpoint_inspector=A / 'bin/inspect_qualified_checkpoint')
        raise ValueError('offline regression never installs a cell')
    jobs = configuration(args.jobset, args.manifest)
    if args.check_config:
        print(json.dumps({'cells': len(jobs), 'install_roots': sorted({hfc.blob_root(c['kind']) for c in jobs}),
                          'program_archive': str(A / 'program.tar.gz'),
                          'checkpoint_inspector': str(A / 'bin/inspect_qualified_checkpoint')}))
        return
    work = R / 'harvest-v40' / args.jobset
    status = R / 'status-v40' / args.jobset
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        endpoint = os.environ['EP']
        status.mkdir(parents=True, exist_ok=True)
        subprocess.run(['s5cmd', '--endpoint-url', endpoint, 'sync',
                        f's3://zentrain/jobs/{args.jobset}/ledger/*', str(status) + '/'], check=True)
        ids = subprocess.check_output(['/var/tmp/fleet-fits/fleetbin/zenfleet-ctl', 'ids',
                                       '--manifest', str(args.manifest)], text=True)
        full_ids = work / 'full-ids.tsv'
        full_ids.write_text(ids)
        pairs = hfc.read_ids(full_ids)
        if len(pairs) != len(jobs) or any(name != c['cell']['image_path'] for (_, name), c in zip(pairs, jobs)):
            raise ValueError('canonical manifest/IDs mismatch')
        rows = [row for f in sorted(status.glob('*.parquet')) for row in pq.read_table(f).to_pylist()]
        relevant = {jid for jid, _ in pairs}
        rows = [row for row in rows if row['job_id'] in relevant]
        latest, done = {}, set()
        for row in rows:
            jid = row['job_id']
            if row['status'] == 'done' and row.get('output_sha'):
                done.add(jid)
            if jid not in latest or row['ts'] > latest[jid]['ts']:
                latest[jid] = row
        counts = collections.Counter('done' if jid in done else latest[jid]['status'] for jid in latest)
        print(json.dumps({'jobset': args.jobset, 'ledger': dict(counts), 'done': len(done),
                          'total': len(jobs), 'installed': sum(installed(c) for c in jobs)}), flush=True)
        if args.status:
            return
        if args.require_all and len(done) != len(jobs):
            raise ValueError('all registered cells must be DONE before scoring')
        chosen = [(pair, c) for pair, c in zip(pairs, jobs) if pair[0] in done]
        if not chosen:
            return
        manifest = work / 'manifest.json'
        manifest.write_text(json.dumps([c for _, c in chosen]) + '\n')
        ledger = work / 'ledger.parquet'
        pq.write_table(pa.Table.from_pylist(rows), ledger)
        ids_path = work / 'ids.tsv'
        ids_path.write_text(''.join(f'{i}\t{jid}\t{name}\n' for i, ((jid, name), _) in enumerate(chosen)))
        cmd = [sys.executable, str(A / 'committed-tools/harvest_fit_cells.py'), '--manifest', str(manifest),
               '--ids', str(ids_path), '--ledger', str(ledger), '--blobs-prefix', f's3://zentrain/jobs/{args.jobset}/blobs',
               '--endpoint', endpoint, '--scratch', str(work / 'scratch'),
               '--program-archive', str(A / 'program.tar.gz'),
               '--checkpoint-inspector', str(A / 'bin/inspect_qualified_checkpoint')]
        if args.install:
            cmd += ['--install', '--rescue-root', str(R / 'original-era-v40' / args.jobset)]
        subprocess.run(cmd, check=True)
        if args.install and args.require_all and not all(installed(c) for c in jobs):
            raise ValueError('installed cell identity/checkpoint check failed')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'V40 HARVEST REFUSED: {exc}', file=sys.stderr)
        sys.exit(1)
