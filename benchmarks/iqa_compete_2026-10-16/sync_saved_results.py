#!/usr/bin/env python3
"""Capture three hosts, publish allowlisted timing rows, commit and verify push.

The caller supplies private host/path configuration outside git. The systemd
service serializes cycles. Repository mutations also require .workongoing and
an empty working copy; contention is reported, never overridden.

A host that is gone for good (config `retired: {"since": DATE, "capture": DIR}`) is
not contacted: its final complete capture is reused every cycle and sync.json
names it under `retired_hosts`, so its pending cells show as never run rather
than as unfinished work. Cycle directories older than the newest KEEP_CYCLES are
removed after each cycle; they are copies of host-held evidence, and a retired
host's capture lives outside them.

A cycle that holds new evidence but cannot publish it (lock held, dirty
checkout, unintegrated parent, refused or failed push) exits nonzero, so the
service shows failed, and status.json records the reason class, when the stall
began and how many cycles it has blocked. Detail (diff summary, push-log tail)
stays in the private status; `reason` is a fixed class safe to display publicly.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

FILES = ('capture_saved_results.py', 'export_chart_snapshot.py', 'publish_saved_results.py',
         'durable.py', 'admission_recovery.py')
DESTINATION = 'benchmarks/iqa_saved_latest'
BLOCKED_EXIT = 2
KEEP_CYCLES = 12
CYCLE_NAME = re.compile(r'\d{8}T\d{12}Z')
# scripts/safe_push.sh exit codes -> publicly displayable reason classes.
PUSH_REASONS = {2: 'push revision unresolved', 3: 'push not a fast-forward of master', 4: 'push fetch failed',
                5: 'push transport failed', 6: 'push not verified on remote', 7: 'push refused by hygiene guard',
                8: 'push refused by lock guard', 9: 'push refused by dashboard guard'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(argv, **kwargs):
    return subprocess.check_output(list(map(str, argv)), text=True, **kwargs).strip()


def ssh(host, argv):
    return run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, shlex.join(list(map(str, argv)))])


def fingerprint(folder):
    """Ignore capture timestamps/report identities; commit when measured evidence changes."""
    import csv
    digest = hashlib.sha256()
    for path in sorted(list(folder.glob('*-observations.csv'))+list(folder.glob('*-rounds.csv'))):
        digest.update(path.name.encode())
        rows = []
        with path.open() as stream:
            for row in csv.DictReader(stream):
                rows.append({k: v for k, v in row.items() if k not in ('capture_utc', 'source')})
        for row in sorted(rows, key=lambda r: json.dumps(r, sort_keys=True)):
            digest.update(json.dumps(row, sort_keys=True).encode())
    coverage = folder / 'coverage.json'
    if coverage.exists():
        # Per-host logical cell counts; the capture windows beside them change every cycle.
        cells = {h: v.get('cells') for h, v in json.loads(coverage.read_text()).items()}
        digest.update(json.dumps(cells, sort_keys=True).encode())
    return digest.hexdigest()


def public_checks(folder):
    import csv
    hashes = json.loads((folder / 'SHA256SUMS.json').read_text())
    for name, expected in hashes.items():
        if Path(name).name != name or hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError('public projection checksum differs: ' + name)
    for host in ('mac', 'i265', 'r7900x'):
        with (folder / (host + '-observations.csv')).open() as stream:
            reader = csv.DictReader(stream)
            if {'score', 'score_bits', 'reference', 'distorted', 'weights'} & set(reader.fieldnames or ()):
                raise ValueError('unexpected private field in public projection')
            for row in reader:
                if row['protocol'] != 'canonical-cold3-warm10-100ms-scalar' or row['only_mode'] != 'score':
                    raise ValueError('noncanonical projection row')
                if int(row['threads']) not in (1, 8) or max(int(row['width']), int(row['height'])) >= 7680:
                    raise ValueError('projection contains excluded geometry or workers')


def capture(config, host, cycle, tools):
    entry = config['hosts'][host]
    spec = cycle / (host + '-spec.json')
    spec.write_text(json.dumps(entry['campaigns'], indent=2) + '\n')
    output = cycle / 'captures' / host
    if entry.get('retired'):
        frozen = Path(entry['retired']['capture'])
        receipt = json.loads((frozen / 'capture-receipt.json').read_text())
        if receipt['host'] != host:
            raise ValueError('retired capture belongs to ' + receipt['host'])
        output.symlink_to(frozen, target_is_directory=True)
        return receipt
    if entry.get('ssh'):
        remote = entry['scratch'].rstrip('/') + '/' + cycle.name
        ssh(entry['ssh'], ['mkdir', '-p', remote])
        subprocess.run(['scp', '-q', *[str(tools / n) for n in FILES], str(spec), entry['ssh'] + ':' + remote + '/'], check=True)
        ssh(entry['ssh'], ['nice', '-n', '19', entry.get('python', 'python3'), remote + '/capture_saved_results.py',
                          remote + '/' + spec.name, remote + '/capture', '--host', host])
        subprocess.run(['scp', '-q', '-r', entry['ssh'] + ':' + remote + '/capture', str(output)], check=True)
    else:
        subprocess.run([sys.executable, str(tools / FILES[0]), str(spec), str(output), '--host', host], check=True)
    return json.loads((output / 'capture-receipt.json').read_text())


def recover_published(repo, revision):
    """Record a manually repaired push only when its exact data is on master."""
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('full verified commit hash required')
    subprocess.run(['git', 'merge-base', '--is-ancestor', revision, 'origin/master'], cwd=repo, check=True)
    folder = repo / DESTINATION
    public_checks(folder)
    for path in folder.iterdir():
        if not path.is_file():
            raise ValueError('unexpected directory in saved publication')
        committed = subprocess.check_output(['git', 'show', revision + ':' + DESTINATION + '/' + path.name], cwd=repo)
        if path.read_bytes() != committed:
            raise ValueError('working data differs from verified publication: ' + path.name)
    return dict(state='committed and remote verified', revision=revision, updated_utc=now(),
                evidence_fingerprint=fingerprint(folder), recovery='exact committed data and remote ancestry verified')


def guarded_push(repo, revision, log):
    """Retry a forward master race even when it happens after guard validation."""
    with log.open('w') as stream:
        for attempt in range(3):
            before = run(['git', 'rev-parse', 'origin/master'], cwd=repo)
            pushed = subprocess.run(['bash', 'scripts/safe_push.sh', '-r', revision],
                                    cwd=repo, stdout=stream, stderr=subprocess.STDOUT)
            if pushed.returncode == 0:
                return revision
            error = subprocess.CalledProcessError(pushed.returncode, 'safe_push; inspect ' + str(log))
            if pushed.returncode not in (3, 5) or attempt == 2:
                raise error
            run(['jj', 'git', 'fetch'], cwd=repo)
            after = run(['git', 'rev-parse', 'origin/master'], cwd=repo)
            if pushed.returncode == 5:
                # Exit5 also covers transport failures. Rebase only a proved
                # forward remote advance; unrelated push failures stay visible.
                if before == after or subprocess.run(
                        ['git', 'merge-base', '--is-ancestor', before, after], cwd=repo).returncode:
                    raise error
            run(['jj', 'rebase', '-r', revision, '-d', 'master@origin'], cwd=repo)
            revision = run(['jj', 'log', '-r', '@', '--no-graph', '-T', 'commit_id'], cwd=repo)


def prune(root, keep):
    """Remove old cycle directories; the newest KEEP_CYCLES and any named in `keep` stay."""
    cycles = sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink() and CYCLE_NAME.fullmatch(p.name))
    for path in cycles[:-KEEP_CYCLES]:
        if path.name not in keep:
            shutil.rmtree(path)


def publish(repo, projection, digest, cycle, config=None):
    lock = repo / '.workongoing'
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return dict(state='captured; repository owned by another lane', captured=str(cycle), blocked=True,
                    reason='checkout locked by another lane', detail=lock.read_text(errors='replace').strip()[:300])
    owner = now() + ' Codex-result-sync five-minute results publication\n'
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(owner)
        dirty = run(['jj', 'diff', '--summary', '--color', 'never'], cwd=repo)
        if dirty:
            return dict(state='captured; uncommitted repository changes left untouched', captured=str(cycle), blocked=True,
                        reason='checkout has uncommitted changes', detail=dirty[:2000])
        run(['jj', 'git', 'fetch'], cwd=repo)
        parent = run(['jj', 'log', '-r', '@-', '--no-graph', '-T', 'commit_id'], cwd=repo)
        if subprocess.run(['git', 'merge-base', '--is-ancestor', parent, 'origin/master'], cwd=repo).returncode:
            return dict(state='captured; unpublished source commits require owner integration', captured=str(cycle), blocked=True,
                        reason='checkout parent not on master', detail=parent)
        run(['jj', 'new', 'master@origin', '-m', 'data: synchronize measured IQA results'], cwd=repo)
        destination = repo / DESTINATION
        destination.mkdir(exist_ok=True)
        for source in projection.iterdir():
            shutil.copy2(source, destination / source.name)
        index = dict(updated_utc=now(), evidence_fingerprint=digest,
                     capture_windows={h: json.loads((cycle / 'captures' / h / 'capture-receipt.json').read_text())['capture_finished_utc']
                                      for h in ('mac', 'i265', 'r7900x')},
                     qualification='Use preferred=True qualified rows; gaps remain explicit; no implementation/protocol pooling')
        retired = {h: dict(since=e['retired']['since']) for h, e in (config or {}).get('hosts', {}).items() if e.get('retired')}
        if retired:
            index['retired_hosts'] = retired
        (destination / 'sync.json').write_text(json.dumps(index, indent=2) + '\n')
        run(['jj', 'describe', '-m', 'data: synchronize three-host IQA results ' + cycle.name], cwd=repo)
        revision = run(['jj', 'log', '-r', '@', '--no-graph', '-T', 'commit_id'], cwd=repo)
        revision = guarded_push(repo, revision, cycle / 'push.log')
        subprocess.run(['git', 'merge-base', '--is-ancestor', revision, 'origin/master'], cwd=repo, check=True)
        run(['jj', 'new', 'master@origin'], cwd=repo)
        return dict(state='committed and remote verified', revision=revision, updated_utc=now(), evidence_fingerprint=digest)
    finally:
        if lock.exists() and lock.read_text() == owner:
            lock.unlink()


def push_failure(error, cycle):
    """Reason class and private detail for an exception raised while publishing."""
    import traceback
    reason = 'sync cycle failed (%s)' % type(error).__name__
    command = str(getattr(error, 'cmd', ''))
    if isinstance(error, subprocess.CalledProcessError) and command.startswith('safe_push'):
        reason = PUSH_REASONS.get(error.returncode, 'push failed (exit %d)' % error.returncode)
    elif isinstance(error, subprocess.CalledProcessError) and re.search(r"\b(ssh|scp)\b", command):
        reason = 'host capture failed'
    log = cycle / 'push.log'
    tail = log.read_text(errors='replace').splitlines()[-25:] if log.exists() else []
    return reason, '\n'.join([traceback.format_exc().rstrip()] + tail)


def stalled(previous, result):
    """Carry the stall start and the first failure forward until a cycle publishes."""
    if previous.get('blocked'):
        result['stalled_since'] = previous.get('stalled_since') or previous.get('updated_utc')
        result['blocked_cycles'] = previous.get('blocked_cycles', 1) + 1
        result['first_reason'] = previous.get('first_reason') or previous.get('reason')
        result['first_detail'] = previous.get('first_detail') or previous.get('detail')
    else:
        result.update(stalled_since=result['updated_utc'], blocked_cycles=1,
                      first_reason=result['reason'], first_detail=result.get('detail'))
    return result


def write_status(root, result):
    target = root / 'status.json'
    temporary = target.with_suffix('.new')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--recover-published', metavar='FULL_COMMIT',
                        help='Record an already repaired guarded push; verify ancestry and exact published bytes')
    a = parser.parse_args()
    config = json.loads(a.config.read_text())
    root = Path(config['scratch'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'cycle.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.recover_published:
            result = recover_published(Path(config['repo']), a.recover_published)
            for name in ('last-published.json', 'status.json'):
                target = root / name
                temporary = target.with_suffix('.new')
                temporary.write_text(json.dumps(result, indent=2) + '\n')
                temporary.replace(target)
            print(json.dumps(result))
            return
        prior = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
        cycle = root / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        (cycle / 'captures').mkdir(parents=True)
        repo = Path(config['repo'])
        # Tools are pinned to committed master, even when another lane edits the checkout.
        revision = run(['git', 'rev-parse', 'origin/master'], cwd=repo)
        tools = cycle / 'tools'
        tools.mkdir()
        for name in FILES:
            data = subprocess.check_output(['git', 'show', revision + ':benchmarks/iqa_compete_2026-10-16/' + name], cwd=repo)
            (tools / name).write_bytes(data)
        try:
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(lambda h: capture(config, h, cycle, tools), ('mac', 'i265', 'r7900x')))
            projection = cycle / 'public'
            subprocess.run([sys.executable, str(tools / FILES[2]), str(cycle / 'captures'),
                            str(repo / 'benchmarks/private_classical_2026-10-19'), str(projection)], check=True)
            public_checks(projection)
            digest = fingerprint(projection)
            state_path = root / 'last-published.json'
            previous = json.loads(state_path.read_text()) if state_path.exists() else {}
            if digest == previous.get('evidence_fingerprint'):
                result = dict(state='no new measured evidence', updated_utc=now(), captured=str(cycle), revision=previous.get('revision'))
            else:
                result = publish(repo, projection, digest, cycle, config)
                if result['state'] == 'committed and remote verified':
                    state_path.write_text(json.dumps(result, indent=2) + '\n')
        except Exception as error:
            reason, detail = push_failure(error, cycle)
            result = dict(state='failed; previous published data retained', error=str(error), captured=str(cycle),
                          blocked=True, reason=reason, detail=detail)
        result.setdefault('updated_utc', now())
        published = json.loads((root / 'last-published.json').read_text()) if (root / 'last-published.json').exists() else {}
        prune(root, {Path(published.get('captured') or '').name, cycle.name})
        if result.get('blocked'):
            result = stalled(prior, result)
            write_status(root, result)
            print(json.dumps(result))
            print('IQA RESULT SYNC BLOCKED since %s (%d cycles): %s\n%s' % (
                result['stalled_since'], result['blocked_cycles'], result['reason'], result.get('detail') or ''), file=sys.stderr)
            sys.exit(BLOCKED_EXIT)
        write_status(root, result)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
