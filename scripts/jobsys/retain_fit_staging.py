#!/usr/bin/env python3
"""Retain output inodes from already-running pre-retention FitCell images.

Compatibility only: makes no claims, signals no fits and removes no files.
Run in a bounded container with the existing disk scratch and host PID view.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


def identity(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return text[19] if text[0] != 'Z' else None
    except FileNotFoundError:
        return None


def retain(source, target, seen):
    count = 0
    for path in source.rglob('*'):
        try:
            if path.is_symlink() or not path.is_file():
                continue
            stat = path.stat()
            rel = path.relative_to(source)
            key = (str(rel), stat.st_dev, stat.st_ino)
            if key in seen:
                continue
            dest = target / rel.parent / (rel.name + f'.inode-{stat.st_ino}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, dest)
            except FileExistsError:
                if dest.stat().st_ino != path.stat().st_ino:
                    raise
            actual = dest.stat()
            seen.add((str(rel), actual.st_dev, actual.st_ino))
            count += 1
        except FileNotFoundError:
            continue
    return count


def launch():
    p = argparse.ArgumentParser(description='Attach bounded retention to verified live allocations')
    for flag in ('inventory','host','run','data-sha','arm','receipt'):
        p.add_argument('--'+flag, required=True)
    a = p.parse_args(sys.argv[2:])
    rows = [w for w in json.loads(Path(a.inventory).read_text())['workers']
            if w['host']==a.host and w['run']==a.run and w['kind']=='container']
    if not rows or not re.fullmatch('[0-9a-f]{64}',a.data_sha) or not re.fullmatch('[A-Za-z][A-Za-z0-9_+-]*',a.arm):
        raise ValueError('verified allocations and safe cell identity required')
    containers = json.loads(subprocess.check_output(['ssh',a.host,shlex.join(['docker','inspect',*[w['container'] for w in rows]])],text=True))
    scratch = set()
    for c,w in zip(containers,rows):
        env = dict(v.split('=',1) for v in c['Config']['Env'] if '=' in v)
        if not c['State']['Running'] or c['State']['Pid']!=w['pid'] or env['ZEN_RUN']!='jobs/'+a.run or env['ZEN_WORKER']!=w['worker'] or c['Config']['Image']!=w['image']:
            raise ValueError('live allocation changed; nothing launched')
        scratch.add(next(m['Source'] for m in c['Mounts'] if m['Destination']=='/scratch'))
    if len(scratch)!=1:
        raise ValueError('one shared scratch mount required')
    name='zenfleet-retain-'+a.run.lower()+'-'+a.host
    source='/scratch/fit-cell/'+a.data_sha+'/rev4-featpot/byc/cells/'+a.arm
    out='/scratch/retained-staging-'+a.run+'-'+a.host
    cmd=['docker','run','--rm','-i','--name',name,'--pid','host','--network','none','--memory','128m','--memory-swap','128m','--cpus','0.1','--cpu-shares','256','--cpuset-cpus',rows[0]['cpuset'],'-e','TMPDIR=/scratch/tmp','-v',next(iter(scratch))+':/scratch','--entrypoint','nice',rows[0]['image'],'-n','19','ionice','-c','3','python3','-u','-','--source',source,'--out',out]
    for c in containers:
        cmd+=['--until-pid',str(c['State']['Pid'])]
    report=dict(observed_at=datetime.now(timezone.utc).isoformat(),host=a.host,run=a.run,container=name,source=source,retained_directory=out,host_scratch=next(iter(scratch)),original_allocations=[c['Id'] for c in containers],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),memory_max_bytes=134217728,cpu_quota=0.1)
    with Path(a.receipt).open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report),flush=True)
    subprocess.run(['ssh',a.host,shlex.join(cmd)],input=Path(__file__).read_text(),text=True,check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--until-pid', type=int, action='append', required=True)
    p.add_argument('--seconds', type=int, default=7200)
    args = p.parse_args()
    args.out.mkdir(exist_ok=False)
    owners = {pid: identity(pid) for pid in args.until_pid}
    if not all(owners.values()):
        raise ValueError('recorded container owner already exited')
    seen = set()
    start = time.monotonic()
    while True:
        added = retain(args.source, args.out/'files', seen)
        if added:
            print(json.dumps(dict(observed_at=datetime.now(timezone.utc).isoformat(),new_files=added,total=len(seen))), flush=True)
        if all(identity(pid) != token for pid, token in owners.items()):
            break
        if time.monotonic()-start > args.seconds:
            raise TimeoutError('staging retained; owner still active at deadline')
        time.sleep(0.1)
    files = {}
    for path in (args.out/'files').rglob('*'):
        if path.is_file():
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 << 20), b''):
                    h.update(block)
            files[str(path.relative_to(args.out))] = h.hexdigest()
    report = dict(observed_at=datetime.now(timezone.utc).isoformat(),source=str(args.source),
                  container_pids=owners,all_owners_exited=True,files_sha256=files)
    (args.out/'VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(retained_files=len(files),receipt=str(args.out/'VERIFIED.json'))), flush=True)


if __name__ == '__main__':
    launch() if sys.argv[1:2]==['launch'] else main()
