#!/usr/bin/env python3
"""Hash every unique pair-list image using one stream and 1 MiB reads.

Preserve existing inventories. Verification refuses missing, extra or changed
inputs; errno-12 reads are retried from the beginning, never from a partial hash.
Caller supplies the shared heavy locks and resource wrapper.
"""
import argparse
import csv
from datetime import datetime, timezone
import errno
import json
from pathlib import Path
import time

from canonical_run import sha


def stable_hash(path):
    for attempt in range(3):
        try:
            before = path.stat()
            digest = sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('input changed while hashing: '+str(path))
            return digest
        except OSError as error:
            if error.errno != errno.ENOMEM or attempt == 2:
                raise
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pairs', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.receipt.exists() or (args.inventory.exists() and not args.verify):
        raise ValueError('retain existing inventory/receipt; supply a new path or --verify')
    started = datetime.now(timezone.utc).isoformat()
    with args.pairs.open() as source:
        pairs = list(csv.DictReader(source, delimiter='\t'))
    if len({row['distorted'] for row in pairs}) != len(pairs):
        raise ValueError('duplicate distorted-image identity')
    paths = sorted({row[key] for row in pairs for key in ('ref_path', 'dist_path')})
    expected = json.loads(args.inventory.read_text()) if args.verify else None
    if expected is not None and set(expected) != set(paths):
        raise ValueError('input inventory membership changed')
    observed = {}
    for index, path in enumerate(paths):
        digest = stable_hash(Path(path))
        if expected is not None and expected[path] != digest:
            raise ValueError('input bytes changed: '+path)
        observed[path] = digest
        if (index+1) % 1000 == 0:
            print(f'hashed {index+1}/{len(paths)} inputs', flush=True)
    if not args.verify:
        with args.inventory.open('x') as target:
            target.write(json.dumps(observed, indent=2)+'\n')
    receipt = dict(started=started, finished=datetime.now(timezone.utc).isoformat(),
        action='verify' if args.verify else 'create', pairs=len(pairs), unique_inputs=len(paths),
        pairs_sha256=sha(args.pairs), inventory_sha256=sha(args.inventory),
        worker_sha256=sha(__file__), read_chunk_bytes=1024*1024, concurrent_read_streams=1)
    with args.receipt.open('x') as target:
        target.write(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
