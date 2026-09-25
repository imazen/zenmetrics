#!/usr/bin/env python3
"""A fast-ssim2 TRAIN-pair executor for the existing zenfleet-worker contract."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path('/var/tmp/gmsd-chroma')
CAPABILITY = 'gmsd-chroma-fast-ssim2-main'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    if sys.argv[1:] == ['capabilities']:
        print(CAPABILITY)
        return
    assert not sys.argv[1:]
    job = json.load(sys.stdin)
    assert job['kind'] == dict(kind='metric', metric=CAPABILITY)
    row = json.loads(job['cell']['knob_tuple_json'])
    fields = ['pair_key', 'width', 'height', 'ref_rgb', 'dist_rgb']
    assert set(row) == set(fields)
    binary = ROOT/'peer-target/release/gmsd-chroma-fast-ssim2-peer'
    required = {sha(binary), sha(Path(__file__)), sha(ROOT/'colour_v2/raw_pairs.tsv')}
    for field in ['ref_rgb', 'dist_rgb']:
        path = Path(row[field])
        assert path.parent == ROOT/'colour_v2/rgb'
        assert sha(path) == path.stem
        required.add(path.stem)
    assert set(job['inputs']) == required
    # Preserve each attempt's input/output files for the footprint audit;
    # the job-system owner retains the content address and ledger row.
    directory = Path(tempfile.mkdtemp(prefix='peer-job-', dir=ROOT/'tmp'))
    pairs, scores = directory/'pair.tsv', directory/'score.tsv'
    with pairs.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerow(row)
    result = subprocess.run([str(binary), str(pairs), str(scores)], capture_output=True)
    (directory/'stdout.log').write_bytes(result.stdout)
    (directory/'stderr.log').write_bytes(result.stderr)
    assert result.returncode == 0, 'CPU pair executor failed'
    with scores.open() as f:
        values = list(csv.DictReader(f, delimiter='\t'))
    assert len(values) == 1 and values[0]['pair_key'] == row['pair_key']
    output = dict(pair_key=row['pair_key'], fast_ssim2=values[0]['fast_ssim2'])
    with (ROOT/'peer_progress.jsonl').open('a') as f:
        f.write(json.dumps(dict(pair_key=row['pair_key'], status='done'))+'\n')
    print(json.dumps(output, sort_keys=True))


if __name__ == '__main__':
    main()
