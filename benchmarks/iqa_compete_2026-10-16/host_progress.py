"""Freeze a complete-row prefix and publish compact remote-host coverage counts."""
import argparse
import collections
import datetime
import hashlib
import itertools
import json
from pathlib import Path
from run import coverage_samples

SIZES = ['cid22', 'aic-small', 'aic-medium', 'aic-large', '2k', '4k', '8k']
OLD_MAP_OMISSIONS = {'fsim-y', 'haarpsi-y', 'iwssim-piq'}


def summarize(rows, inventory, threads, gpu):
    expected = {(m['metric_name'], size, count, mode)
                for m in inventory if gpu or not m['gpu']
                for size, count, mode in itertools.product(SIZES, threads, ['score', 'score+map'])}
    seen = {(r['metric_name'], r['size_name'], r['threads'], r['mode']) for r in rows}
    if len(seen) != len(rows) or not seen <= expected:
        raise ValueError('Duplicate or unexpected cells; collect attempts first')
    sizes = {}
    for size in SIZES:
        group = [r for r in rows if r['size_name'] == size]
        timed = [r for r in group if r.get('n', 0) >= coverage_samples(size)
                 and r['status'] in ['ok', 'variant_mismatch', 'partial']]
        unsupported = [r for r in group if r['status'] == 'unsupported']
        omissions = [r for r in unsupported if r['metric_name'] in OLD_MAP_OMISSIONS
                     and r['mode'] == 'score+map']
        sizes[size] = dict(present=len(group), expected=sum(k[1] == size for k in expected),
                           minimum_n=coverage_samples(size), timed_minimum_met=len(timed),
                           timed_source_contaminated=sum(bool(r.get('contaminated')) for r in timed),
                           source_unsupported=len(unsupported), adapter_omissions=len(omissions),
                           refused_or_short=len(group)-len(timed)-len(unsupported),
                           actual_n_counts=dict(collections.Counter(r.get('n', 0) for r in group)))
    return dict(present=len(rows), expected=len(expected), missing=len(expected-seen), per_size=sizes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['raw', 'inventory', 'archive', 'output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('--threads', required=True)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    original = args.raw.read_bytes()
    prefix = original[:original.rfind(b'\n')+1]
    if not prefix:
        raise ValueError('No complete rows')
    rows = [json.loads(line) for line in prefix.splitlines()]
    if len({(r['host'], r['os']) for r in rows}) != 1:
        raise ValueError('Freeze one host/OS cohort at a time')
    digest = hashlib.sha256(prefix).hexdigest()
    args.archive.mkdir(parents=True, exist_ok=True)
    archived = args.archive / (rows[0]['host']+'-'+digest[:16]+'.jsonl')
    if archived.exists():
        if archived.read_bytes() != prefix:
            raise ValueError('Existing archive differs')
    else:
        with archived.open('xb') as output:
            output.write(prefix)
    result = summarize(rows, json.loads(args.inventory.read_text()),
                       list(map(int, args.threads.split(','))), args.gpu)
    result.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  host=rows[0]['host'], cpu=rows[0]['cpu_model'], os=rows[0]['os'],
                  raw_archive=str(archived), raw_sha256=digest, raw_bytes=len(prefix),
                  incomplete_trailing_bytes=len(original)-len(prefix),
                  source_commits=sorted({r['coverage_policy']['source_commit'] for r in rows}),
                  binary_row_counts=dict(collections.Counter(r['impl_binary_sha'] for r in rows)),
                  statuses=dict(collections.Counter(r['status'] for r in rows)),
                  sample_policy='Five at CID22/AIC-small/AIC-medium; three at AIC-large/2K/4K; one at 8K. Actual n retained.',
                  limits='Partial historical cohort. Source contamination labels are preserved; final assessment also considers resource-interruption receipts. Unsupported rows with known adapter omissions do not establish map unavailability. No new metric or parity evaluations.')
    payload = (json.dumps(result, indent=2)+'\n').encode()
    if len(payload) > 30000:
        raise ValueError('Compact receipt exceeds generated-data limit')
    with args.output.open('xb') as output:
        output.write(payload)
    print(json.dumps(dict(host=result['host'], present=result['present'], expected=result['expected'], raw_sha256=digest)))


if __name__ == '__main__':
    main()
