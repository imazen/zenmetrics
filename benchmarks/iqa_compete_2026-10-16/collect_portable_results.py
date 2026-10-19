#!/usr/bin/env python3
"""Read-only export of portable CPU rows with bound worker resource receipts."""
import argparse
import csv
import io
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def collect(roots, output):
    records, sources = [], []
    for root in roots:
        for rows_path in sorted(root.rglob('rows.jsonl')):
            raw = rows_path.read_bytes()
            rows = [json.loads(line) for line in raw.splitlines()]
            if any(row['mode'] != 'score' for row in rows):
                raise ValueError('scalar-only export refuses map rows')
            for row in rows:
                candidates = []
                for config_path in rows_path.parent.glob(f"{row['metric_name']}-{row['size_name']}-{row['threads']}-*.json"):
                    config_raw = config_path.read_bytes()
                    config = json.loads(config_raw)
                    log_path = config_path.with_suffix('.log')
                    if not log_path.is_file():
                        continue
                    log_raw = log_path.read_bytes()
                    logged = [json.loads(line[8:]) for line in log_raw.decode().splitlines() if line.startswith('IQA_ROW ')]
                    if row not in logged:
                        continue
                    if config.get('only_mode') != 'score':
                        raise ValueError('config did not request scalar-only')
                    for key in ('metric_name', 'size_name', 'width', 'height', 'threads', 'impl_binary_sha', 'pair_sha256'):
                        if config[key] != row[key]:
                            raise ValueError(f'config/row mismatch: {key}')
                    summaries = [line for line in log_raw.decode().splitlines() if line.startswith('run-heavy: done ')]
                    if len(summaries) != 1:
                        raise ValueError('missing or ambiguous finalized resource receipt')
                    summary = summaries[0]
                    match = re.search(r'peak-RSS\s+([0-9.]+)\s*GiB', summary)
                    if not match:
                        raise ValueError('unrecognized resource receipt')
                    exact_rss = re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$', log_raw.decode(), re.MULTILINE) if row.get('os') == 'macos' else []
                    if len(exact_rss) > 1:
                        raise ValueError('ambiguous native OS peak RSS receipt')
                    candidates.append(dict(os_time_max_rss_bytes=int(exact_rss[0]) if exact_rss else None, config_path=str(config_path), config_sha256=sha(config_raw),
                        log_path=str(log_path), log_sha256=sha(log_raw), resource_summary=summary,
                        peak_rss_gib_rounded=float(match.group(1)), peak_rss_precision_gib=0.01,
                        memory_scope='Resource wrapper measurement of the worker scope, including loading/setup; not metric heap alone'))
                if len(candidates) != 1:
                    raise ValueError(f'expected one bound worker log, found {len(candidates)}: {row["metric_name"]}')
                samples = row.get('zenbench_samples', [])
                required = row['required_samples']
                qualified = (row['status'] == 'ok' and row.get('n', 0) >= required
                             and len(samples) >= required and all(s['gate_clean'] for s in samples))
                records.append(dict(observation=row, qualified_timing=qualified,
                    timing_protocol='portable-coverage-5-3-1-v1', row_sha256=sha(encode(row)),
                    evidence=candidates[0]))
            if rows_path.read_bytes() != raw:
                raise ValueError('rows changed during collection; collect beneath the shared lock')
            sources.append(dict(path=str(rows_path), sha256=sha(raw), rows=len(rows),
                complete=(rows_path.parent / 'run.meta').is_file()))
    if not records:
        raise ValueError('no observations')
    output.mkdir(parents=True, exist_ok=False)
    payload = b''.join(encode(record) + b'\n' for record in records)
    (output / 'observations.jsonl').write_bytes(payload)
    columns = ['host', 'metric_name', 'impl_version', 'impl_binary_sha', 'size_name',
               'width', 'height', 'threads', 'timing_phase', 'status', 'n', 'required_samples',
               'median_ns', 'p10_ns', 'p90_ns', 'mean_ns', 'score', 'qualified_timing',
               'peak_rss_gib_rounded', 'os_time_max_rss_bytes', 'row_sha256', 'log_sha256']
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for record in records:
        flat = {key: record['observation'].get(key) for key in columns}
        flat.update(qualified_timing=record['qualified_timing'], row_sha256=record['row_sha256'],
                    peak_rss_gib_rounded=record['evidence']['peak_rss_gib_rounded'],
                    os_time_max_rss_bytes=record['evidence']['os_time_max_rss_bytes'],
                    log_sha256=record['evidence']['log_sha256'])
        writer.writerow(flat)
    csv_payload = buffer.getvalue().encode()
    (output / 'timings.csv').write_bytes(csv_payload)
    snapshot = dict(schema=1, captured_utc=datetime.now(timezone.utc).isoformat(),
        collector_sha256=sha(Path(__file__).read_bytes()), sources=sources,
        rows=len(records), qualified_timing_rows=sum(r['qualified_timing'] for r in records),
        observations_sha256=sha(payload), timings_csv_sha256=sha(csv_payload), memory_precision_note='Peak RSS is retained at the resource wrapper reporting precision; exact bytes are not inferred.')
    (output / 'snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    print(json.dumps(snapshot))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('roots', nargs='+', type=Path)
    args = parser.parse_args()
    collect(args.roots, args.output)
