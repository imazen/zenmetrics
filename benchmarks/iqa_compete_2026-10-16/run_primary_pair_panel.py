#!/usr/bin/env python3
"""Collect separately randomized zenbench cases; retain exploratory intervals.

The fixed number of blocks is frozen in the plan. No sample pooling, trimming,
or stopping when an interval becomes favorable. A per-case scope releases both
host locks before the next case. Iterate-mode intervals do not assert formal
zenbench confirmation or complete runtime closure.
"""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['zenbench', 'plan', 'output', 'regressions']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.zenbench = a.zenbench.resolve(strict=True)
    a.plan = a.plan.resolve(strict=True)
    plan = json.loads(a.plan.read_text())
    assert plan['blocks'] >= 20 and plan['aa']
    from scipy import __version__ as scipy_version
    from scipy.stats import t
    a.output.mkdir()
    root = a.output.resolve()
    identity = dict(plan_sha256=sha(a.plan), zenbench_sha256=sha(a.zenbench),
                    worker_sha256=sha(Path(__file__)), scipy_version=scipy_version,
                    blocks=plan['blocks'], cases=len(plan['cases']), mode='iterate',
                    interval='pointwise Student-t on whole-block log ratios; timing quantization/systematic error unbounded',
                    qualified_timing_cells=0)
    identity['per_case_blocks'] = {case['id']: int(case['fields'].get('paired_blocks', plan['blocks'])) for case in plan['cases']}
    (root / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    outcomes = []
    for case in plan['cases']:
        assert sha(a.plan) == identity['plan_sha256'] and sha(a.zenbench) == identity['zenbench_sha256']
        directory = root / case['id']
        directory.mkdir()
        case_plan = dict(plan, cases=[case], blocks=identity['per_case_blocks'][case['id']])
        case_plan_path = directory / 'plan.json'
        case_plan_path.write_text(json.dumps(case_plan, indent=2) + '\n')
        command = ['flock', str(Path.home() / 'tmp/lilith-heavy.lock'), 'flock', str(Path.home() / 'tmp/lilith-wsl-heavy.lock'),
                   str(Path.home() / 'work/claudehints/scripts/run-heavy'), '--mem', '12G', '--jobs', case['fields']['threads'],
                   '--', str(a.zenbench), 'pair', str(case_plan_path), '--output', str(directory / 'attempt')]
        with (directory / 'wrapper.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        packet_path = directory / 'attempt/packet.json.gz'
        outcome = dict(case=case['id'], command=command, returncode=result.returncode, status='process-failure')
        if packet_path.exists():
            packet = json.loads(gzip.decompress(packet_path.read_bytes()))
            outcome['packet_sha256'] = sha(packet_path)
            summaries = {s['arm']: s for s in packet['summaries']}
            assert packet['mode'] == 'iterate'
            if packet['complete'] and result.returncode == 0 and summaries['rust']['complete_blocks'] == case_plan['blocks']:
                fields = case['fields']
                records = []
                for arm in ['rust', '__aa']:
                    summary = summaries[arm]
                    n = summary['complete_blocks']
                    critical = float(t.ppf(0.975, n - 1))
                    half = critical * summary['log_ratio_sd'] / math.sqrt(n)
                    estimate = math.exp(summary['mean_log_ratio'])
                    records.append(dict(arm=arm, estimate=estimate,
                                        ci95_low=math.exp(summary['mean_log_ratio'] - half),
                                        ci95_high=math.exp(summary['mean_log_ratio'] + half)))
                comparison, control = records
                scores, memories = {'rust': [], 'canonical': []}, {'rust': [], 'canonical': []}
                for invocation in packet['invocations']:
                    record = json.loads((Path(invocation['output']) / 'arm.json').read_text())
                    scores[record['arm']].extend(record['scores'])
                    memories[record['arm']].append(record['process_rss_kbytes'])
                delta = max(abs(x - y) for x in scores['rust'] for y in scores['canonical'])
                assert delta <= 5e-5
                receipts = [s for s in (directory / 'wrapper.log').read_text().splitlines() if s.startswith('run-heavy: done rc=')]
                assert len(receipts) == 1
                interval_width = max(comparison['estimate'] / comparison['ci95_low'] - 1, comparison['ci95_high'] / comparison['estimate'] - 1)
                row = dict(metric=fields['metric'], canonical='author HaarPSI.m 3b9599b0 / Octave 6.4',
                           case=case['id'], size=fields['width'] + 'x' + fields['height'], threads=int(fields['threads']),
                           phase=fields['phase'], max_abs_delta=delta, bound=5e-5, passed=True,
                           ours_ms=summaries['rust']['arm_ns']['mean']/1e6,
                           canonical_ms=summaries['rust']['baseline_ns']['mean']/1e6,
                           ratio=comparison['estimate'], ci95_low=comparison['ci95_low'], ci95_high=comparison['ci95_high'],
                           precision_within_3pct=interval_width < 0.03,
                           flag='borderline' if comparison['ci95_low'] <= 1.08 <= comparison['ci95_high'] else
                                'above-1.08' if comparison['estimate'] > 1.08 else 'below-1.08',
                           aa_ratio=control['estimate'], aa_ci95_low=control['ci95_low'], aa_ci95_high=control['ci95_high'],
                           ours_max_process_rss_kbytes=max(memories['rust']), canonical_max_process_rss_kbytes=max(memories['canonical']),
                           qualification='exploratory', qualified_ratio=False)
                outcome.update(status='exploratory-complete', row=row, resource_receipt=receipts[0])
                if row['ratio'] > 1.08:
                    a.regressions.parent.mkdir(parents=True, exist_ok=True)
                    with a.regressions.open('a') as f:
                        f.write(f'{datetime.now(timezone.utc).isoformat()} | {row["metric"]} | {row["size"]} | {row["threads"]} | ours {row["ours_ms"]:.17g} ms | canonical {row["canonical_ms"]:.17g} ms | {row["ratio"]:.17g} | measured-here exploratory {fields["phase"]}; CI95 [{row["ci95_low"]:.17g},{row["ci95_high"]:.17g}]; {packet_path}\n')
        (directory / 'outcome.json').write_text(json.dumps(outcome, indent=2) + '\n')
        outcomes.append(outcome)
        rows = [o['row'] for o in outcomes if 'row' in o]
        if rows:
            with (root / 'summary.tsv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
                writer.writeheader(); writer.writerows(rows)
        print('CASE_DONE', len(outcomes), '/', len(plan['cases']), case['id'], outcome['status'],
              json.dumps(outcome.get('row', {})), flush=True)
    (root / 'completion.json').write_text(json.dumps(dict(cases=outcomes, qualified_timing_cells=0), indent=2) + '\n')
    if any(o['status'] != 'exploratory-complete' for o in outcomes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
