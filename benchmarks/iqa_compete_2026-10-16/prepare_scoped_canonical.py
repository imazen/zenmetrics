#!/usr/bin/env python3
"""Preserve explicit timing-scope caveats in newly frozen comparisons.

The original preparer stays unchanged for active plans. A caveat withholds a
comparable timing ratio; it never changes numerical bounds or acquired evidence.
"""
import argparse
import json
from pathlib import Path

from compare_canonical import sha
from prepare_ordered_canonical import prepare


def scoped_prepare(manifest, output, threads):
    data = json.loads(manifest.read_text())
    caveats = data.get('scope_dispositions', {})
    if not isinstance(caveats, dict) or any(
        name not in data['metric_plans'] or not isinstance(reason, str) or not reason.strip()
        for name, reason in caveats.items()
    ):
        raise ValueError('scope dispositions require selected route names and nonempty explanations')
    metrics, cases = prepare(manifest, output, threads)
    plan = json.loads(output.read_text())
    plan['bindings'][str(Path(__file__).resolve())] = sha(Path(__file__))
    for case in plan['cases']:
        reason = caveats.get(case['metric'])
        if reason:
            case['matched_scope'] = False
            case['scope_disposition'] = reason
            case['scope'] += '; timing scope caveat: '+reason
    plan['scope_dispositions'] = caveats
    plan['scope_qualification'] = (
        'Caveats are coordinator source-review findings, not human-approved exceptions. '
        'Caveated cases retain numerical observations and withhold comparable timing ratios.'
    )
    # Only this invocation's freshly created plan is updated; existing paths
    # were already refused by the original preparer.
    output.write_text(json.dumps(plan, indent=2)+'\n')
    return dict(metrics=metrics, cases=cases, scope_caveated_cases=sum(
        case['metric'] in caveats for case in plan['cases']), plan=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--threads', default='1,8')
    args = parser.parse_args()
    print(json.dumps(scoped_prepare(args.manifest, args.output,
                                  list(map(int, args.threads.split(','))))))


if __name__ == '__main__':
    main()
