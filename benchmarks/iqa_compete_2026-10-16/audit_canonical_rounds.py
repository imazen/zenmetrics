#!/usr/bin/env python3
"""Audit recorded native warm counts; this does not qualify metric results."""
import argparse
import json
from pathlib import Path

from compare_canonical import sha
from reference_publication import register_outputs


def inspect(cohorts):
    arms = []
    missing = blocks = 0
    for root in cohorts:
        plan_digest = sha(root / 'plan.json')
        for path in sorted(root.glob('*/block-*/result.json')):
            blocks += 1
            native = json.loads(path.read_text()).get('arms', {}).get('rust')
            if native is None:
                missing += 1
                continue
            counts = [native['extra'].get('n'), len(native['times_ns']),
                      len(native['extra'].get('round_means_ns', []))]
            arms.append(dict(path=str(path.resolve()), sha256=sha(path),
                             plan_sha256=plan_digest, counts=counts))
    return dict(observed_blocks=blocks, successful_native_arms=len(arms),
                missing_native_arms=missing, required_rounds=10,
                findings=sum(row['counts'] != [10, 10, 10] for row in arms)), arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('raw_manifest', type=Path)
    parser.add_argument('cohorts', nargs='+', type=Path)
    args = parser.parse_args()
    if len({path.resolve() for path in args.cohorts}) != len(args.cohorts):
        parser.error('duplicate cohort')
    summary, arms = inspect(args.cohorts)
    with args.raw_manifest.open('x') as stream:
        json.dump(dict(**summary, native_arms=arms), stream, indent=2)
        stream.write('\n')
    raw = args.raw_manifest.resolve()
    pointer = '~/' + str(raw.relative_to(Path.home())) if raw.is_relative_to(Path.home()) else str(raw)
    summary.update(raw_manifest=pointer, raw_manifest_sha256=sha(raw),
                   auditor_sha256=sha(__file__),
                   qualification='retained successful arms only; missing arms are reported separately; no parity, matrix-completeness or performance acceptance')
    text = json.dumps(summary, indent=2) + '\n'
    if len(text.encode()) > 30_000:
        raise ValueError('audit summary exceeds generated-data cap')
    args.output.write_text(text)
    register_outputs([(args.output, 'provenance')])
    print(json.dumps(summary))
    if summary['findings'] or not summary['successful_native_arms']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
