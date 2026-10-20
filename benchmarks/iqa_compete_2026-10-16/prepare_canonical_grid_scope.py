#!/usr/bin/env python3
"""Expand a numerical grid while preserving reviewed timing-scope caveats.

The original and reviewed plans must have identical numerical/callable contracts.
Completed plans are published atomically; failed preparation remains in its
staging directory and never becomes an advertised executable plan.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile

from compare_canonical import sha
from prepare_canonical_grid import prepare


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reviewed_contract(original, reviewed):
    require((original['binary'], original['rust_source']) ==
            (reviewed['binary'], reviewed['rust_source']), 'scope review worker/source differs')
    left = {row['id']: row for row in original['cases']}
    right = {row['id']: row for row in reviewed['cases']}
    require(len(left) == len(original['cases']) and len(right) == len(reviewed['cases'])
            and left.keys() == right.keys(), 'complete unique original case inventory required')
    caveats = reviewed['scope_dispositions']
    names = {row['metric'] for row in left.values()}
    require(isinstance(caveats, dict) and all(
        name in names and isinstance(reason, str) and reason.strip()
        for name, reason in caveats.items()), 'invalid scope disposition')
    ignored = {'scope', 'matched_scope', 'scope_disposition'}
    for key, before in left.items():
        after = right[key]
        require({k: v for k, v in before.items() if k not in ignored} ==
                {k: v for k, v in after.items() if k not in ignored},
                'numerical/input/callable contract changed: '+key)
        reason = caveats.get(before['metric'])
        if reason:
            require(after['matched_scope'] is False and after.get('scope_disposition') == reason,
                    'caveat and reviewed case disagree: '+key)
        else:
            require(after['matched_scope'] is before['matched_scope'] and
                    after.get('scope_disposition') == before.get('scope_disposition'),
                    'reviewed case lost its scope disposition: '+key)
    return caveats


def bindings_for(original, reviewed):
    bindings = dict(original['bindings'])
    for path, expected in reviewed['bindings'].items():
        require(path not in bindings or bindings[path] == expected,
                'incompatible source binding: '+path)
        bindings[path] = expected
    return bindings


def verify(bindings):
    for path, expected in bindings.items():
        require(sha(path) == expected, 'bound artifact changed: '+path)


def qualify(plan, caveats, qualification):
    plan = copy.deepcopy(plan)
    for row in plan['cases']:
        reason = caveats.get(row['metric'])
        if reason:
            row['matched_scope'] = False
            row['scope_disposition'] = reason
            row['scope'] += '; timing scope caveat: '+reason
    plan['scope_dispositions'] = {name: reason for name, reason in caveats.items()
                                  if any(row['metric'] == name for row in plan['cases'])}
    plan['scope_qualification'] = qualification
    return plan


def prepare_scoped_grid(source, pairs, scoped, output, metrics, threads):
    source, pairs, scoped = (path.resolve(strict=True) for path in (source, pairs, scoped))
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    original = json.loads(source.read_text())
    reviewed = json.loads(scoped.read_text())
    caveats = reviewed_contract(original, reviewed)
    bindings = bindings_for(original, reviewed)
    for path in (source, pairs, scoped, Path(__file__).resolve()):
        name, actual = str(path), sha(path)
        require(name not in bindings or bindings[name] == actual,
                'preparation input changed: '+name)
        bindings[name] = actual
    verify(bindings)
    stage = Path(tempfile.mkdtemp(prefix=output.name+'.prepare-', dir=output.parent))
    unqualified = stage/'source-grid.plan.json'
    count = prepare(source, pairs, unqualified, metrics, threads)
    plan = qualify(json.loads(unqualified.read_text()), caveats, reviewed['scope_qualification'])
    plan['bindings'].update(bindings)
    verify(plan['bindings'])
    candidate = stage/'qualified-grid.plan.json'
    with candidate.open('x') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write('\n')
    # Same-filesystem hard link publishes a complete file without replacing an
    # existing path. Keep the staging evidence on both success and failure.
    os.link(candidate, output)
    return dict(cases=count, scope_caveated_cases=sum(not row['matched_scope'] for row in plan['cases']),
                plan=str(output), preparation_directory=str(stage))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'pairs', 'scoped', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--threads', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_scoped_grid(args.source, args.pairs, args.scoped, args.output,
                                        args.metrics.split(','), list(map(int, args.threads.split(','))))))


if __name__ == '__main__':
    main()
