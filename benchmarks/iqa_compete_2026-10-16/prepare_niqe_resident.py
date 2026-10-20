#!/usr/bin/env python3
"""Prepare NIQE reruns and explicitly amend a caller-identified unexecuted plan."""
import argparse
import copy
import json
from pathlib import Path

from compare_canonical import sha


def corrected(plan, helper, control):
    value = copy.deepcopy(plan)
    for asset, expected in plan['bindings'].items():
        if sha(asset) != expected:
            raise ValueError('Frozen artifact changed: ' + asset)
    value['bindings'][str(helper)] = sha(helper)
    value['bindings'][str(control)] = sha(control)
    value['bindings'][str(Path(__file__).resolve())] = sha(__file__)
    found = 0
    for case in value['cases']:
        if case['metric'] != 'niqe':
            continue
        command = case['canonical']['command']
        if Path(command[1]).name != 'canonical_cpu_cell.py':
            raise ValueError('Expected the original NIQE fixture')
        command[1] = str(helper)
        index = command.index('--weights')
        del command[index:index+2]
        archive = Path(command[command.index('--extra-path')+1])/'basicsr/metrics/niqe_pris_params.npz'
        archive = archive.resolve(strict=True)
        value['bindings'][str(archive)] = sha(archive)
        case['canonical']['label'] = 'BasicSR 1.4.2 / resident parameters'
        case['scope'] = ('Resident RGB8 conversion and unchanged BasicSR pair call; '
                         'immutable archive I/O excluded, owned parameter-array copies retained')
        found += 1
    if not found:
        raise ValueError('No NIQE cases in source plan')
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('control_log', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('receipt', type=Path)
    p.add_argument('--future-plan', type=Path, required=True)
    p.add_argument('--future-run', type=Path, required=True)
    a = p.parse_args()
    helper = Path(__file__).with_name('canonical_niqe_resident_cell.py').resolve(strict=True)
    control = a.control_log.resolve(strict=True)
    text = control.read_text()
    rows = [json.loads(line[10:]) for line in text.splitlines() if line.startswith('CANONICAL ')]
    if len(rows) != 1 or rows[0]['original_score_bits'] != rows[0]['resident_score_bits'] or 'done rc=0' not in text:
        raise ValueError('A completed successful original/resident control is required')
    if a.future_run.exists():
        raise ValueError('Future plan already acquired an output directory')
    backup = a.future_plan.with_name(a.future_plan.name+'.original-niqe-loader.bak')
    temporary = a.future_plan.with_name(a.future_plan.name+'.resident-new')
    if any(path.exists() for path in (backup, temporary, a.output, a.receipt)):
        raise ValueError('Prepared outputs or backups already exist')
    source = json.loads(a.source.read_text())
    original_future = json.loads(a.future_plan.read_text())
    future = corrected(original_future, helper, control)
    for old, new in zip(original_future['cases'], future['cases'], strict=True):
        if {k:v for k,v in old.items() if k not in ('canonical','scope')} != {k:v for k,v in new.items() if k not in ('canonical','scope')}:
            raise ValueError('A pre-existing case contract changed')
    replay = corrected(source, helper, control)
    template = next(c for c in replay['cases'] if c['metric'] == 'niqe')
    replay['cases'] = []
    for threads in (1, 8):
        case = copy.deepcopy(template)
        case.update(threads=threads, id=f'niqe-resident-pair-cid22-t{threads}')
        replay['cases'].append(case)
    replay['bindings'][str(a.source.resolve())] = sha(a.source)
    record = dict(helper_sha256=sha(helper), preparer_sha256=sha(__file__), control=rows[0],
                  control_log_sha256=sha(control), original_future_plan_sha256=sha(a.future_plan),
                  preserved_future_plan=str(backup), future_run=str(a.future_run),
                  numerical_contract='All numerical case fields, bounds and native sampling settings preserved')
    with a.output.open('x') as stream:
        json.dump(replay, stream, indent=2)
    with temporary.open('x') as stream:
        json.dump(future, stream, indent=2)
    if a.future_run.exists():
        raise ValueError('Future plan started during preparation; original remains unchanged')
    with backup.open('xb') as stream:
        stream.write(a.future_plan.read_bytes())
    temporary.replace(a.future_plan)
    record.update(future_plan_sha256=sha(a.future_plan), replay_plan_sha256=sha(a.output))
    with a.receipt.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
