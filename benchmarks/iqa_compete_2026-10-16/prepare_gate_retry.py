#!/usr/bin/env python3
"""Freeze only flagged warm cells with remaining original attempts; retain full lineage."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import durable
from mac_service import verify
from prepare_coverage import freeze


def selection(report, cap):
    if not report.get('complete'):
        raise ValueError('original campaign must be terminal before retry selection')
    pending, exhausted = defaultdict(list), []
    for index, cell in enumerate(report['cells']):
        if cell['status'] != 'ok' or cell['config']['timing_phase'] != 'warm':
            continue
        row = cell.get('row') or {}
        rounds = row.get('zenbench_samples', [])
        clean = row.get('n') == 10 and len(rounds) == 10 and all(s.get('gate_clean') is True for s in rounds)
        if clean:
            continue
        remaining = cap - len(cell['attempts'])
        if remaining < 0:
            raise ValueError('attempt history exceeds original cap')
        if remaining:
            pending[remaining].append(index)
        else:
            exhausted.append(cell['cell_id'])
    return pending, exhausted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('results', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--lock', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('new successor directory required; old artifacts are immutable')
    with durable.locked(a.lock, wait=True), durable.locked(a.results / 'coordinator.lock'):
        verify(a.bundle)
        _, configs, frozen = durable.manifest(a.bundle / 'manifest.json')
        saved = durable.load(a.results / 'identity.json')
        report = durable.load(a.results / 'report.json')
        identity = durable.digest(durable.encoded(saved))
        if saved['frozen'] != frozen or report['identity'] != identity:
            raise ValueError('parent report does not identify the frozen bundle')
        if [c['cell_id'] for c in report['cells']] != [c['durable_cell_id'] for c in configs]:
            raise ValueError('parent report cell order differs')
        if not report.get('complete'):
            raise ValueError('original campaign must be terminal before retry selection')
        # Validate every retained attempt seal/config/log and its projected report count.
        for config, cell in zip(configs, report['cells']):
            _, records = durable.attempts(a.results, config, identity)
            final = next((r for r in records if r['status'] in ('ok', 'unsupported')), None)
            if cell['config'] != config or cell['attempts'] != records or cell.get('row') != (final['row'] if final else None):
                raise ValueError('parent report projection differs from sealed attempt history')
        groups, exhausted = selection(report, saved['policy']['max_attempts'])
        original = durable.load(a.bundle / 'manifest.json')['cells']
        a.output.mkdir(parents=True, exist_ok=False)
        successors = []
        for remaining, indices in sorted(groups.items()):
            successor = a.output / ('remaining-attempts-' + str(remaining))
            receipt = freeze(a.bundle, successor, [original[i] for i in indices], a.results / 'report.json')
            receipt.update(max_attempts=remaining, parent_cell_ids=[report['cells'][i]['cell_id'] for i in indices],
                           lineage_rule='use qualified original rows except these replacement cells; no duplicate counting')
            successors.append(receipt)
        value = dict(parent_identity=identity, parent_results=str(a.results.resolve()),
                     parent_attempt_cap=saved['policy']['max_attempts'], successors=successors,
                     exhausted_flagged_cell_ids=exhausted, protocol_or_gate_changes=False,
                     installation='not installed; execute each frozen successor with exactly its max_attempts')
        (a.output / 'retry-plan.json').write_text(json.dumps(value, indent=2) + '\n')
        print(json.dumps(value))


if __name__ == '__main__':
    main()
