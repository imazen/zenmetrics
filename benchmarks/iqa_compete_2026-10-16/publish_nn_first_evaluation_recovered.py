#!/usr/bin/env python3
"""Project resident-model first calls with explicit recovered-block provenance.

The original publisher stays frozen. Numerical observation validation is shared
with it; no score, threshold or timing protocol changes.
"""
import argparse
import json
from pathlib import Path
import statistics
from compare_canonical import sha
from publish_nn_first_evaluation import observation, __file__ as ORIGINAL_PUBLISHER_SOURCE
from reference_publication import register_outputs
from resume_canonical_campaign import _completed, __file__ as RECOVERY_SOURCE


def collect(root, receipt):
    plan_path = root / 'plan.json'
    plan = json.loads(plan_path.read_text())
    digest = sha(plan_path)
    rows = []
    for case in plan['cases']:
        path = root / case['id'] / 'summary.json'
        if not path.exists():
            continue
        binary = case['native']['command'][0]
        if (plan['bindings'].get(binary) != receipt['binary_sha256']
                or sha(binary) != receipt['binary_sha256']):
            raise ValueError('first-evaluation probe differs from its build receipt')
        row = json.loads(path.read_text())
        if row['case'] != case['id'] or len(row['blocks']) != 3:
            raise ValueError('three identified first-evaluation blocks required')
        for field in ('metric', 'width', 'height', 'threads'):
            if row[field] != case[field]:
                raise ValueError('first-evaluation summary contract differs')
        if row['plan_sha256'] != digest or row['phase'] != plan['phase']:
            raise ValueError('first-evaluation plan identity differs')
        blocks = []
        hashes = []
        resources = []
        raw_paths=[]
        for number in range(3):
            block_path = _completed(path.parent,number)
            if block_path is None:
                raise ValueError('completed first-evaluation result missing')
            block = json.loads(block_path.read_text())
            if block['phase'] != plan['phase'] or block != row['blocks'][number]:
                raise ValueError('first-evaluation block or phase differs')
            blocks.append(observation(block, case))
            hashes.append(sha(block_path))
            raw_paths.append(block_path)
            log = Path(str(block_path.parent)+'.log')
            resources.extend(s for s in log.read_text().splitlines()
                             if s.startswith('run-heavy: done '))
        passed = (False if any(b['passed'] is False for b in blocks)
                  else None if any(b['passed'] is None for b in blocks) else True)
        # The original acquisition summary retains a descriptive ratio even
        # for a numerical failure. Keep it in raw evidence, withhold it here.
        ratio = (statistics.median(b['ratio'] for b in blocks)
                 if case['matched_scope'] and all(b['ratio'] is not None for b in blocks)
                 else None)
        if row['passed'] is not (passed is True):
            raise ValueError('stored first-evaluation summary decision differs')
        retained_ratio = (statistics.median(b['ratio'] for b in row['blocks'])
                          if all('ratio' in b for b in row['blocks']) else None)
        if row['ratio'] != retained_ratio:
            raise ValueError('stored first-evaluation summary ratio differs')
        rows.append(dict(metric=case['metric'], case=case['id'],
                         width=case['width'], height=case['height'], threads=case['threads'],
                         canonical=case['canonical']['label'], phase=plan['phase'],
                         native_source=receipt['source_revision'],
                         native_binary_sha256=receipt['binary_sha256'],
                         engine=case['native'].get('environment', {}).get('ZENMETRICS_NN_ENGINE'),
                         absolute_bound=case['absolute_bound'], relative_bound=case.get('relative_bound', 0),
                         passed=passed, max_abs_delta=max((b['delta'] for b in blocks if 'delta' in b), default=None),
                         ratio=ratio, blocks=blocks, wrapper_receipts=resources,
                         raw_directory=str(root), plan_sha256=digest,
                         summary_sha256=sha(path), block_sha256=hashes))
        if any(raw.parent.name!=f'block-{number:02d}' for number,raw in enumerate(raw_paths)):
            completed_logs={Path(str(raw.parent)+'.log') for raw in raw_paths}
            rows[-1]['recovery_evidence']=dict(completed_blocks=[dict(number=number,path=str(raw),sha256=sha(raw)) for number,raw in enumerate(raw_paths)],interrupted_logs=[dict(path=str(log),sha256=sha(log)) for log in sorted(set(path.parent.glob('block-*.log'))-completed_logs)],qualification='Original completed blocks and interrupted output retained; recovered result paths supply their own actual wrapper receipts.')
    return rows, len(plan['cases'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('build_receipt', type=Path)
    parser.add_argument('roots', type=Path, nargs='+')
    args = parser.parse_args()
    receipt = json.loads(args.build_receipt.read_text())
    args.output.mkdir(exist_ok=True)
    rows = []
    cohorts = []
    for root in args.roots:
        acquired, expected = collect(root, receipt)
        rows.extend(acquired)
        cohorts.append(dict(raw_directory=str(root), observed=len(acquired), expected=expected,
                            plan_sha256=sha(root / 'plan.json')))
    outputs = []
    shards = []
    part = []

    def emit():
        path = args.output / f'cases-{len(shards):02d}.json'
        path.write_text(json.dumps(dict(cases=part), indent=2) + '\n')
        if path.stat().st_size > 30_000:
            raise ValueError('single first-evaluation record exceeds artifact cap')
        outputs.append((path, 'summary'))
        shards.append(dict(file=path.name, cases=len(part), sha256=sha(path)))

    for row in rows:
        if part and len(json.dumps(dict(cases=part + [row]), indent=2).encode()) > 24_000:
            emit()
            part = []
        part.append(row)
    if part:
        emit()
    index = args.output / 'index.json'
    index.write_text(json.dumps(dict(
        phase='First evaluation after resident immutable-model construction',
        qualification='Descriptive alternating first calls; no warm timing, full-startup or regression acceptance claim',
        host_cpu='Ryzen 9 7950X / WSL2', observed=len(rows), cohorts=cohorts,
        build_receipt_sha256=sha(args.build_receipt),
        publisher_sha256=sha(__file__), publisher_dependencies={Path(source).name:sha(source) for source in [ORIGINAL_PUBLISHER_SOURCE,RECOVERY_SOURCE]}, shards=shards), indent=2) + '\n')
    if index.stat().st_size > 30_000:
        raise ValueError('first-evaluation index exceeds artifact cap')
    outputs.append((index, 'summary'))
    register_outputs(outputs)
    print(json.dumps(dict(observed=len(rows), cohorts=len(cohorts), shards=len(shards))))


if __name__ == '__main__':
    main()
