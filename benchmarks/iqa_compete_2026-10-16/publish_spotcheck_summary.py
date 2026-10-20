#!/usr/bin/env python3
"""Export published warm cells and exact headline cases for the spot-check site."""
import argparse
import csv
import io
import json
from pathlib import Path

from compare_canonical import sha
from reference_publication import read_cohort_index, register_outputs


COLUMNS = ['metric', 'canonical', 'size', 'threads', 'max_abs_delta', 'bound',
           'pass', 'ours_ms', 'canonical_ms', 'ratio', 'flag', 'status', 'observation', 'cohort',
           'rust_source', 'plan_sha256', 'phase', 'case']


def tsv(rows, columns=COLUMNS):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, columns, delimiter='\t', lineterminator='\n', extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def read_json(path):
    return json.loads(path.read_text())


def export(source, inventory, output):
    names = sorted(row['metric_name'] for row in read_json(inventory)['inventory'])
    if len(names) != len(set(names)):
        raise ValueError('unique inventory names required')
    index = source / 'results.json'
    selected = {}
    bindings = {str(index.resolve()): sha(index), str(inventory.resolve()): sha(inventory), str(Path(__file__).resolve()): sha(__file__)}
    for name in ('cohorts.json', 'unmeasured.json'):
        path = source / name
        bindings[str(path.resolve())] = sha(path)
    for shard in read_json(index)['shards']:
        path = source / shard['file']
        if sha(path) != shard['sha256']:
            raise ValueError('published headline shard changed')
        selected.update(read_json(path))
        bindings[str(path.resolve())] = sha(path)
    observations = []
    originals = []
    for cohort, record in read_cohort_index(source / 'cohorts.json').items():
        for artifact in record['artifacts']:
            path = source / artifact['file']
            if sha(path) != artifact['sha256']:
                raise ValueError('published cohort artifact changed')
            bindings[str(path.resolve())] = sha(path)
            if path.suffix != '.tsv':
                continue
            for raw in csv.DictReader(io.StringIO(path.read_text()), delimiter='\t'):
                eligible = raw['matched_scope'] == 'True' and not raw.get('excluded_contract')
                ratio = raw['ratio'] if eligible else ''
                row = dict(metric=raw['metric'], canonical=raw['canonical'],
                    size=raw['width'] + 'x' + raw['height'], threads=raw['threads'],
                    max_abs_delta=raw['max_abs_delta'], bound=raw['absolute_bound'],
                    **{'pass': raw['passed']}, ours_ms=raw['rust_ms'], canonical_ms=raw['canonical_ms'],
                    ratio=ratio, flag=str(float(ratio) > 1.08) if ratio else '',
                    status='contract excluded' if raw.get('excluded_contract') else
                        ('process finding' if int(raw.get('process_errors') or 0) else
                         'descriptive; numerical bound unassigned' if not raw['absolute_bound'] else
                         'observed; performance unqualified'),
                    cohort=cohort, rust_source=raw['rust_source'], plan_sha256=raw['plan_sha256'],
                    phase='warm full call', case=raw['case'], observation=len(observations))
                observations.append(row)
                originals.append(raw)
    statuses = read_json(source / 'unmeasured.json')
    headlines = []
    for name in names:
        value = selected.get(name, {})
        candidates = []
        if value.get('ratio') is not None:
            pool = []
            for row, raw in zip(observations, originals):
                if (row['metric'] == name and row['ratio']
                        and row['rust_source'] == value['ratio_source']
                        and row['plan_sha256'] == value['ratio_plan_sha256']
                        and int(raw['width']) == value['ratio_width']
                        and int(raw['height']) == value['ratio_height']
                        and int(row['threads']) == value['ratio_threads']):
                    pool.append(row)
                    if float(row['ratio']) == value['ratio']:
                        candidates.append(row)
            if not candidates and pool:
                candidates = [min(pool, key=lambda row: (row['cohort'], row['case']))]
            if not candidates:
                raise ValueError('headline does not identify an exact retained timing cell: ' + name)
        if candidates:
            row = dict(candidates[0], status='descriptive')
        else:
            row = dict.fromkeys(COLUMNS, '')
            row.update(metric=name, canonical=value.get('canonical', ''),
                       status=statuses.get(name, {}).get('check', value.get('check', 'not measured')),
                       phase='no selected comparable warm cell')
        headlines.append(row)
    if any(sha(path) != expected for path, expected in bindings.items()):
        raise ValueError('publication inputs changed during projection')
    output.mkdir()
    artifacts = []
    shards = []
    def save(path, text, kind):
        with path.open('x') as stream:
            stream.write(text)
        if path.stat().st_size > 30000:
            raise ValueError('generated artifact exceeds 30 KB: ' + str(path))
        artifacts.append((path, kind))
    save(output / 'summary.tsv', tsv(headlines, COLUMNS[:13]), 'cohort')
    for start in range(0, len(observations), 40):
        rows = observations[start:start + 40]
        path = output / ('cells-%03d.tsv' % (start // 40))
        save(path, tsv(rows), 'cohort')
        shards.append(dict(file=path.name, rows=len(rows), sha256=sha(path)))
    input_shards = []
    items = list(bindings.items())
    for start in range(0, len(items), 35):
        path = output / ('inputs-%02d.json' % (start // 35))
        save(path, json.dumps(dict(items[start:start + 35]), indent=2) + '\n', 'provenance')
        input_shards.append(dict(file=path.name, sha256=sha(path)))
    manifest = dict(inventory_names=len(names), headline_rows=len(headlines),
        observed_warm_cells=len(observations), shards=shards, input_binding_shards=input_shards,
        ratio_definition='Rust/canonical. Flag is ratio > 1.08; not an accepted regression.',
        qualification='Warm full-call observations only. First-evaluation and map evidence remain '
            'separate. All source cohorts, including exclusions and historical sources, are retained. '
            'summary.tsv prefers an exact retained README timing cell. When the README reports an aggregate '
            'without an exact median cell, it selects a deterministic actual case from that same geometry/source/cohort. '
            'Its times, ratio, delta and bound always describe that individual observed cell, not an invented '
            'decomposition of the aggregate or a pooled all-size parity claim. observation identifies the '
            'zero-based row in the concatenated cells shards.',
        exclusions=dict(aimos='excluded: proprietary, unreleased; outside the 126-name CLI inventory'))
    save(output / 'index.json', json.dumps(manifest, indent=2) + '\n', 'summary')
    register_outputs(artifacts)
    return dict(names=len(names), observed_cells=len(observations), shards=len(shards))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.inventory, args.output)))


if __name__ == '__main__':
    main()
