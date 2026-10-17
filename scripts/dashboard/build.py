#!/usr/bin/env python3
"""Strict registry adapters and dependency-free static dashboard assembly."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]
DATA_SUFFIXES = {'.csv', '.tsv', '.json', '.jsonl', '.parquet', '.meta'}


def inventory(root=ROOT):
    return sorted(p.relative_to(root).as_posix() for p in (root / 'benchmarks').rglob('*')
                  if p.is_file() and (p.suffix in DATA_SUFFIXES or p.name.endswith('.pointer.md')))


def number(value):
    if value is None or value == '':
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'non-finite measurement: {value}')
    return result


def table(path, columns):
    lines = [line for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')]
    reader = csv.DictReader(lines, delimiter='\t' if path.suffix == '.tsv' else ',')
    if set(reader.fieldnames or []) != set(columns):
        raise ValueError(f'{path.name}: schema mismatch: {reader.fieldnames}')
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f'{path.name}: malformed row')
        yield row


def safe_text(text):
    # Publish selected evidence fields only; historical metadata can contain local paths.
    return re.sub(r'(?:/(?:home|mnt|Users|tmp)/[^\s`"<>]*|~[^\s`"<>]*)', '[local path]', text)


def provenance(path, spec):
    paths = [ROOT / p for p in spec.get('metadata', [])]
    paths += [path.with_suffix('.meta'), path.with_suffix('.meta.json')]
    paths += [path.parent / 'README.md']
    existing = list(dict.fromkeys(p for p in paths if p.is_file()))
    evidence = '\n'.join(p.read_text(errors='strict') for p in existing)
    commits = re.findall(r'(?:build_commit|repo_commit[^:\n]*|commit_sha)[\"\s]*[:=][\"\s]*([0-9a-f]{40})', evidence)
    dates = re.findall(r'20\d\d-\d\d-\d\d', str(path))
    return dict(source=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                commit=commits[0] if commits else 'missing', host=spec.get('host', 'missing'),
                date=dates[0] if dates else 'missing', command=spec.get('command', 'missing'),
                metadata=[p.relative_to(ROOT).as_posix() for p in existing],
                note=spec.get('note', 'Historical observation; consult source conditions.'))


def adapt(path, spec):
    adapter = spec['adapter']
    if adapter == 'loop':
        for row in table(path, spec['columns']):
            for mode, col in [('cold', 'oneshot_ms'), ('warm reference', 'warm_per_cand_ms')]:
                yield dict(metric=row['metric'], implementation='zenmetrics', backend='CPU',
                           host=spec['host'], os='macOS', threads=spec['threads'], mode=mode,
                           width=int(row['w']), height=int(row['h']), ms=number(row[col]), status='ok')
    elif adapter == 'parity':
        for row in table(path, spec['columns']):
            ours, ref = number(row['ours']), number(row['oracle'])
            yield dict(metric=spec['metric'], pair=row['key'], ours=ours, reference=ref,
                       abs_delta=abs(ours-ref), relative_delta=abs((ours-ref)/ref) if ref else None,
                       jnd_delta=number(row['oracle_jnd_error']), size='missing', status='historical')
    elif adapter == 'quality':
        for row in table(path, spec['columns']):
            yield dict(metric=f"margarine/{row['arm']}/{row['norm']}", dataset=row['dataset'],
                       scope=row['scope'], subset=row['key'], n=int(row['n']), status=row['status'],
                       srocc=number(row['srocc']), plcc=number(row['plcc']), krocc=number(row['krocc']),
                       ci='missing', mos=None, score=None)
    elif adapter == 'competition':
        rows = json.loads(path.read_text()) if path.suffix == '.json' else [json.loads(l) for l in path.read_text().splitlines()]
        if not isinstance(rows, list):
            raise ValueError('competition input must be a row array or JSONL')
        for row in rows:
            extra = set(row) - set(spec['columns'])
            if extra:
                raise ValueError(f'{path}: unknown competition columns: {extra}')
            for field in ['host', 'metric_name', 'impl', 'device', 'threads', 'mode', 'width', 'height', 'status']:
                if field not in row:
                    raise ValueError(f'missing competition field: {field}')
            for field in ['width', 'height', 'threads']:
                if not isinstance(row[field], int) or row[field] <= 0:
                    raise ValueError(f'invalid competition {field}')
            if row.get('median_ns') is not None and number(row['median_ns']) <= 0:
                raise ValueError('nonpositive timing')
            if row['status'] == 'ok' and row.get('median_ns') is None:
                raise ValueError('successful timing row has no measurement')
            result = dict(metric=row['metric_name'], implementation=row['impl'], backend=row['device'],
                          host=row['host'], os=row.get('os', 'missing'), threads=row['threads'], mode=row['mode'],
                          width=row['width'], height=row['height'], ms=None if row.get('median_ns') is None else number(row['median_ns'])/1e6,
                          status=row['status'], size=row.get('size_name', f"{row['width']}×{row['height']}"), engine=row.get('engine', 'missing'),
                          comparison_status=row.get('comparison_status', 'missing'),
                          reproduction_required=row.get('reproduction_required', False),
                          ours=number(row.get('score')), reference=number(row.get('ref_score')),
                          abs_delta=number(row.get('abs_delta')), relative_delta=number(row.get('relative_delta')))
            yield result
    else:
        raise ValueError(f'unknown adapter: {adapter}')


def catalog():
    source = (ROOT/'crates/zenmetrics-cli/src/metrics/mod.rs').read_text()
    names = re.findall(r'MetricKind::\w+ => "([^"]+)"', source.split('pub fn name(self)')[1].split('pub fn backend')[0])
    ledger = (ROOT/'docs/METRIC_PROVENANCE.md').read_text().splitlines()
    cards = list((ROOT/'crates/zenmetrics-nn/models').glob('*/card.toml'))
    for name in sorted(set(names)):
        entries = [line for line in ledger if line.startswith('| ') and f'`{name}`' in line.split('|')[1]]
        fields = entries[0].split('|') if entries else []
        card = next((p for p in cards if p.parent.name == name.replace('-', '_')), None)
        license_name = 'missing; see source'
        if card:
            parsed = tomllib.loads(card.read_text())
            license_name = str(parsed.get('weights', {}).get('terms', parsed.get('weights', {}).get('licence_class', license_name)))
        yield dict(metric=name, backend='GPU' if name.endswith('-gpu') else 'CPU (build features apply)',
                   reference=safe_text(fields[4].strip()) if len(fields)>4 else 'missing',
                   parity='Evidence linked; no qualification inferred' if entries else 'missing',
                   license=license_name, docs='docs/METRIC_PROVENANCE.md')


def build(output):
    registry = tomllib.loads((ROOT/'benchmarks/dashboard/datasets.toml').read_text())
    families = {k: [] for k in ['speed', 'agreement', 'quality', 'catalog']}
    sources = {}; registered = set()
    for spec in registry['dataset']:
        paths = sorted(ROOT.glob(spec['glob']))
        if not paths and not spec.get('allow_absent', False):
            raise ValueError(f"registered glob has no files: {spec['glob']}")
        for path in paths:
            key = path.relative_to(ROOT).as_posix()
            if key in registered:
                raise ValueError(f'duplicate registration: {key}')
            registered.add(key); sources[key] = provenance(path, spec)
            for row in adapt(path, spec):
                row['source'] = key
                families[spec['family']].append(row)
                if spec['adapter']=='competition' and row.get('abs_delta') is not None:
                    families['agreement'].append(row.copy())
    families['catalog'] = list(catalog())
    output.mkdir(parents=True, exist_ok=True)
    for path in (ROOT/'web/dashboard').iterdir():
        if path.is_file(): shutil.copyfile(path, output/path.name)
    data = output/'data'; data.mkdir(exist_ok=True)
    for family, rows in families.items():
        (data/f'{family}.json').write_text(json.dumps(rows, allow_nan=False, separators=(',', ':')))
    manifest = dict(schema=1, counts={k:len(v) for k,v in families.items()}, sources=sources,
                    registered=sorted(registered), unregistered=sorted(set(inventory())-registered))
    (data/'manifest.json').write_text(json.dumps(manifest, indent=2))
    # No raw metadata, local paths or input-image identifiers are copied into the site.
    for p in output.rglob('*'):
        if p.is_file() and re.search(r'/home/|/mnt/v|/Users/|192\.168\.', p.read_text()):
            raise ValueError(f'private location in public output: {p.name}')
    print(json.dumps(manifest['counts']))
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
