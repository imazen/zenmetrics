"""Classify only explicitly generated native-reference publication artifacts."""
from pathlib import Path


REASONS = {
    'cohort': 'Native reference comparison cohort; separate from canonical multi-host timing and published-column agreement series.',
    'summary': 'Native reference comparison summary; separate from canonical multi-host timing and published-column agreement series.',
    'provenance': 'Native reference comparison source and artifact provenance; not timing observations.',
    'interval': 'Descriptive native reference block-ratio intervals with explicit qualification; separate from canonical multi-host timing series.',
}


def register_outputs(artifacts, repo=None):
    repo = Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[2]
    benchmark_root = repo / 'benchmarks'
    rows = []
    for path, kind in artifacts:
        path = Path(path).resolve()
        if kind not in REASONS:
            raise ValueError('unknown native publication artifact kind')
        if not path.is_relative_to(benchmark_root):
            continue  # Scratch projections do not modify the source repository.
        expected = '.tsv' if kind == 'cohort' else '.json'
        if path.suffix != expected or not path.is_file():
            raise ValueError('generated native publication artifact differs from declared kind')
        name = str(path.relative_to(repo))
        if any(char in name for char in '\t\r\n'):
            raise ValueError('artifact path contains a classification delimiter')
        rows.append((name, kind))
    if not rows:
        return []
    registry = benchmark_root / 'dashboard/classification.tsv'
    original = registry.read_text()
    known = {line.split('\t')[0] for line in original.splitlines()}
    added = []
    for name, kind in rows:
        if name not in known:
            added.append(name + '\tignored\t' + REASONS[kind] + '\n')
            known.add(name)
    if added:
        temporary = registry.with_name('.classification.tsv.new')
        temporary.write_text(original + ''.join(added))
        temporary.replace(registry)
    return [name for name, _ in rows]


def write_cohort_index(path, records):
    """Keep every cohort in bounded, hashed shards when a flat index grows."""
    import hashlib
    import json
    path=Path(path)
    text=json.dumps(records,indent=2)+'\n'
    if len(text.encode())<=30_000:
        path.write_text(text);return []
    shards=[];part={};outputs=[]
    def emit():
        name=f'{path.stem}-{len(shards):02d}.json';target=path.with_name(name)
        data=(json.dumps(part,indent=2)+'\n').encode()
        if len(data)>30_000:raise ValueError('single cohort record exceeds generated-data cap')
        target.write_bytes(data);outputs.append(target)
        shards.append(dict(file=name,cohorts=len(part),sha256=hashlib.sha256(data).hexdigest()))
    for name,row in sorted(records.items()):
        candidate=dict(part,**{name:row})
        if part and len(json.dumps(candidate,indent=2).encode())>24_000:emit();part={}
        part[name]=row
    if part:emit()
    path.write_text(json.dumps(dict(schema=2,kind='cohorts',cohorts=len(records),shards=shards),indent=2)+'\n')
    return outputs


def read_cohort_index(path):
    """Read legacy flat indexes or verify every shard before returning records."""
    import hashlib
    import json
    path=Path(path);index=json.loads(path.read_text())
    if index.get('schema')!=2 or index.get('kind')!='cohorts':return index
    records={}
    for item in index['shards']:
        target=path.parent/item['file'];data=target.read_bytes()
        if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('cohort shard hash mismatch')
        part=json.loads(data)
        if len(part)!=item['cohorts'] or records.keys() & part.keys():raise ValueError('cohort shard count or duplicate key')
        records.update(part)
    if len(records)!=index['cohorts']:raise ValueError('cohort index count mismatch')
    return records
