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
