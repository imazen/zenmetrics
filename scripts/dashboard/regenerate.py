#!/usr/bin/env python3
"""Refresh derived dashboard copies; report drift without suppressing data errors."""
import os
from pathlib import Path
import re
from catalog import ROOT, render as catalog_render
from extract_markdown import render as markdown_render


def refresh(directory, expected, pattern):
    """Replace only this generator's owned files, including obsolete shards."""
    directory.mkdir(parents=True, exist_ok=True)
    actual = {p.name: p.read_bytes() for p in directory.iterdir()
              if p.is_file() and re.fullmatch(pattern, p.name)}
    encoded = {name: text.encode() for name, text in expected.items()}
    added = set(encoded) - set(actual)
    removed = set(actual) - set(encoded)
    changed = {name for name in set(actual) & set(encoded) if actual[name] != encoded[name]}
    for name, data in encoded.items():
        if actual.get(name) != data:
            (directory / name).write_bytes(data)
    for name in removed:
        (directory / name).unlink()
    return len(added), len(changed), len(removed)


def regenerate():
    # Render both before writing: malformed source/schema errors still fail the job.
    outputs = [('Metric catalog', ROOT / 'benchmarks/dashboard/catalog',
                catalog_render(), r'catalog-\d+\.tsv'),
               ('Markdown extracts', ROOT / 'benchmarks/dashboard/extracts',
                markdown_render(), r'(?:tables-\d+\.tsv|sources\.json)')]
    summary = ['## Dashboard derived data', '',
               'Deployment uses freshly generated data. Committed-copy drift is informational.', '',
               '| Derived data | Committed copies differed | Added | Changed | Removed |',
               '| --- | --- | ---: | ---: | ---: |']
    for label, directory, expected, pattern in outputs:
        counts = refresh(directory, expected, pattern)
        summary.append(f'| {label} | {"yes" if any(counts) else "no"} | '+
                       ' | '.join(str(n) for n in counts) + ' |')
    text = '\n'.join(summary) + '\n'
    print(text, end='')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a') as file:
            file.write(text)


if __name__ == '__main__':
    regenerate()
