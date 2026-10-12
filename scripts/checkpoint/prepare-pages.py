#!/usr/bin/env python3
"""Assemble a Pages viewer from a hash-verified immutable browser checkpoint."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

spec = importlib.util.spec_from_file_location('prepare_release', Path(__file__).with_name('prepare-release.py'))
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--source', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output already exists')
    expected, filename = args.archive.with_name(args.archive.name + '.sha256').read_text().split()
    with args.archive.open('rb') as stream:
        if filename != args.archive.name or hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            parser.error('Archive checksum mismatch')
    manifest = release.verify_archive(args.archive, args.source)
    if manifest['profile'] != 'browser-simd128-relaxed-rayon':
        parser.error('Browser checkpoint required')
    args.out.mkdir(parents=True)
    with tarfile.open(args.archive) as archive:
        for member in archive.getmembers():
            relative = Path(member.name).relative_to('checkpoint')
            if member.isfile() and (relative.parts[0] == 'pkg' or str(relative) in
                                     ('index.html', 'app.js', 'worker.js', 'checkpoint.json', 'LICENSE-AGPL3')):
                target = args.out / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
    shutil.copyfile(Path(__file__).resolve().parents[2] / 'web/site/_headers', args.out / '_headers')
    index = args.out / 'index.html'
    text = index.read_text()
    if text.count('src="app.js"') != 1:
        parser.error('Checkpoint viewer entry point changed')
    text = text.replace('<h1>',
        '<nav><a href="https://github.com/imazen/zenmetrics">Source and documentation</a> · '
        '<a href="https://github.com/imazen/zenmetrics/releases">Native and browser downloads</a></nav>\n<h1>', 1)
    index.write_text(text)
    (args.out / 'site-source.json').write_text(json.dumps({'schema':1, 'checkpoint_tag':args.tag,
        'checkpoint_source':args.source, 'archive_sha256':expected}, indent=2) + '\n')


if __name__ == '__main__':
    main()
