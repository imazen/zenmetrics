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
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--archive', type=Path)
    inputs.add_argument('--build', type=Path, help='Fresh source-matching browser build')
    parser.add_argument('--source', required=True)
    parser.add_argument('--tag')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output already exists')
    if len(args.source)!=40 or any(c not in '0123456789abcdef' for c in args.source):
        parser.error('Full source commit SHA required')
    if args.build:
        if not (args.build/'pkg/zenmetrics_browser_bg.wasm').is_file():
            parser.error('Compiled browser module required')
        args.out.mkdir(parents=True)
        for name in ('pkg','pkg-full','codecs','ort','index.html','app.js','worker.js','nn.js','nn-models.json','LICENSE-AGPL3'):
            source=args.build/name
            if source.is_dir(): shutil.copytree(source,args.out/name)
            else: shutil.copyfile(source,args.out/name)
        # Every CLI metric's browser status, verified against the module by
        # check-pages.py; also read cross-origin by the dashboard.
        shutil.copyfile(Path(__file__).resolve().parents[2] / 'web/metrics/coverage.json', args.out / 'coverage.json')
        provenance={'schema':2,'source_commit':args.source,'files':{
            str(path.relative_to(args.out)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.out.rglob('*')) if path.is_file()}}
    else:
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
        provenance={'schema':1,'checkpoint_tag':args.tag,'checkpoint_source':args.source,'archive_sha256':expected}
    for name in ('_headers', '_redirects'):
        shutil.copyfile(Path(__file__).resolve().parents[2] / 'web/site' / name, args.out / name)
    index = args.out / 'index.html'
    text = index.read_text()
    if text.count('src="app.js"') != 1:
        parser.error('Checkpoint viewer entry point changed')
    if not args.build:
        text = text.replace('<h1>',
        '<nav><a href="https://github.com/imazen/zenmetrics">Source and documentation</a> · '
        '<a href="https://github.com/imazen/zenmetrics/releases">Native and browser downloads</a></nav>\n<h1>', 1)
    # Every module, WASM file and nested Rayon worker shares one immutable URL
    # generation. Fixed package paths can mix cached glue with a newer module.
    generation = Path('assets') / args.source
    destination = args.out / generation
    destination.mkdir(parents=True)
    for name in ('app.js', 'worker.js', 'nn.js', 'nn-models.json', 'pkg', 'pkg-full', 'codecs', 'ort'):
        if (args.out / name).exists():
            shutil.move(str(args.out / name), destination / name)
    text = text.replace('src="app.js"', f'src="{generation}/app.js"')
    # Compatibility worker entry for qualification tools, never used by the UI.
    (args.out / 'worker.js').write_text(f"import './{generation}/worker.js';\n")
    provenance['asset_prefix'] = str(generation)
    index.write_text(text)
    if args.build:
        provenance['files']={str(path.relative_to(args.out)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.out.rglob('*')) if path.is_file()}
    (args.out / 'site-source.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
