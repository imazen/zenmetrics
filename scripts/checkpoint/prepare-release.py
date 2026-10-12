#!/usr/bin/env python3
"""Download and verify green checkpoint assets; never tag or publish a release."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

REPO = 'imazen/zenmetrics'
def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', f'repos/{REPO}' + (f'/{path}' if path else '')], text=True))

def verify_archive(path, source):
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        for member in members:
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != 'checkpoint' or '..' in parts or not (member.isdir() or member.isfile()):
                raise ValueError(f'Unexpected archive member: {member.name}')
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive member')
        with archive.extractfile('checkpoint/checkpoint.json') as stream:
            manifest = json.load(stream)
        if manifest['schema'] != 1 or manifest['source_commit'] != source:
            raise ValueError('Checkpoint source/schema mismatch')
        expected = {'checkpoint/' + relative for relative in manifest['files']} | {'checkpoint/checkpoint.json'}
        if {m.name for m in members if m.isfile()} != expected:
            raise ValueError('Archive file set differs from manifest')
        for relative, expected_hash in manifest['files'].items():
            with archive.extractfile('checkpoint/' + relative) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected_hash:
                    raise ValueError(f'Artifact hash mismatch: {relative}')
    return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=int)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if len(args.source) != 40 or any(c not in '0123456789abcdef' for c in args.source):
        parser.error('Full immutable source SHA required')
    if args.out.exists(): parser.error('Output already exists')
    run = api(f'actions/runs/{args.run}')
    if run['event'] == 'pull_request' or run['head_sha'] != args.source or run['conclusion'] != 'success':
        parser.error('Release requires a successful source-matching push/dispatch run')
    if run['path'] != '.github/workflows/checkpoint.yml':
        parser.error('Run is not the checkpoint workflow')
    jobs = api(f'actions/runs/{args.run}/jobs?per_page=100')['jobs']
    if not jobs or any(job['conclusion'] != 'success' for job in jobs):
        parser.error('Every checkpoint job must succeed')
    repository = api('')
    comparison = api(f'compare/{args.source}...{repository["default_branch"]}')
    if comparison['status'] not in ('ahead', 'identical'):
        parser.error('Source has not landed on the default branch')
    runs = api(f'actions/runs?head_sha={args.source}&per_page=100')['workflow_runs']
    for required in ['CI', 'Native benchmark fleet']:
        candidates = [r for r in runs if r['name'] == required and r['event'] != 'pull_request']
        if not candidates or max(candidates, key=lambda r: r['id'])['conclusion'] != 'success':
            parser.error(f'{required} must be green on this source')
    config = json.loads((Path(__file__).resolve().parents[2] / 'ci/checkpoint-profiles.json').read_text())
    artifacts = api(f'actions/runs/{args.run}/artifacts?per_page=100')['artifacts']
    if {a['name'] for a in artifacts if not a['expired']} != set(config['release_artifacts']):
        parser.error('Missing, extra or expired target artifacts')
    subprocess.run(['gh','run','download',str(args.run),'--repo',REPO,'--dir',str(args.out)],check=True)
    archives = []
    for artifact in config['release_artifacts']:
        path = args.out / artifact / (artifact + '.tar.gz')
        expected, name = path.with_name(path.name + '.sha256').read_text().split()
        with path.open('rb') as stream:
            if name != path.name or hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'Archive checksum mismatch: {artifact}')
        manifest = verify_archive(path, args.source)
        if artifact == 'checkpoint-browser-simd128-relaxed-rayon':
            if (manifest.get('target'), manifest.get('profile')) != ('wasm32-unknown-unknown', 'browser-simd128-relaxed-rayon'):
                raise ValueError('Browser artifact target mismatch')
        elif artifact != f"checkpoint-{manifest.get('target')}-{manifest.get('profile')}":
            raise ValueError(f'Native artifact identity mismatch: {artifact}')
        archives.append(str(path.resolve()))
    (args.out / 'release-assets.json').write_text(json.dumps({'schema':1,'source':args.source,
        'run':run['html_url'],'archives':archives},indent=2)+'\n')
    print(f'Verified {len(archives)} checkpoint archives. Ready for GitHub checkpoint publication.')

if __name__ == '__main__': main()
