#!/usr/bin/env python3
"""Attach verified checkpoint archives to a GitHub prerelease after README approval."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = 'imazen/zenmetrics'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--readme-approved', action='store_true')
    parser.add_argument('--readme-sha256', required=True, help='SHA-256 of the reviewed source README')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.readme_approved:
        parser.error('Explicit approval of the source README is required')
    if not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+-checkpoint\.[A-Za-z0-9][A-Za-z0-9.-]*', args.tag):
        parser.error('Use a versioned checkpoint tag: vX.Y.Z-checkpoint.ID')
    if not re.fullmatch(r'[0-9a-f]{40}', args.source):
        parser.error('Full immutable source SHA required')
    if not re.fullmatch(r'[0-9a-f]{64}', args.readme_sha256):
        parser.error('Full README SHA-256 required')
    root = Path(__file__).resolve().parents[2]
    readme = subprocess.check_output(['git', 'show', f'{args.source}:README.md'], cwd=root)
    if hashlib.sha256(readme).hexdigest() != args.readme_sha256:
        parser.error('Reviewed README hash does not match the release source')
    # Preparation rechecks all platform jobs, required workflows and artifact hashes.
    subprocess.run([sys.executable, str(Path(__file__).with_name('prepare-release.py')),
                    '--run', str(args.run), '--source', args.source, '--out', str(args.out)], check=True)
    index_path = args.out / 'release-assets.json'
    index = json.loads(index_path.read_text())
    index['tag'] = args.tag
    index['readme_sha256'] = args.readme_sha256
    index_path.write_text(json.dumps(index, indent=2) + '\n')
    assets = []
    for archive in index['archives']:
        assets.extend([archive, archive + '.sha256'])
    assets.append(str(index_path.resolve()))
    command = ['gh', 'release', 'create', args.tag, *assets, '--repo', REPO,
               '--prerelease', '--target', args.source,
               '--title', f'CLI and browser checkpoint {args.tag}', '--generate-notes']
    if args.dry_run:
        print(json.dumps({'tag': args.tag, 'source': args.source, 'command': command}, indent=2))
        return
    # Refuse replacing tags or assets. All gates above run before any mutation.
    remote_tags = json.loads(subprocess.check_output(
        ['gh', 'api', f'repos/{REPO}/git/matching-refs/tags/{args.tag}'], text=True))
    if any(ref['ref'] == f'refs/tags/{args.tag}' for ref in remote_tags):
        parser.error('Checkpoint tag already exists; replacement is refused')
    subprocess.run(['git', 'tag', args.tag, args.source], cwd=root, check=True)
    subprocess.run(['git', 'push', 'origin', f'refs/tags/{args.tag}'], cwd=root, check=True)
    subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
