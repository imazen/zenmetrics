#!/usr/bin/env python3
"""Independently check archive-byte preservation and overlay rejection controls."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import image_resize_source as implementation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('evidence', type=Path)
    a = p.parse_args()
    a.evidence.mkdir()
    evidence = a.evidence.resolve()
    overlay = evidence / "quoted source's overlay"
    receipt = implementation.prepare(a.archive, overlay)
    implementation.verify(a.archive, overlay, receipt)
    checks = ['fresh source accepted', 'unchanged overlay verified']
    with tarfile.open(a.archive) as original:
        for name in ['imresize.m', 'private/interp_method.m']:
            expected = original.extractfile('image-2.20.1/inst/' + name).read()
            assert (overlay / name).read_bytes() == expected
            assert hashlib.sha256(expected).hexdigest() == receipt['files'][name]
            checks.append(name + ' exactly matches archive bytes')
    sentinel = overlay / 'retained.txt'
    sentinel.write_text('preserve existing evidence\n')
    try:
        implementation.prepare(a.archive, overlay)
    except FileExistsError:
        pass
    else:
        raise AssertionError('existing output accepted')
    assert sentinel.read_text() == 'preserve existing evidence\n'
    checks.append('existing output preserved and refused')
    wrong = evidence / 'wrong.tar.gz'
    wrong.write_bytes(b'not the pinned archive')
    destination = evidence / 'must remain absent'
    try:
        implementation.prepare(wrong, destination)
    except ValueError:
        pass
    else:
        raise AssertionError('wrong archive accepted')
    assert not destination.exists()
    checks.append('wrong archive rejected before output mutation')
    corrupt = evidence / 'deliberately altered control'
    original_receipt = implementation.prepare(a.archive, corrupt)
    with (corrupt / 'imresize.m').open('ab') as stream:
        stream.write(b'\n% deliberate integrity-control mutation\n')
    try:
        implementation.verify(a.archive, corrupt, original_receipt)
    except ValueError:
        pass
    else:
        raise AssertionError('changed overlay accepted')
    checks.append('changed overlay refused')
    result = {'controls': checks, 'count': len(checks), 'receipt': receipt,
              'scope': 'artifact integrity only; no metric executed'}
    (evidence / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
