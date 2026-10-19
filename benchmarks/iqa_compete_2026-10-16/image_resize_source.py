"""Bind two unchanged image-2.20.1 M-files without installing a package.

The complete package requires Octave >=8.1. This limited overlay is explicitly
identified and exercised on the available runtime; it is not a package install.
"""
import hashlib
from pathlib import Path
import tarfile

ARCHIVE_SHA = 'b4b6865e7f65ee5b9c254a69128bf10e93b38bc27f5bc5b973fef300217fc725'
ORIGIN = 'https://downloads.sourceforge.net/project/octave/Octave%20Forge%20Packages/Individual%20Package%20Releases/image-2.20.1.tar.gz'
FILES = ['imresize.m', 'private/interp_method.m']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_archive(archive):
    if digest(archive) != ARCHIVE_SHA:
        raise ValueError('image-2.20.1 archive SHA256 mismatch')


def prepare(archive, output):
    validate_archive(archive)
    output.mkdir()
    (output / 'private').mkdir()
    with tarfile.open(archive) as source:
        for name in FILES:
            member = source.getmember('image-2.20.1/inst/' + name)
            if not member.isfile():
                raise ValueError('expected a regular image-package source file')
            data = source.extractfile(member).read()
            with (output / name).open('xb') as stream:
                stream.write(data)
    return {'archive_sha256': ARCHIVE_SHA, 'origin': ORIGIN,
            'adapter_sha256': digest(Path(__file__)),
            'package_version': '2.20.1', 'package_installed': False,
            'package_minimum_octave': '8.1.0',
            'scope': 'unchanged imresize and private interp_method only; other image functions remain system bindings',
            'files': {name: digest(output / name) for name in FILES}}


def verify(archive, output, receipt):
    validate_archive(archive)
    if any(digest(output / name) != expected for name, expected in receipt['files'].items()):
        raise ValueError('resize overlay changed during scoring')
