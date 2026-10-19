#!/usr/bin/env python3
"""Prepare explicit precision counterfactuals of the author VSI reference.

This writes diagnostic copies, never modifies the primary reference, and does
not produce a parity verdict. Run the generated entry under both host locks.
"""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE_SHA = '5939b9053966f58377e08052b5e2d0e0e3f954040fe732625591363285433954'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('pairs', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--image-source', type=Path,
                   help='explicit unpacked image package; only its unchanged imresize and private helper are selected')
    a = p.parse_args()
    data = a.source.read_bytes()
    if sha(data) != SOURCE_SHA:
        p.error('unknown primary VSI source')
    text = data.decode()
    variants = {'original': [],
                'lab32': [('lab = RGB2Lab(dsImage);',
                           'lab = RGB2Lab(dsImage); lab = double(single(lab));')],
                'fft32': [],
                'resize32': [('lab = RGB2Lab(dsImage);',
                              'dsImage = double(single(dsImage)); lab = RGB2Lab(dsImage);'),
                             ("VSMap =  imresize(VSMap, [oriRows, oriCols],'bilinear');",
                              "VSMap =  imresize(VSMap, [oriRows, oriCols],'bilinear'); VSMap = double(single(VSMap));")]}
    for channel in ['L', 'A', 'B']:
        variants['fft32'].append((f'{channel}FFT = fft2(double({channel}Channel));',
                                  f'{channel}FFT = fft2(single({channel}Channel));'))
        variants['fft32'].append((f'Final{channel}Result = real(ifft2({channel}FFT.*LG));',
                                  f'Final{channel}Result = double(real(ifft2({channel}FFT.*single(LG))));'))
    a.output.mkdir()
    root = a.output.resolve()
    records = []
    entry = ['more off;', 'pkg load image;', "fftw('threads',1);"]
    resize_binding = None
    if a.image_source:
        package = a.image_source.resolve(strict=True)
        overlay = root / 'resize-overlay'
        (overlay / 'private').mkdir(parents=True)
        files = ['imresize.m', 'private/interp_method.m']
        hashes = {}
        for name in files:
            content = (package / 'inst' / name).read_bytes()
            (overlay / name).write_bytes(content)
            hashes[name] = sha(content)
        resize_binding = {'package_source': str(package), 'files': hashes,
                          'package_installed': False, 'scope': 'two unchanged M-files only; system package remains loaded'}
        entry += [f'addpath({quote(overlay)},\'-begin\'); clear imresize;',
                  f'assert(strcmp(which(\'imresize\'),{quote(overlay / "imresize.m")}));',
                  'fprintf(\'RESIZE_BIND %s\\n\',which(\'imresize\'));']
    pairs = json.loads(a.pairs.read_text())['pairs']
    for i, pair in enumerate(pairs):
        for role in ['reference', 'distorted']:
            path = (a.pairs.parent / pair[role]).resolve(strict=True)
            assert sha(path.read_bytes()) == pair[role + '_sha256']
            count = pair['width'] * pair['height'] * 3
            entry += [f'f=fopen({quote(path)},\'rb\'); assert(f>=0);',
                      f'[v,n]=fread(f,Inf,\'*uint8\'); assert(n=={count}); assert(fclose(f)==0);',
                      f'{role}_{i}=permute(reshape(v,[3,{pair["width"]},{pair["height"]}]),[3,2,1]);']
    for name, replacements in variants.items():
        altered = text
        for before, after in replacements:
            assert altered.count(before) == 1
            altered = altered.replace(before, after)
        directory = root / name
        directory.mkdir()
        path = directory / 'VSI.m'
        path.write_text(altered)
        records.append({'variant': name, 'source_sha256': sha(path.read_bytes()),
                        'replacements': replacements})
        entry += [f'addpath({quote(directory)},\'-begin\'); clear VSI;',
                  f'assert(strcmp(which(\'VSI\'),{quote(path)}));']
        for i, pair in enumerate(pairs):
            entry.append(f'fprintf(\'DIAGNOSTIC {name} {pair["pair"]} %.17g\\n\',VSI(reference_{i},distorted_{i})); fflush(stdout);')
        entry.append(f'rmpath({quote(directory)}); clear VSI;')
    (root / 'entry.m').write_text('\n'.join(entry) + '\n')
    (root / 'manifest.json').write_text(json.dumps({
        'purpose': 'precision counterfactuals only; modified variants are not primary oracles',
        'original_source_sha256': SOURCE_SHA, 'pairs_sha256': sha(a.pairs.read_bytes()),
        'resize_override': resize_binding,
        'variants': records, 'entry_sha256': sha((root / 'entry.m').read_bytes())}, indent=2) + '\n')


if __name__ == '__main__':
    main()
