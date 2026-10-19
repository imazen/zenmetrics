#!/usr/bin/env python3
"""Prepare hash-bound RGB8 center crops from retained raw IQA pairs.

No decoding, resizing, or color conversion. Source rows may be strided; only
active crop pixels are copied. Preserves original files and existing attempts.
"""
import argparse
import json
from pathlib import Path

from fixed_factor_oracle import write_artifact
from octave_primary import sha

SIZES = [('64', 64, 64), ('odd', 255, 257), ('1mp', 1024, 1024),
         ('4k', 3840, 2160)]


def crop_rgb8(path, width, height, stride, left, top, out_width, out_height):
    if min(width, height, out_width, out_height) <= 0 or stride < width * 3:
        raise ValueError('invalid RGB8 geometry')
    needed = (height - 1) * stride + width * 3
    if path.stat().st_size < needed:
        raise ValueError('short strided RGB8 file')
    if min(left, top) < 0 or left + out_width > width or top + out_height > height:
        raise ValueError('crop outside source')
    rows = []
    with path.open('rb') as source:
        if left == 0 and stride == width * 3 and out_width == width:
            source.seek(top * stride)
            data = source.read(out_width * out_height * 3)
            if len(data) != out_width * out_height * 3:
                raise ValueError('short packed RGB8 crop')
            return data
        for row in range(out_height):
            source.seek((top + row) * stride + left * 3)
            data = source.read(out_width * 3)
            if len(data) != out_width * 3:
                raise ValueError('short RGB8 row during crop')
            rows.append(data)
    return b''.join(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    if not plan['pairs']:
        p.error('nonempty pair plan required')
    # Verify all source identities before creating the output directory.
    names = set()
    for pair in plan['pairs']:
        name = pair['name']
        if not name or name in names or Path(name).name != name or name in ['.', '..']:
            p.error('unique single-component pair names required')
        names.add(name)
        for role in ['reference', 'distorted']:
            source = Path(pair[role]['path']).resolve(strict=True)
            if sha(source) != pair[role]['sha256']:
                p.error('source SHA256 mismatch')
    a.output.mkdir()  # A new preparation, never replacement or partial overwrite.
    output = a.output.resolve()
    rows = []
    for pair in plan['pairs']:
        width, height, stride = pair['width'], pair['height'], pair['stride']
        for size, w, h in SIZES:
            left, top = (width - w) // 2, (height - h) // 2
            record = {'pair': pair['name'], 'content': pair['content'],
                      'size_name': size, 'width': w, 'height': h,
                      'crop': {'left': left, 'top': top},
                      'source_width': width, 'source_height': height,
                      'source_stride': stride,
                      'method': 'center crop of retained raw RGB8; no pixel arithmetic'}
            for role in ['reference', 'distorted']:
                source = Path(pair[role]['path']).resolve(strict=True)
                target = output / f'{pair["name"]}-{size}-{role}.rgb'
                pixels = crop_rgb8(source, width, height, stride, left, top, w, h)
                write_artifact(target, pixels)
                record[role] = target.name
                record[role + '_sha256'] = sha(target)
                record[role + '_source_sha256'] = pair[role]['sha256']
            rows.append(record)
        for role in ['reference', 'distorted']:
            if sha(Path(pair[role]['path'])) != pair[role]['sha256']:
                raise RuntimeError('source changed during preparation')
    manifest = {'schema': 'primary-raw-crops/1', 'plan_sha256': sha(a.plan),
                'adapter_sha256': sha(Path(__file__)), 'pairs': rows,
                'source_lineage': plan['source_lineage']}
    write_artifact(output / 'pairs.json', (json.dumps(manifest, indent=2) + '\n').encode())
    print(json.dumps({'prepared_pairs': len(rows), 'manifest_sha256': sha(output / 'pairs.json')}))


if __name__ == '__main__':
    main()
