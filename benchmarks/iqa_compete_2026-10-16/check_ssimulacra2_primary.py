#!/usr/bin/env python3
"""Primary adapter controls. --official explicitly enables canonical score calls.

Only synthetic control pixels are written. These are adapter ingress checks,
not a Rust parity gate, benchmark matrix, or qualified performance dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--official', type=Path)
    args = parser.parse_args()
    assert digest(args.archive) == '530e9b23b5be10f5981367ba06ce804b3b536a409fcb3b510fa420170304d9c8'
    args.evidence.mkdir()  # Fresh evidence only; preserve existing attempts.
    receipts = []
    with tarfile.open(args.archive) as archive:
        members = [item for item in archive if item.isfile()]
        expected = set()
        for item in members:
            relative = Path(*Path(item.name).parts[1:])
            assert '..' not in relative.parts and not relative.is_absolute()
            expected.add(relative.as_posix())
            assert archive.extractfile(item).read() == (args.source_root / relative).read_bytes(), relative
        actual = {p.relative_to(args.source_root).as_posix()
                  for p in args.source_root.rglob('*') if p.is_file()}
        assert actual == expected, (actual - expected, expected - actual)
    receipts.append({'case': 'all extracted source files equal official archive', 'files': len(expected)})

    def run(name, argv, expected_rc, executable=args.binary):
        result = subprocess.run([str(executable), *map(str, argv)],
                                capture_output=True, text=True, timeout=30)
        (args.evidence / f'{name}.stdout').write_text(result.stdout)
        (args.evidence / f'{name}.stderr').write_text(result.stderr)
        assert result.returncode == expected_rc, (name, result.returncode, result.stderr)
        receipts.append({'case': name, 'rc': result.returncode})
        return result.stdout

    missing = args.evidence / 'missing.rgb'
    base = ['ssim2', '8', '8', missing, missing, '1', 'score', '1']
    run('argc', [], 2)
    for name, index, value in [('metric', 0, 'other'), ('width-zero', 1, '0'),
                               ('negative', 1, '-8'), ('suffix', 1, '8x'),
                               ('uint-overflow', 1, '4294967296'),
                               ('minimum', 2, '7'), ('threads-zero', 5, '0'),
                               ('samples-zero', 7, '0'), ('mode', 6, 'other')]:
        argv = base.copy()
        argv[index] = value
        assert not run(name, argv, 2)
    argv = base.copy()
    argv[1] = argv[2] = '4294967295'
    assert not run('multiplication-overflow', argv, 2)
    argv = base.copy()
    argv[6] = 'score+map'
    assert run('unsupported-map-with-absent-files', argv, 0).startswith('UNSUPPORTED_MAP ')
    assert not run('missing-file', base, 2)
    for name, count in [('short', 191), ('long', 193)]:
        path = args.evidence / f'{name}.rgb'
        path.write_bytes(bytes(count))
        argv = base.copy()
        argv[3] = argv[4] = path
        assert not run(f'{name}-file', argv, 2)

    numerical = []
    if args.official:
        for width, height in [(8, 8), (17, 19), (64, 64)]:
            reference = bytes((x * 31 + y * 17 + c * 83 + x * y * 7) % 256
                              for y in range(height) for x in range(width) for c in range(3))
            distorted = bytes(min(255, max(0, value + (i % 7 - 3) * 5))
                              for i, value in enumerate(reference))
            files = []
            for label, data in [('ref', reference), ('dist', distorted)]:
                raw = args.evidence / f'{width}x{height}-{label}.rgb'
                raw.write_bytes(data)
                ppm = raw.with_suffix('.ppm')
                ppm.write_bytes(f'P6\n{width} {height}\n255\n'.encode() + data)
                files.append((raw, ppm))
            official = run(f'{width}x{height}-official', [files[0][1], files[1][1]], 0,
                           args.official).strip()
            for threads in [1, 2]:
                output = run(f'{width}x{height}-rgb-{threads}',
                             ['ssim2', width, height, files[0][0], files[1][0], threads, 'score', 1], 0)
                rows = [line.split() for line in output.splitlines()
                        if line.startswith(('COLD ', 'SAMPLE '))]
                assert len(rows) == 2 and [r[0] for r in rows] == ['COLD', 'SAMPLE']
                assert all(int(r[1]) > 0 for r in rows)
                assert rows[0][2] == rows[1][2], 'cold/warm score transport differs'
                # The official CLI rounds to eight decimal places. Compare that
                # exact representation, without defining a Rust numerical bound.
                assert f'{float(rows[0][2]):.8f}' == official, (rows, official)
                numerical.append({'width': width, 'height': height, 'threads': threads,
                                  'raw_score': rows[0][2], 'official_display': official,
                                  'input_sha256': [digest(p[0]) for p in files]})
    result = {'binary_sha256': digest(args.binary), 'official_sha256': digest(args.official)
              if args.official else None, 'controls': receipts, 'ingress_comparisons': numerical,
              'qualification': 'adapter controls only; no Rust parity or performance conclusion'}
    (args.evidence / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'controls': len(receipts), 'ingress_comparisons': len(numerical)}))


if __name__ == '__main__':
    main()
