#!/usr/bin/env python3
"""Pixel-only preparation and hash gate for the committed colour population.

prepare writes a canonical-decoder input; verify runs after that decoder.
Neither operation reads human values or chooses rows by metric scores.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/var/tmp/gmsd-chroma/colour_v2')
POPULATION_SHA = '13db0836bff1966a8593b89907752e3c7bdc9b9fc26116039280f33aadf91940'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    population = ROOT/'population.json'
    assert sha(population) == POPULATION_SHA
    pairs = json.loads(population.read_text())['pairs']
    assert all(r['role'] == 'TRAIN' for r in pairs)
    assert len({r['pair_key'] for r in pairs}) == len(pairs)
    if sys.argv[1] == 'prepare':
        out = ROOT/'decode_pairs.tsv'
        with out.open('x') as f:
            for r in pairs:
                f.write('\t'.join(r[k] for k in
                        ('pair_key','ref_sha','dist_sha','ref_path','dist_path'))+'\n')
    elif sys.argv[1] == 'verify':
        raw = ROOT/'rgb'
        with (raw/'planes.tsv').open() as f:
            planes = list(csv.reader(f, delimiter='\t'))
        by_key = {r[0]: r for r in planes}
        assert len(by_key) == len(planes) == len(pairs)
        assert by_key.keys() == {r['pair_key'] for r in pairs}
        checked = set()
        out = ROOT/'raw_pairs.tsv'
        with out.open('x') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            writer.writerow(['pair_key','width','height','ref_rgb','dist_rgb'])
            for r in pairs:
                p = by_key[r['pair_key']]
                w, h = r['width'], r['height']
                assert (int(p[1]), int(p[2])) == (w,h)
                files = []
                for side, filename in zip(('ref','dist'),p[3:]):
                    digest = r[side+'_sha']
                    assert filename == digest+'.rgb'
                    path = raw/filename
                    assert path.stat().st_size == 3*w*h
                    if digest not in checked:
                        assert sha(path) == digest
                        checked.add(digest)
                    files.append(str(path))
                writer.writerow([r['pair_key'],w,h,*files])
        print('verified_unique_rgb',len(checked))
    else:
        raise SystemExit('expected prepare or verify')
    print('colour_pairs',len(pairs))
    print('output',str(out))
    print('output_sha256',sha(out))


if __name__ == '__main__':
    main()
