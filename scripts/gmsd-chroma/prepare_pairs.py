#!/usr/bin/env python3
"""Freeze label-free TRAIN pairs, then stage their previously verified RGB8.

select: writes selections and the canonical decoder's input TSV.
stage: after decoding KADID via gmsbank_decode_dump, checks raw pixel hashes
and writes a single oracle TSV, plus crops/identities. No labels are read.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/var/tmp/gmsd-chroma')
CAL = Path('/var/tmp/gmsbank/calibration')
KADID = Path('/var/tmp/gmsbank/peer_gmsd/kadid_train.keys.tsv')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def order(value):
    return hashlib.sha256(('20260924:' + value).encode()).hexdigest()


def select():
    assert sha(KADID) == '0d573a4452d908c79c7128f62d3e2d3ddc02c9ef4a57c0ba732507b9dd00d398'
    selected = []
    all_rows = [line.split('\t') for line in KADID.read_text().splitlines()]
    for kind in ('04', '05', '06', '07', '08'):
        candidates = [r for r in all_rows if Path(r[6]).stem.split('_')[1] == kind]
        seen = set()
        for r in sorted(candidates, key=lambda r: order(r[0])):
            if r[1] in seen:
                continue
            seen.add(r[1])
            selected.append(dict(set='kadid_train', kind=kind, key=r[0], ref_sha=r[1],
                                 dist_sha=r[2], width=int(r[3]), height=int(r[4]),
                                 ref_path=r[5], dist_path=r[6], raw_root=str(ROOT/'kadid_rgb')))
            if len(seen) == 8:
                break
        assert len(seen) == 8
    records = {r['pair_key']: r for r in json.loads((CAL/'selection.json').read_text())['pairs']}
    paths = {r[0]: r for r in (line.split('\t') for line in (CAL/'pairs.tsv').read_text().splitlines())}
    dims = {r[0]: (int(r[1]), int(r[2])) for r in
            (line.split('\t') for line in (CAL/'planes.tsv').read_text().splitlines())}
    for group in ('cid22_train', 'safesyn'):
        seen = set()
        for key in sorted(records, key=order):
            r = paths[key]
            if records[key]['set'] != group or r[1] in seen:
                continue
            seen.add(r[1])
            w, h = dims[key]
            selected.append(dict(set=group, kind='stored', key=key, ref_sha=r[1], dist_sha=r[2],
                                 width=w, height=h, ref_path=r[3], dist_path=r[4], raw_root=str(CAL)))
            if len(seen) == 16:
                break
        assert len(seen) == 16
    assert len(selected) == 72
    (ROOT/'selection.json').write_text(json.dumps(dict(seed=20260924, pairs=selected), indent=2)+'\n')
    with (ROOT/'kadid_decode_pairs.tsv').open('w') as f:
        for r in selected:
            if r['set'] == 'kadid_train':
                f.write('\t'.join(r[k] for k in ('key','ref_sha','dist_sha','ref_path','dist_path'))+'\n')
    print('native_train_pairs', len(selected))
    print('selection_sha256', sha(ROOT/'selection.json'))
    print('decode_pairs_sha256', sha(ROOT/'kadid_decode_pairs.tsv'))


def stage():
    out = ROOT/'inputs'
    out.mkdir(exist_ok=True)
    rows = []
    manifest = []
    for i, r in enumerate(json.loads((ROOT/'selection.json').read_text())['pairs']):
        data = []
        for side in ('ref', 'dist'):
            path = Path(r['raw_root'])/(r[side+'_sha']+'.rgb')
            assert sha(path) == r[side+'_sha'], path
            b = path.read_bytes()
            assert len(b) == r['width']*r['height']*3
            data.append(b)
        cases = [('native', r['width'], r['height'])]
        if i < 12:
            cases += [('odd', min(r['width'], 63), min(r['height'], 47)), ('tiny', 17, 13),
                      ('odd_large', min(r['width'], 511), min(r['height'], 385))]
        for variant, w, h in cases:
            name = f'{i:03}_{variant}'
            files = []
            for side, b in zip(('ref', 'dist'), data):
                cropped = b''.join(b[3*y*r['width']:3*y*r['width']+3*w] for y in range(h))
                path = out/(name+'.'+side+'.rgb')
                path.write_bytes(cropped)
                files.append(str(path))
                manifest.append(dict(path=str(path), sha256=sha(path), source=r[side+'_sha'],
                                     crop=[0,0,w,h], role='TRAIN', set=r['set']))
            rows.append([name,w,h,*files])
        if i < 4:
            rows.append([f'{i:03}_identity',r['width'],r['height'],rows[-len(cases)][3],rows[-len(cases)][3]])
    # Adversarial colour values exercise principal complex roots; these are
    # explicitly synthetic oracle tests, never TRAIN corpus membership.
    for variant, (w,h) in enumerate(((1,1),(3,5),(17,13),(33,31))):
        files=[]
        for side in ('ref','dist'):
            b=bytearray()
            for y in range(h):
                for x in range(w):
                    colour=(255,0,0) if ((x+y)%3) else (0,255,0)
                    if side=='dist':
                        colour=tuple(255-v for v in colour)
                    b.extend(colour)
            path=out/f'stress{variant}.{side}.rgb'
            path.write_bytes(b)
            files.append(str(path))
            manifest.append(dict(path=str(path),sha256=sha(path),role='synthetic-oracle'))
        rows.append([f'stress{variant}',w,h,*files])
    with (ROOT/'oracle_inputs.tsv').open('w') as f:
        writer=csv.writer(f,delimiter='\t',lineterminator='\n')
        writer.writerow(['id','width','height','ref_rgb','dist_rgb'])
        writer.writerows(rows)
    (ROOT/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('oracle_rows',len(rows))
    print('oracle_inputs_sha256',sha(ROOT/'oracle_inputs.tsv'))
    print('input_manifest_sha256',sha(ROOT/'input_manifest.json'))


if __name__ == '__main__':
    {'select': select, 'stage': stage}[sys.argv[1]]()
