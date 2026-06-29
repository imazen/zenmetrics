#!/usr/bin/env python3
"""Raw-curve sanity check: for one axis VALUE vs baseline, dump per-image
(q, bytes, score) curves so a big BD-rate can be confirmed real vs thin-overlap.
Usage: inspect_axis.py TSV SCORE_COL CODEC CELL_ID"""
import sys, json, csv, collections
TSV, SCORE_COL, CODEC, CELL = sys.argv[1:5]
base = 's4' if CODEC == 'avif' else 'jp3_t0_small_420'
d = collections.defaultdict(lambda: collections.defaultdict(list))  # img -> cell -> [(q,by,sc)]
for r in csv.DictReader(open(TSV), delimiter='\t'):
    s = r.get(SCORE_COL, '')
    if not s:
        continue
    cell = json.loads(r['knob_tuple_json'])['cell']
    if cell not in (base, CELL):
        continue
    try:
        by = float(r['encoded_bytes'])
    except ValueError:
        continue
    if by <= 0:
        continue
    d[r['image_path']][cell].append((int(r['q']), by, float(s)))
shown = 0
for img in sorted(d):
    if base not in d[img] or CELL not in d[img]:
        continue
    shown += 1
    if shown > 4:
        break
    print(f'\n# {img.split("/")[-1]}')
    print(f'  {"q":>4} {"base_by":>9} {"base_sc":>8}   {"v_by":>9} {"v_sc":>8}   byteR  scD')
    bm = {q: (by, sc) for q, by, sc in d[img][base]}
    vm = {q: (by, sc) for q, by, sc in d[img][CELL]}
    for q in sorted(set(bm) & set(vm)):
        bb, bs = bm[q]; vb, vs = vm[q]
        print(f'  {q:>4} {bb:9.0f} {bs:8.2f}   {vb:9.0f} {vs:8.2f}   {vb/bb:5.3f} {vs-bs:+5.2f}')
