#!/usr/bin/env python3
"""Work-WEIGHTED completion for the chroma-split arms.

Cell counts overstate progress badly here: the measured zenravif speed ladder
(/mnt/v/output/avif-speed-instrument-2026-09-03/speed_alpha_beta.tsv) puts
83.9 % of the CPU in speeds 1-3.  Weights are used for their SHAPE only -- the
pooled fits carry linear_model_failed=True (POOLING_NOT_MODEL), so they are not
quoted as wall-time.
"""
import csv, json, sys
rows = [r for r in csv.DictReader(open('/mnt/v/output/avif-speed-instrument-2026-09-03/speed_alpha_beta.tsv'), delimiter='\t') if r['backend'] == 'zenravif']
MP = 1.157
W = {int(r['speed']): max(float(r['alpha_ms']) + float(r['beta_ms_per_mp']) * MP, 1.0) for r in rows}
TOTAL_PER_IMAGE_Q = sum(W.values())

def done_weight(pairs_tsv):
    w = 0.0; n = 0; per_speed = {}
    try: rs = list(csv.DictReader(open(pairs_tsv), delimiter='\t'))
    except FileNotFoundError: return 0.0, 0, {}
    for r in rs:
        kt = json.loads(r['knob_tuple_json']); sp = int(kt['speed'])
        w += W[sp]; n += 1; per_speed[sp] = per_speed.get(sp, 0) + 1
    return w, n, per_speed

TOTAL = TOTAL_PER_IMAGE_Q * 9 * 32   # 9 q x 32 images x all 10 speeds
for tag, p in (('br420', sys.argv[1]), ('br444', sys.argv[2])):
    w, n, ps = done_weight(p)
    print(f"{tag}: cells {n}/2880 ({100*n/2880:5.1f}%)  |  WORK {100*w/TOTAL:5.1f}%  "
          f"|  speeds done: {dict(sorted(ps.items()))}")
print(f"\nweight shape (share of one image-q's CPU): "
      + "  ".join(f"s{s}:{100*W[s]/TOTAL_PER_IMAGE_Q:.1f}%" for s in sorted(W)))
