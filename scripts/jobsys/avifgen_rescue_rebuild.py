#!/usr/bin/env python3
# Rescue verdict + scores rebuild for the avif944 corpus.
#  1. Parse the rescue run's blobs (metric rows only).
#  2. SAMPLE VERDICT: compare rescored vs stored values on the 2,000-pair sample
#     (pairs_rescue_sample.parquet): |d ssim2| > 0.5 or |d butter_max| > 0.5 = mismatch.
#  3. REBUILD scores.parquet: main-cache rows first, rescue rows LAST (deterministic
#     rescue-wins) — features.parquet untouched (CPU queue, clean).
import glob
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MAIN_BLOBS = '/mnt/v/zen/writeback-zenavif-avifgen-sf-gpu-20260806/blobs'
RESCUE_BLOBS = '/mnt/v/zen/writeback-rescue/blobs'
U = '/mnt/v/output/avifgen-2026-08-06/unified'
SCOLS = ['butteraugli_max_gpu', 'butteraugli_pnorm3_gpu', 'ssim2_gpu', 'cvvdp_cpu_imazen_v0_1_0']

def norm(s):
    return os.path.basename(s or '')

def parse_dir(d, store, tag):
    files = glob.glob(d + '/*')
    print(f'{tag}: {len(files)} blob files', flush=True)
    for bp in files:
        with open(bp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get('kind') != 'metric' or 'error' in r:
                    continue
                key = (norm(r['image_path']), norm(r['encode_sha']))
                store.setdefault(key, {}).update(r.get('scores') or {})

# rescue rows
rescue = {}
parse_dir(RESCUE_BLOBS, rescue, 'rescue')
print('rescue pairs with data:', len(rescue), flush=True)

# 2) sample verdict against the stored table
sp = pq.read_table('/mnt/v/output/avifgen-2026-08-06/pairs_rescue_sample.parquet').to_pydict()
stored = pq.read_table(U + '/scores.parquet').to_pydict()
sidx = {}
for i in range(len(stored['image_path'])):
    sidx[(stored['image_path'][i], stored['encode_sha'][i])] = i
mismatch = []
checked = 0
for i in range(len(sp['ref_path'])):
    key = (os.path.basename(sp['ref_path'][i]), os.path.basename(sp['dist_path'][i]))
    r = rescue.get(key)
    si = sidx.get(key)
    if r is None or si is None:
        continue
    ds = abs((r.get('ssim2_gpu') or 0) - (stored['ssim2_gpu'][si] or 0))
    db = abs((r.get('butteraugli_max_gpu') or 0) - (stored['butteraugli_max_gpu'][si] or 0))
    checked += 1
    if ds > 0.5 or db > 0.5:
        mismatch.append((key, round(ds, 3), round(db, 3)))
print(f'SAMPLE VERDICT: checked={checked} mismatches={len(mismatch)} rate={len(mismatch)/max(checked,1):.5f}')
for m in mismatch[:10]:
    print('  MISMATCH', m)
with open('/mnt/v/output/avifgen-2026-08-06/rescue_sample_verdict.json', 'w') as f:
    json.dump({'checked': checked, 'mismatches': len(mismatch),
               'rate': len(mismatch) / max(checked, 1),
               'examples': [[list(k), a, b] for k, a, b in mismatch[:50]]}, f, indent=1)
if len(mismatch) / max(checked, 1) > 0.005:
    print('VERDICT: BROADER CORRUPTION — full re-score required; NOT rebuilding')
    sys.exit(2)

# 3) rebuild scores: overlay rescue rows onto the stored table
t = pq.read_table('/mnt/v/output/avifgen-2026-08-06/pairs_final_fanned.parquet',
                  columns=['image_path', 'q', 'knob_tuple_json', 'encode_sha']).to_pydict()
n = len(t['image_path'])
key_rows = {}
for i in range(n):
    key_rows.setdefault((os.path.basename(t['image_path'][i]), norm(t['encode_sha'][i])), []).append(i)
arr = {c: np.array(stored[c], dtype=np.float64) for c in SCOLS}
# stored table rows are aligned with pairs_final_fanned (same build order) — verify
assert stored['image_path'] == [os.path.basename(x) for x in t['image_path']], 'row alignment lost'
patched = 0
for key, scores in rescue.items():
    rows = key_rows.get(key)
    if not rows:
        continue
    for c, v in scores.items():
        if c in arr and v is not None:
            for i in rows:
                arr[c][i] = v
    patched += len(rows)
print('cells patched with rescue values:', patched, flush=True)
out = pa.table({
    'image_path': stored['image_path'], 'q': stored['q'],
    'knob_tuple_json': stored['knob_tuple_json'], 'encode_sha': stored['encode_sha'],
    **{c: pa.array(arr[c]) for c in SCOLS}})
os.rename(U + '/scores.parquet', U + '/scores.pre-rescue.bak.parquet')
pq.write_table(out, U + '/scores.parquet', compression='zstd')
print('scores.parquet REBUILT (pre-rescue kept as scores.pre-rescue.bak.parquet)')
print('RESCUE_REBUILD_DONE')
