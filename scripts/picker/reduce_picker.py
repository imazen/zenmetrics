#!/usr/bin/env python3
"""Can zenpicker reduce to a simple algorithm? Measure RD overhead (bytes at target, the metric
that matters) of dumb rules vs the oracle, on the UNBIASED support-complete cells (picker_data).
If 'always-JXL' / a fixed ranking / a cheap k-shot verify costs little, the MLP isn't load-bearing."""
import sys, statistics as st, collections
import numpy as np
sys.path.insert(0, 'scripts/picker')
from picker_data import load_rd, oracle_rows
BASE = '/mnt/v/output/canonical-picker-2026-06-27'
LOSSY = [('jpeg', 'zenjpeg_lossy'), ('webp', 'zenwebp_lossy'), ('jxl', 'zenjxl_lossy'), ('avif', 'zenavif_lossy')]
LL = [('png', 'zenpng_lossless'), ('webp', 'zenwebp_lossless'), ('jxl', 'zenjxl_lossless')]

def report(rows, label, ranking, kshots):
    print(f"\n=== {label}: {len(rows)} support-complete cells, RD overhead vs oracle ===")
    def oh_fam(b, fam):
        return b[fam] / min(b.values()) - 1.0
    def oh_rank(b, order):  # first family in the fixed preference order that's present
        for f in order:
            if f in b:
                return b[f] / min(b.values()) - 1.0
        return None
    def oh_kshot(b, fams):  # encode these k, keep smallest (no model)
        cand = [b[f] for f in fams if f in b]
        return (min(cand) / min(b.values()) - 1.0) if cand else None
    fams = [f for f, _ in (LOSSY if 'lossy' in label.lower() else LL)]
    for f in fams:
        o = sorted(oh_fam(r['bytes'], f) for r in rows if f in r['bytes'])
        print(f"  always-{f:4}      : mean={st.mean(o)*100:5.2f}%  median={o[len(o)//2]*100:5.2f}%  p90={o[int(len(o)*.9)]*100:5.2f}%")
    o = sorted(oh_rank(r['bytes'], ranking) for r in rows)
    print(f"  fixed-rank {'>'.join(ranking):17}: mean={st.mean(o)*100:5.2f}%  median={o[len(o)//2]*100:5.2f}%  p90={o[int(len(o)*.9)]*100:5.2f}%")
    for ks in kshots:
        o = sorted(x for x in (oh_kshot(r['bytes'], ks) for r in rows) if x is not None)
        print(f"  {len(ks)}-shot {'+'.join(ks):16}: mean={st.mean(o)*100:5.2f}%  median={o[len(o)//2]*100:5.2f}%  p90={o[int(len(o)*.9)]*100:5.2f}%  (encode {len(ks)}, keep smallest)")
    # how dominant is the top-ranked family?
    top = ranking[0]
    opt = sum(1 for r in rows if r['oracle'] == top) / len(rows)
    near = sum(1 for r in rows if top in r['bytes'] and r['bytes'][top] <= 1.01 * min(r['bytes'].values())) / len(rows)
    print(f"  {top} IS the oracle on {opt:.0%}; {top} within 1% of oracle on {near:.0%}")

rd = load_rd(BASE, LOSSY, 'test')
rows, _ = oracle_rows(rd, LOSSY, list(np.arange(45, 91, 3.0)), require='all')
report(rows, 'LOSSY family', ['jxl', 'avif', 'webp', 'jpeg'],
       [['jxl', 'avif'], ['jxl', 'webp'], ['jxl', 'avif', 'webp']])
print("  (full MLP lossy router measured: ~3.9% mean / 0% median / ~11% p90)")

import pandas as pd  # lossless is degenerate (all at quality~100) -> direct min-bytes per codec
mb = collections.defaultdict(dict)
for fam, d in LL:
    df = pd.read_parquet(f'{BASE}/{d}/test.parquet', columns=['variant_name', 'score_zensim', 'encoded_bytes']).dropna()
    for v, b in df[df.score_zensim >= 99.999].groupby('variant_name')['encoded_bytes'].min().items():
        mb[v][fam] = float(b)
rowsl = [{'bytes': mb[v], 'oracle': min(mb[v], key=mb[v].get)} for v in mb if len(mb[v]) >= 2]
report(rowsl, 'LOSSLESS family', ['jxl', 'webp', 'png'], [['jxl', 'webp']])
