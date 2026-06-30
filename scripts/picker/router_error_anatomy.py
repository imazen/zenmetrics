#!/usr/bin/env python3
"""Error anatomy of the SHIPPED i8 routers: not just family-acc, but per-misroute
  - iso-quality BYTE overhead (router family needs N% more bytes to hit the same zq), and
  - iso-bytes QUALITY deficit (at the oracle's byte budget, how many zq points the router
    family falls short)
plus a confusion matrix, zq-band breakdown, and a focused zq95-98 'dead-zone' study
(lossy reachability, byte explosion near the ceiling, the AVIF-vs-JXL margin, gate coin-flip).
Router picks come from the shipped i8 forward (numpy quant matching the .bin exactly)."""
import json, collections, statistics as st
import numpy as np, pandas as pd
SIDE = '/mnt/v/output/router-features-2026-06-30/zenanalyze_features.parquet'
BASE = '/mnt/v/output/canonical-picker-2026-06-27'
OUT = '/mnt/v/output/router-features-2026-06-30'
NAMES = ['jpeg', 'webp', 'jxl', 'avif', 'png', 'gif']
FAM_IDX = {n: i for i, n in enumerate(NAMES)}
LOSSY = [('jpeg', 'zenjpeg_lossy'), ('webp', 'zenwebp_lossy'), ('jxl', 'zenjxl_lossy'), ('avif', 'zenavif_lossy')]
LL = [('png', 'zenpng_lossless'), ('webp', 'zenwebp_lossless'), ('jxl', 'zenjxl_lossless')]
side = pd.read_parquet(SIDE).drop_duplicates('variant_name').set_index('variant_name')
QCOLS = [c for c in side.columns if '@' in c]
snp = side[QCOLS].to_numpy(np.float64); vix = {v: i for i, v in enumerate(side.index)}

def bytes_at(pts, zq):
    pts = sorted(pts)
    for i in range(1, len(pts)):
        z0, b0 = pts[i-1]; z1, b1 = pts[i]
        if z0 <= zq <= z1 and z1 > z0: return b0 + (b1-b0)*(zq-z0)/(z1-z0)
    return None

def zq_at(pts, B):  # invert: bytes -> achieved zq (bytes monotone-up with zq); clamp at ends
    pts = sorted(pts)
    if B <= pts[0][1]: return pts[0][0]
    if B >= pts[-1][1]: return pts[-1][0]
    for i in range(1, len(pts)):
        z0, b0 = pts[i-1]; z1, b1 = pts[i]
        if b0 <= B <= b1 and b1 > b0: return z0 + (z1-z0)*(B-b0)/(b1-b0)
    return pts[-1][0]

def zq_ceiling(pts): return max(z for z, _ in pts)

def lossy_rd(s):
    rd = collections.defaultdict(lambda: collections.defaultdict(list))
    for fam, d in LOSSY:
        df = pd.read_parquet(f'{BASE}/{d}/{s}.parquet', columns=['variant_name', 'score_zensim', 'encoded_bytes']).dropna()
        for v, z, b in zip(df.variant_name.values, df.score_zensim.values, df.encoded_bytes.values): rd[v][fam].append((float(z), float(b)))
    return rd

def ll_min(s):
    mb = collections.defaultdict(dict)
    for fam, d in LL:
        df = pd.read_parquet(f'{BASE}/{d}/{s}.parquet', columns=['variant_name', 'score_zensim', 'encoded_bytes']).dropna()
        for v, b in df[df.score_zensim >= 99.999].groupby('variant_name')['encoded_bytes'].min().items(): mb[v][fam] = float(b)
    return mb

# shipped i8 forward (matches the .bin: per-output max-abs scale /127, round-half-away, dequant)
def qd_i8(W):
    sc = np.abs(W).max(0) / 127.0; safe = np.where(sc == 0, 1.0, sc)
    q = np.clip(np.sign(W) * np.floor(np.abs(W / safe) + 0.5), -128, 127)
    return np.where(sc == 0, 0.0, q) * sc

def fwd_i8(spec, X):
    h = (X - np.array(spec['scaler_mean'])) / np.array(spec['scaler_scale'])
    for L in spec['layers']:
        W = qd_i8(np.array(L['weights']).reshape(L['in_dim'], L['out_dim']))
        h = h @ W + np.array(L['biases'])
        if L['activation'] == 'relu': h = np.maximum(h, 0.0)
    return h

lossy_spec = json.load(open(f'{OUT}/router_lossy.bakereq.json'))
gate_spec = json.load(open(f'{OUT}/router_gate.bakereq.json'))
rd = lossy_rd('test'); mb = ll_min('test')

# ---- Part 1: lossy router misroute anatomy ----
rows = []
for v in rd:
    if v not in vix: continue
    base = snp[vix[v]]
    for zq in np.arange(45, 91, 3.0):
        bb = {f: bytes_at(rd[v][f], zq) for f in rd[v]}; bb = {f: b for f, b in bb.items() if b is not None}
        if len(bb) < 2: continue
        oracle = min(bb, key=bb.get)
        out = fwd_i8(lossy_spec, np.append(base, zq))
        m = np.full(6, np.inf);
        for f in bb: m[FAM_IDX[f]] = out[FAM_IDX[f]]
        pred = NAMES[int(np.argmin(m))]
        byte_oh = bb[pred] / bb[oracle] - 1.0
        # iso-bytes quality deficit: pred family at oracle's byte budget -> achieved zq
        q_ach = zq_at(rd[v][pred], bb[oracle]); q_gap = zq - q_ach
        rows.append((v, zq, oracle, pred, byte_oh, q_gap))
df = pd.DataFrame(rows, columns=['v', 'zq', 'oracle', 'pred', 'byte_oh', 'q_gap'])
wrong = df[df.oracle != df.pred]
print(f"=== LOSSY ROUTER (shipped i8) — {len(df)} cells, acc={1-len(wrong)/len(df):.1%} ===")
print(f"  ALL cells   byte-overhead: mean={df.byte_oh.mean()*100:.2f}% p90={df.byte_oh.quantile(.9)*100:.2f}% p99={df.byte_oh.quantile(.99)*100:.2f}%")
print(f"  MISROUTES   byte-overhead: mean={wrong.byte_oh.mean()*100:.2f}% median={wrong.byte_oh.median()*100:.2f}% p90={wrong.byte_oh.quantile(.9)*100:.2f}% max={wrong.byte_oh.max()*100:.1f}%")
print(f"  MISROUTES   quality-gap (zq pts at iso-bytes): mean={wrong.q_gap.mean():.2f} median={wrong.q_gap.median():.2f} p90={wrong.q_gap.quantile(.9):.2f} max={wrong.q_gap.max():.2f}")
print("  by zq band (acc | misroute mean byte-oh | misroute mean q-gap):")
for lo, hi in [(45, 70), (70, 90), (90, 91)]:
    b = df[(df.zq >= lo) & (df.zq < hi)]; bw = b[b.oracle != b.pred]
    if len(b): print(f"    zq[{lo},{hi}): acc={1-len(bw)/len(b):.1%} | byte-oh={bw.byte_oh.mean()*100 if len(bw) else 0:.2f}% | q-gap={bw.q_gap.mean() if len(bw) else 0:.2f}")
print("  top oracle->pred confusions (count | median byte-oh | median q-gap):")
for (o, p), g in sorted(wrong.groupby(['oracle', 'pred']), key=lambda kv: -len(kv[1]))[:6]:
    print(f"    {o:>4}->{p:<4} n={len(g):5d} | byte-oh={g.byte_oh.median()*100:5.2f}% | q-gap={g.q_gap.median():.2f}")

# ---- Part 2: the zq97 dead-zone ----
print("\n=== zq95-98 DEAD-ZONE (lossy reachability, byte explosion, AVIF-vs-JXL, gate) ===")
reach = collections.Counter(); explode = []; avif_jxl = []; gate_ratio = []
for v in rd:
    if v not in vix: continue
    curves = rd[v]
    # reachability at 97: which lossy families' ceiling >= 97
    can97 = [f for f in curves if zq_ceiling(curves[f]) >= 97.0]
    reach['any' if can97 else 'none'] += 1
    # byte explosion best-lossy bytes(97)/bytes(90)
    b90 = {f: bytes_at(curves[f], 90) for f in curves}; b90 = {f: x for f, x in b90.items() if x}
    b97 = {f: bytes_at(curves[f], 97) for f in curves}; b97 = {f: x for f, x in b97.items() if x}
    if b90 and b97: explode.append(min(b97.values()) / min(b90.values()))
    # AVIF vs JXL at 97 (both reachable)
    if 'avif' in b97 and 'jxl' in b97:
        avif_jxl.append((b97['avif'], b97['jxl']))
    # gate decision at 97 vs the lossless floor
    if v in mb and mb[v] and b97:
        llb = min(mb[v].values()); lyb = min(b97.values())
        gate_ratio.append(lyb / llb)  # <1 => lossy still cheaper (gate should stay lossy)
print(f"  lossy can reach zq97 on {reach['any']/(reach['any']+reach['none']):.1%} of variants ({reach['none']} can't -> forced lossless/extrapolated)")
ceils = collections.defaultdict(list)
for vv in rd:
    for f in rd[vv]: ceils[f].append(zq_ceiling(rd[vv][f]))
print("  per-family lossy zq CEILING (max achieved zensim across the swept grid):")
for f in ['jpeg', 'webp', 'jxl', 'avif']:
    cs = ceils[f]
    print(f"    {f:>4}: median={st.median(cs):.1f} p10={np.quantile(cs,.1):.1f} p90={np.quantile(cs,.9):.1f} | reaches>=95 on {sum(1 for x in cs if x>=95)/len(cs):.0%}  (CAVEAT: a ceiling reflects the per-codec sweep's quality range, not only the codec)")
print(f"  byte explosion best-lossy bytes(zq97)/bytes(zq90): median={st.median(explode):.2f}x  p90={np.quantile(explode,.9):.2f}x")
if avif_jxl:
    rels = [abs(a-j)/min(a,j) for a, j in avif_jxl]; avwin = sum(1 for a, j in avif_jxl if a < j)/len(avif_jxl)
    print(f"  AVIF vs JXL @zq97 (n={len(avif_jxl)} both-reach): AVIF cheaper {avwin:.0%} of the time; |gap|/min median={st.median(rels)*100:.1f}% (small => dead-zone)")
print(f"  gate @zq97: best-lossy/best-lossless median={st.median(gate_ratio):.2f} (>=1 => lossless wins; ~1 => coin-flip). lossy-cheaper on {sum(1 for r in gate_ratio if r<1)/len(gate_ratio):.0%}")
# AVIF-vs-JXL at zq95 (they co-reach far more often than at zq97)
av95 = [(bytes_at(rd[vv]['avif'], 95), bytes_at(rd[vv]['jxl'], 95)) for vv in rd
        if 'avif' in rd[vv] and 'jxl' in rd[vv] and bytes_at(rd[vv]['avif'], 95) and bytes_at(rd[vv]['jxl'], 95)]
if av95:
    rels = [abs(a-j)/min(a, j) for a, j in av95]
    print(f"  AVIF-vs-JXL @zq95 (n={len(av95)} both-reach): AVIF cheaper {sum(1 for a, j in av95 if a < j)/len(av95):.0%}; |gap|/min median={st.median(rels)*100:.1f}% (small => dead-zone)")
# the demo variant o_1016 is a TRAIN origin -> load train RD to explain its route_demo pick
rd_tr = lossy_rd('train'); mb_tr = ll_min('train'); v = 'o_1016.png.scale108x192'
if v in rd_tr:
    c = rd_tr[v]; b97 = {f: bytes_at(c[f], 97) for f in c if bytes_at(c[f], 97)}
    print(f"\n  demo {v}: ceilings={{{', '.join(f'{f}:{zq_ceiling(c[f]):.1f}' for f in c)}}}")
    print(f"    @zq97 lossy bytes={{{', '.join(f'{f}:{int(b)}' for f, b in b97.items())}}} | lossless={{{', '.join(f'{f}:{int(b)}' for f, b in mb_tr.get(v, {}).items())}}}")
    if b97 and mb_tr.get(v):
        bl, bll = min(b97.values()), min(mb_tr[v].values())
        print(f"    gate@97: best-lossy {int(bl)} vs best-lossless {int(bll)} -> {'LOSSY' if bl < bll else 'LOSSLESS'} (route_demo showed AVIF lossy)")
print("  NOTE: lossy router trained on zq45-90; zq97 is EXTRAPOLATION for it (the gate, trained to zq98, is the real zq97 decider).")
