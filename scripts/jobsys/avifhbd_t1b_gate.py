#!/usr/bin/env python3
"""t1b gate evaluation — reuses the OWNER's BD-rate (avifdoe_stagea_analyze)."""
import json, glob, sys, collections, statistics, random
sys.path.insert(0, '/home/lilith/work/zen/zenmetrics/scripts/jobsys')
from avifdoe_stagea_analyze import bd_rate, frontier   # the owner's algorithm
import pyarrow.parquet as pq

# ---- my T1 side: encode_sha -> ssim2, and encode_sha -> bytes
scores = {}
for f in glob.glob('/home/lilith/tmp/hbdexec/gate/sf/*'):
    for line in open(f):
        r = json.loads(line)
        if r.get('metric') == 'ssim2' and r.get('score') is not None:
            sha = r['encode_sha'].rsplit('/', 1)[-1]
            scores[sha] = r['score']
sizes = {}
for run in ('t1ac', 't1d'):
    for line in open(f'/home/lilith/tmp/hbdexec/gate/sizes/{run}.tsv'):
        k, v = line.split('\t'); sizes[k.strip()] = int(v)
print(f"my side: {len(scores)} ssim2 scores, {len(sizes)} blob sizes")

mine = collections.defaultdict(list)     # (speed, image, arm, size_class) -> [(bytes, ssim2)]
for run, cls in (('t1ac', 'budget'), ('t1d', 'native')):
    t = pq.read_table(f'/home/lilith/tmp/hbdexec/t1_pairs/avifdoe-svt-{run}-20260902_pairs.parquet').to_pydict()
    for i in range(len(t['encode_sha'])):
        sha = t['encode_sha'][i]
        if sha not in scores or sha not in sizes: continue
        cell = json.loads(t['knob_tuple_json'][i])['cell']
        sp = int(cell.split('-')[0][1:])
        mine[(sp, t['image_path'][i], cell, cls)].append((scores[sha], sizes[sha]))  # (quality, rate)

# ---- control side from the Stage-A scored dataset
sa = pq.read_table('/mnt/v/output/zensim-avifdoe/doe_scored_2026-09-02.parquet').to_pydict()
ctl = collections.defaultdict(list)      # (run, speed, image, arm) -> [(bytes, ssim2)]
for i in range(len(sa['run'])):
    if sa['m_ssim2'][i] is None: continue
    ctl[(sa['run'][i], sa['speed'][i], sa['image'][i], sa['arm'][i])].append((sa['m_ssim2'][i], sa['bytes'][i]))  # (quality, rate)

def boot(vals, n=10000, seed=20260902):
    rnd = random.Random(seed); k = len(vals)
    ms = sorted(statistics.mean(rnd.choices(vals, k=k)) for _ in range(n))
    return ms[int(.025*n)], ms[int(.975*n)]

def report(label, arm_key, ctl_key, size_cls):
    vals, imgs = [], []
    for (sp, img, cell, cls), pts in sorted(mine.items()):
        if cell != arm_key or cls != size_cls: continue
        ref = ctl.get((ctl_key[0], ctl_key[1], img, ctl_key[2]))
        if not ref: continue
        v = bd_rate(frontier(pts), frontier(ref))
        if v is None: continue
        vals.append(v); imgs.append(img)
    if not vals:
        print(f"\n{label}: NO COMPARABLE CELLS"); return None
    wins = sum(1 for v in vals if v < 0)
    lo, hi = boot(vals)
    print(f"\n{label}")
    print(f"  arm={arm_key} vs control={ctl_key[0]}/{ctl_key[2]} ({size_cls})")
    print(f"  n_images={len(vals)}  BD-rate mean={statistics.mean(vals):+.3f}%  median={statistics.median(vals):+.3f}%")
    print(f"  95% CI [{lo:+.3f}%, {hi:+.3f}%]   wins(<0)={wins}/{len(vals)}   worst={max(vals):+.2f}%  best={min(vals):+.2f}%")
    return dict(n=len(vals), mean=statistics.mean(vals), median=statistics.median(vals), ci=(lo,hi), wins=wins)

print("\n" + "="*72)
print("CHAIN VALIDATION — reproduce A1's published bd10@s4 = -1.02% (CI -1.23,-0.36)")
print("="*72)
r_s4 = report("A1 REPRO: bd10 @ s4 (budget)", "s4-svt-420-bd10", ("a0r", 4, "s4-svt-420"), "budget")
print("\n" + "="*72)
print("THE GATE")
print("="*72)
r_s6 = report("T1-c (Q1): bd10 @ s6 (budget)", "s6-svt-420-bd10", ("a0r", 6, "s6-svt-420"), "budget")
r_nat = report("T1-d (Q4): bd10 @ s4 NATIVE", "s4-svt-420-bd10", ("ag", 4, "s4-svt-420"), "native")

# ---- T1-d: AG's native control has only 3 q points, and the OWNER's bd_rate
# hard-requires >=4 on both sides. So Q4 is answered by a MATCHED-q comparison,
# which is a different (weaker) instrument and is labelled as such.
print("\n" + "="*72)
print("T1-d (Q4) — MATCHED-q, NOT BD-rate (AG's native control is a 3-point ladder;")
print("            the registered bd_rate needs >=4 points on both sides)")
print("="*72)
rows = []
for (sp, img, cell, cls), pts in sorted(mine.items()):
    if cell != "s4-svt-420-bd10" or cls != "native": continue
    ref = ctl.get(("ag", 4, img, "s4-svt-420"))
    if not ref: continue
    # pts/ref are (quality, rate); rebuild per-q by matching on the q ladder
    rows.append((img, sorted(pts), sorted(ref)))
print(f"  images with both arms: {len(rows)}")
# per-q join via the pairs table (q is authoritative there)
import pyarrow.parquet as _pq
td = _pq.read_table('/home/lilith/tmp/hbdexec/t1_pairs/avifdoe-svt-t1d-20260902_pairs.parquet').to_pydict()
mine_q = {}
for i in range(len(td['encode_sha'])):
    sha = td['encode_sha'][i]
    if sha in scores and sha in sizes:
        mine_q[(td['image_path'][i], td['q'][i])] = (sizes[sha], scores[sha])
ctl_q = {}
for i in range(len(sa['run'])):
    if sa['run'][i] == 'ag' and sa['arm'][i] == 's4-svt-420' and sa['m_ssim2'][i] is not None:
        ctl_q[(sa['image'][i], sa['q'][i])] = (sa['bytes'][i], sa['m_ssim2'][i])
db, dq = [], []
for k, (b10, s10) in sorted(mine_q.items()):
    if k not in ctl_q: continue
    b8, s8 = ctl_q[k]
    db.append((b10 - b8) / b8 * 100.0); dq.append(s10 - s8)
if db:
    print(f"  matched (image,q) cells: {len(db)}")
    print(f"  bytes   delta: mean {statistics.mean(db):+.3f}%  median {statistics.median(db):+.3f}%  (negative = bd10 smaller)")
    print(f"  ssim2   delta: mean {statistics.mean(dq):+.4f}   median {statistics.median(dq):+.4f}  (positive = bd10 better)")
    print(f"  cells where bd10 is BOTH smaller and better: {sum(1 for i in range(len(db)) if db[i]<0 and dq[i]>0)}/{len(db)}")
    lo, hi = boot(dq); print(f"  ssim2 delta 95% CI [{lo:+.4f}, {hi:+.4f}]")
    lob, hib = boot(db); print(f"  bytes delta 95% CI [{lob:+.3f}%, {hib:+.3f}%]")
