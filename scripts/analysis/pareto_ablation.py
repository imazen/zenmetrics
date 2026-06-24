#!/usr/bin/env python3
# RD-full search-space reduction analysis (spend tokens, not credits):
#   (A) Pareto-ablation: which knob combos are NEVER on the (bytes, quality) Pareto front across the corpus
#       -> ablate them from the rd-full search. Also reports the RD-loss if we keep only the top-K knobs,
#       so we can pick K that recovers ~all of the front at a fraction of the cells.
#   (B) Knob-stability vs pixel count: does the front-knob set change with image size? If it stabilizes
#       below some pixel threshold, dense large-image sweeps are unnecessary.
# Runs on an existing dense rd-full parquet (e.g. jpeg v15r, 979 imgs x 1824 cells). No fleet, no credits.
#   usage: pareto_ablation.py <parquet> [metric=score_ssim2]
import sys, json, collections
import numpy as np, pyarrow.parquet as pq

P = sys.argv[1]
METRIC = sys.argv[2] if len(sys.argv) > 2 else "score_ssim2"
cols = ["image_basename", "knob_tuple_json", "q", "encoded_bytes", METRIC, "corpus_feat_pixel_count"]
t = pq.read_table(P, columns=cols).to_pandas().dropna(subset=["encoded_bytes", METRIC])
knobs = sorted(t.knob_tuple_json.unique())
kid = {k: i for i, k in enumerate(knobs)}
t["kid"] = t.knob_tuple_json.map(kid).astype(int)
NK = len(knobs)

def pareto_front_mask(b, s):
    # min bytes, max metric. sort by bytes asc, score desc; front = running-max of score
    order = np.lexsort((-s, b))
    bo, so = b[order], s[order]
    runmax = np.maximum.accumulate(so)
    keep = np.empty(len(so), bool); keep[0] = True
    keep[1:] = so[1:] > runmax[:-1] + 1e-9
    m = np.zeros(len(b), bool); m[order[keep]] = True
    return m

def pixbucket(px):
    mp = px / 1e6
    return ("a<=0.05" if mp <= 0.05 else "b0.05-0.15" if mp <= 0.15 else
            "c0.15-0.4" if mp <= 0.4 else "d0.4-1.0")

# per-image fronts
front_imgs = collections.Counter()          # kid -> #images where it's on the front
front_pts = collections.Counter()            # kid -> total #front cells (RD coverage proxy)
bucket_front = collections.defaultdict(collections.Counter)  # bucket -> kid -> #imgs on front
bucket_n = collections.Counter()
per_img_front_kids = {}                       # img -> set(kid) on front (for RD-loss reduced-set eval)
per_img_arr = {}                              # img -> (b, s, kid)
for img, g in t.groupby("image_basename", sort=False):
    b = g.encoded_bytes.to_numpy(float); s = g[METRIC].to_numpy(float); k = g.kid.to_numpy()
    m = pareto_front_mask(b, s)
    fk = k[m]
    per_img_front_kids[img] = set(fk.tolist())
    per_img_arr[img] = (b, s, k)
    for kk in set(fk.tolist()):
        front_imgs[kk] += 1
    for kk in fk.tolist():
        front_pts[kk] += 1
    bu = pixbucket(g.corpus_feat_pixel_count.iloc[0]); bucket_n[bu] += 1
    for kk in set(fk.tolist()):
        bucket_front[bu][kk] += 1

nimg = t.image_basename.nunique()
print(f"=== {P.split('/')[-1]}  metric={METRIC}  images={nimg}  knobs={NK} ===\n")

# (A) ablation
never = [i for i in range(NK) if front_imgs[i] == 0]
print(f"(A) PARETO-ABLATION")
print(f"  knobs NEVER on any image's front: {len(never)} / {NK}  -> directly ablatable")
freq = sorted(range(NK), key=lambda i: -front_imgs[i])
print(f"  front-frequency (top / bottom) — #images (of {nimg}) where the knob is on the front:")
for i in freq[:6]:
    print(f"    {front_imgs[i]:>4}img {front_pts[i]:>7}pts  {knobs[i][:110]}")
print("    ...")
for i in freq[-6:]:
    print(f"    {front_imgs[i]:>4}img {front_pts[i]:>7}pts  {knobs[i][:110]}")

# reduced-set RD-loss: keep top-K knobs (by #front-imgs), measure max metric-loss vs full front at matched bytes
def reduced_loss(topk_kids):
    keep = set(topk_kids); losses = []
    for img, (b, s, k) in per_img_arr.items():
        full_m = pareto_front_mask(b, s)
        fb, fs = b[full_m], s[full_m]
        sel = np.array([kk in keep for kk in k])
        if not sel.any(): losses.append(99.0); continue
        rb, rs = b[sel], s[sel]
        rm = pareto_front_mask(rb, rs); rb, rs = rb[rm], rs[rm]
        # for each full-front point, best reduced score at <= that bytes
        order = np.argsort(rb)
        rb2, rs2 = rb[order], np.maximum.accumulate(rs[order])
        idx = np.searchsorted(rb2, fb, side="right") - 1
        ok = idx >= 0
        loss = np.where(ok, fs - rs2[np.clip(idx, 0, len(rs2)-1)], fs - rs.min())
        losses.append(float(np.nanmax(loss)) if len(loss) else 0.0)
    a = np.array(losses)
    return a.mean(), np.percentile(a, 95), a.max()

print(f"\n  reduced-set RD-loss ({METRIC} units; how much quality we give up at matched bytes):")
for K in [NK, 48, 32, 24, 16, 12, 8, 6, 4]:
    if K > NK: continue
    topk = freq[:K]
    mean, p95, mx = reduced_loss(topk)
    print(f"    keep top-{K:>2} knobs ({K/NK*100:>4.0f}% of cells): mean {mean:5.2f}  p95 {p95:5.2f}  max {mx:5.2f}")

# (B) knob-stability vs pixel count
print(f"\n(B) KNOB-STABILITY vs PIXEL COUNT (does the front-knob set change with size?)")
for bu in sorted(bucket_front):
    bf = bucket_front[bu]; n = bucket_n[bu]
    topknobs = [i for i, _ in sorted(bf.items(), key=lambda x: -x[1])[:8]]
    print(f"  {bu[1:]:>9}MP  ({n:>3} imgs)  top-8 front knobs (by id): {sorted(topknobs)}")
# jaccard of top-K front-knob sets between adjacent buckets
buorder = sorted(bucket_front)
def topset(bu, K=16): return set(i for i, _ in sorted(bucket_front[bu].items(), key=lambda x: -x[1])[:K])
print("  top-16 front-knob-set Jaccard between adjacent size buckets (1.0 = identical -> stable):")
for a, c in zip(buorder, buorder[1:]):
    sa, sc = topset(a), topset(c)
    j = len(sa & sc) / max(1, len(sa | sc))
    print(f"    {a[1:]:>9} vs {c[1:]:>9}: {j:.2f}")

# (C) per-axis decode: which axis VALUES never make the front -> always-set / always-avoid that axis
print("\n(C) PER-AXIS Pareto-usefulness (front-image-count summed over the knobs holding each value):")
axval = collections.defaultdict(collections.Counter)
for i, k in enumerate(knobs):
    for ax, v in json.loads(k).items():
        if ax in ("fp", "cell", "plan"):
            continue
        axval[ax][str(v)] += front_imgs[i]
for ax in sorted(axval):
    vs = axval[ax]
    line = "   ".join(f"{v}={c}" for v, c in sorted(vs.items(), key=lambda x: -x[1]))
    dead = [v for v, c in vs.items() if c == 0]
    tag = f"   -> ABLATE value(s) {dead} (never on any front)" if dead else ""
    print(f"  {ax:>22}: {line}{tag}")
