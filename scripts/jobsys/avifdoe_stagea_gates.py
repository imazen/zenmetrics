#!/usr/bin/env python3
"""avifdoe_stagea_gates.py — the AG transfer gate, the bytes model, and the Stage-B triggers.

Companion to avifdoe_stagea_analyze.py. Everything here is pre-registered in
benchmarks/avif_doe_plan_2026-09-01.md §3.8 (T1/T2/T3), §3.9 (alpha + beta*pixels)
and §7.2 (B-1..B-6), with the §12.4 degeneracy correction applied.

THREE REGISTERED-VS-OBSERVED DEVIATIONS ARE ENFORCED HERE, NOT PAPERED OVER:

 1. §12.4 — 19 of the 32 references are NATIVE PASSTHROUGHS (crop_sha == source_sha),
    so for them "budget" and "native" are the SAME ENCODE OF THE SAME PIXELS.
    T1/T2/T3 are therefore computed over the 13 CROPPED references only, with the
    bars restated against n=13, and the 19 identity pairs reported separately as
    the null check they actually are (a non-zero residual there is a pipeline bug,
    not a transfer effect).

 2. §3.9's bytes decomposition inherits the SAME degeneracy — the 19 passthroughs
    contribute two IDENTICAL pixel counts and thus no leverage on the intercept.
    alpha + beta*pixels is identifiable on 13 references, and per-image it is an
    exactly-determined 2-point fit (2 unknowns, 2 equations) with NO residual and
    therefore no goodness-of-fit. Stated, not hidden.

 3. The declared AG grid carries knob arms at SPEED 4 ONLY (plus a bare s6
    control), not the {4,6} §3.8 registered, so the gate runs at speed 4 only.
    BD-rate over AG's 3 q points {15,45,90} is a 3-point trapezoid; the usual
    >=4-point guard would make the plan's own gate uncomputable, so the minimum
    is 3 HERE ONLY and the budget side is restricted to the SAME 3 q points.
"""
import argparse, collections, csv, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/work/zen/zensim/scripts"))

def frontier(points):
    bybpp = sorted(points, key=lambda p: (p[1], -p[0]))
    front, best = [], -1e18
    for s, b in bybpp:
        if s > best:
            front.append((s, b)); best = s
    front.sort(key=lambda p: p[0])
    return front

def bd_rate(test, ref, min_pts=4):
    def prep(f):
        seen = {}
        for s, b in f:
            seen[round(s, 6)] = np.log(b)
        xs = sorted(seen)
        return np.array(xs), np.array([seen[x] for x in xs])
    x1, y1 = prep(ref); x2, y2 = prep(test)
    if len(x1) < min_pts or len(x2) < min_pts:
        return None
    lo, hi = max(x1.min(), x2.min()), min(x1.max(), x2.max())
    if hi <= lo:
        return None
    gg = np.linspace(lo, hi, 200)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    avg = (trapz(np.interp(gg, x2, y2), gg) - trapz(np.interp(gg, x1, y1), gg)) / (hi - lo)
    return (np.exp(avg) - 1.0) * 100.0

def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial p for k successes of n. No scipy dependency.

    VERIFIED against scipy.stats.binomtest on 11 cases spanning the range this
    gate uses (0/13, 1/13, 3/13, 5/13, 6/13, 7/13, 13/13, 10/20, 5/10, 0/1, 1/2):
    max |difference| = 0.0 exactly (2026-09-02). The method is the standard
    "sum every outcome no more likely than the observed one".
    """
    if n == 0: return float("nan")
    C = math.comb
    pmf = [C(n, i) * p**i * (1-p)**(n-i) for i in range(n+1)]
    obs = pmf[k]
    return min(1.0, sum(v for v in pmf if v <= obs * (1 + 1e-12)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--crop-manifest", required=True)
    ap.add_argument("--native-sizes", required=True, help="TSV image\\tspeed\\tq\\tbytes for the NATIVE control run")
    ap.add_argument("--native-dims", required=True,
                    help="TSV image\\twidth\\theight read from the NATIVE PNGs. REQUIRED and separate "
                         "from the crop manifest: that manifest's width/height are the CROP dims "
                         "(1024x1024 for every cropped ref), so using it gives pn == pb and silently "
                         "fits nothing at all.")
    ap.add_argument("--main-effects", required=True, help="main_effects.tsv from the analyzer")
    ap.add_argument("--interactions", required=True, help="interactions.tsv from the analyzer")
    ap.add_argument("--bd-per-image", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    man = {}
    for r in csv.DictReader(open(a.crop_manifest), delimiter="\t"):
        man[r["corpus_key"]] = r
    cropped = sorted(k for k, v in man.items() if v["transform"] != "native")
    passthru = sorted(k for k, v in man.items() if v["transform"] == "native")
    px_budget = {k: int(v["width"]) * int(v["height"]) for k, v in man.items()}   # CROP dims
    px_native = {}
    for r in csv.DictReader(open(a.native_dims), delimiter="\t"):
        px_native[r["image"]] = int(r["width"]) * int(r["height"])
    missing = [k for k in man if k not in px_native]
    if missing:
        raise SystemExit(f"native dims missing for {len(missing)} references: {missing[:5]}")
    print(f"references: {len(man)}  cropped: {len(cropped)}  passthrough: {len(passthru)}")

    import pyarrow.parquet as pq
    t = pq.read_table(a.scored).to_pydict()
    n = len(t["run"])
    AGQ = {15, 45, 90}

    # curves[(run,speed,image,arm)] = [(qual, bytes)]
    cur = collections.defaultdict(list)
    cur3 = collections.defaultdict(list)      # restricted to AG's 3 q points
    bytes_budget = {}                          # (image,speed,q) -> bytes  (A0R control)
    for i in range(n):
        b, s_ = t["bytes"][i], t["m_ssim2"][i]
        run, sp, img, arm, q = t["run"][i], t["speed"][i], t["image"][i], t["arm"][i], t["q"][i]
        if run == "a0r" and arm == f"s{sp}-svt-420" and b is not None:
            bytes_budget[(img, sp, q)] = b
        if b is None or s_ is None:
            continue
        cur[(run, sp, img, arm)].append((s_, b))
        if q in AGQ:
            cur3[(run, sp, img, arm)].append((s_, b))

    # ---------------- AG TRANSFER GATE (§3.8 + §12.4 correction) -------------
    # n floors: the §12.4 restatement gives at most 13 cropped refs, and the
    # 3-q ladder loses more to non-monotone frontiers. Below these the test is
    # not evaluable and says so.
    MIN_CROP, MIN_EFF = 5, 3
    knobs = sorted({t["arm"][i].split("-", 3)[3] for i in range(n)
                    if t["run"][i] == "ag" and t["arm"][i].count("-") >= 3})
    gate_rows, ident_rows = [], []
    for k in knobs:
        pairs = []                        # (image, bd_native, bd_budget)
        for img in sorted(man):
            nat = cur3.get(("ag", 4, img, f"s4-svt-420-{k}"))
            natc = cur3.get(("ag", 4, img, "s4-svt-420"))
            bud = cur3.get(("a1", 4, img, f"s4-svt-420-{k}"))
            budc = cur3.get(("a1", 4, img, "s4-svt-420"))
            if not (nat and natc and bud and budc):
                continue
            bn = bd_rate(frontier(nat), frontier(natc), min_pts=3)
            bb = bd_rate(frontier(bud), frontier(budc), min_pts=3)
            if bn is None or bb is None:
                continue
            pairs.append((img, bn, bb))
        crop_p = [p for p in pairs if p[0] in cropped]
        pass_p = [p for p in pairs if p[0] in passthru]
        # null check: on a passthrough the two legs are the SAME encode
        for img, bn, bb in pass_p:
            if abs(bn - bb) > 1e-9:
                ident_rows.append(dict(knob=k, image=img, bd_native=bn, bd_budget=bb, resid=bb - bn))
        # An arm that is byte-identical to the control has no effect to transfer.
        # Reporting it as "FAIL-T1" would be a category error: |BD| >= 0.5% never
        # holds, so T1 has an empty denominator. Same precedent as the zensim band
        # rule -- an unusable cell is NOT-MEASURED with a reason, never a verdict.
        if all(abs(bn) == 0.0 and abs(bb) == 0.0 for _, bn, bb in pairs) and pairs:
            gate_rows.append(dict(knob=k, n_crop=len(crop_p), n_eff=0, T1=None, T2=None,
                                  T3_med=None, T3_p=None, n_pass_identity=len(pass_p),
                                  verdict="INERT (byte-identical to control; nothing to transfer)"))
            continue
        if len(crop_p) < MIN_CROP:
            gate_rows.append(dict(knob=k, n_crop=len(crop_p), n_eff=None, T1=None, T2=None,
                                  T3_med=None, T3_p=None, n_pass_identity=len(pass_p),
                                  verdict=f"NOT-MEASURED (n={len(crop_p)} usable cropped refs, need {MIN_CROP})"))
            continue
        # T1 direction, counting only |bd_native| >= 0.5
        eff = [(i, bn, bb) for i, bn, bb in crop_p if abs(bn) >= 0.5]
        t1_agree = sum(1 for _, bn, bb in eff if (bn > 0) == (bb > 0))
        t1 = (t1_agree / len(eff)) if eff else None
        # T2 rank: SROCC via the canonical owner
        try:
            from lib.zen_stats import srocc as _srocc
            t2 = float(_srocc([p[2] for p in crop_p], [p[1] for p in crop_p]))
            t2src = "zenstats"
        except Exception as e:
            t2, t2src = None, f"UNAVAILABLE ({e})"
        # T3 magnitude
        resid = [p[2] - p[1] for p in crop_p]
        t3med = float(np.median(np.abs(resid)))
        npos = sum(1 for r in resid if r > 0)
        t3p = binom_two_sided(npos, len(resid))
        # A T1 computed on fewer than MIN_EFF images is not a direction test --
        # "sign agreement on 1 of 1" carries no information. Refuse the verdict
        # rather than issue one the n cannot support. The BARS ARE UNCHANGED.
        okT1 = (t1 is not None and len(eff) >= MIN_EFF and t1 >= 0.80)
        okT2 = (t2 is not None and t2 >= 0.70)
        okT3 = (t3med <= 1.0 and t3p >= 0.05)
        if t1 is None or len(eff) < MIN_EFF:
            verdict = f"NOT-MEASURED (only {len(eff)} of {len(crop_p)} cropped refs move >=0.5% at native, need {MIN_EFF})"
        elif okT1 and okT2 and okT3:
            verdict = "PASS"
        elif not okT1:
            verdict = "FAIL-T1 (not screenable at budget)"
        else:
            verdict = "PARTIAL (direction holds; magnitude/rank flagged)"
        gate_rows.append(dict(knob=k, n_crop=len(crop_p), n_eff=len(eff),
                              T1=t1, T2=t2, T2_src=t2src, T3_med=t3med, T3_p=t3p,
                              n_pass_identity=len(pass_p), verdict=verdict))
    with open(f"{a.outdir}/ag_transfer_gate.tsv", "w") as f:
        f.write("knob\tn_cropped\tn_effective\tT1_sign_agree\tT2_srocc\tT3_median_abs_resid\tT3_binom_p\tn_passthrough_identity\tverdict\n")
        for r in gate_rows:
            f.write(f"{r['knob']}\t{r.get('n_crop')}\t{r.get('n_eff','')}\t"
                    f"{'' if r.get('T1') is None else '%.4f'%r['T1']}\t"
                    f"{'' if r.get('T2') is None else '%.4f'%r['T2']}\t"
                    f"{'' if r.get('T3_med') is None else '%.4f'%r['T3_med']}\t"
                    f"{'' if r.get('T3_p') is None else '%.4g'%r['T3_p']}\t"
                    f"{r.get('n_pass_identity','')}\t{r['verdict']}\n")
    with open(f"{a.outdir}/ag_identity_violations.tsv", "w") as f:
        f.write("knob\timage\tbd_native\tbd_budget\tresid\n")
        for r in ident_rows:
            f.write(f"{r['knob']}\t{r['image']}\t{r['bd_native']:.6f}\t{r['bd_budget']:.6f}\t{r['resid']:.6g}\n")
    print(f"AG gate: {len(gate_rows)} knobs; identity violations: {len(ident_rows)}")

    # ---------------- BYTES alpha + beta*pixels (§3.9 + the n=13 caveat) -----
    nat_bytes = {}
    for r in csv.DictReader(open(a.native_sizes), delimiter="\t"):
        nat_bytes[(r["image"], int(r["speed"]), int(r["q"]))] = int(r["bytes"])
    fits, identity_ok, identity_bad = [], 0, 0
    for (img, sp, q), bb in sorted(bytes_budget.items()):
        nb = nat_bytes.get((img, sp, q))
        if nb is None: continue
        if img in passthru:
            # SAME pixels, so the two runs must agree byte-for-byte if their
            # configurations match. This is a free config-identity gate.
            (identity_ok, identity_bad) = (identity_ok + 1, identity_bad) if nb == bb else (identity_ok, identity_bad + 1)
            continue
        pn, pb = px_native[img], px_budget[img]
        if pn == pb:
            continue   # no leverage: identical pixel counts cannot identify alpha
        beta = (nb - bb) / (pn - pb)
        alpha = bb - beta * pb
        fits.append(dict(image=img, speed=sp, q=q, bytes_native=nb, bytes_budget=bb,
                         px_native=pn, px_budget=pb, alpha=alpha, beta=beta))
    with open(f"{a.outdir}/bytes_alpha_beta.tsv", "w") as f:
        f.write("image\tspeed\tq\tbytes_native\tbytes_budget\tpx_native\tpx_budget\talpha_bytes\tbeta_bytes_per_px\tbeta_bpp\n")
        for r in fits:
            f.write(f"{r['image']}\t{r['speed']}\t{r['q']}\t{r['bytes_native']}\t{r['bytes_budget']}"
                    f"\t{r['px_native']}\t{r['px_budget']}\t{r['alpha']:.2f}\t{r['beta']:.8f}\t{r['beta']*8:.6f}\n")
    print(f"bytes fits (cropped refs only): {len(fits)}; "
          f"passthrough config-identity: {identity_ok} identical / {identity_bad} DIFFER")
    json.dump(dict(passthrough_identical=identity_ok, passthrough_differ=identity_bad,
                   n_fits=len(fits), n_cropped=len(cropped), n_passthrough=len(passthru)),
              open(f"{a.outdir}/bytes_model_meta.json", "w"), indent=1)

    # ---------------- STAGE-B TRIGGERS (§7.2) --------------------------------
    main = list(csv.DictReader(open(a.main_effects), delimiter="\t"))
    inter = list(csv.DictReader(open(a.interactions), delimiter="\t"))
    trig = []
    # Structurally inert arms (byte-identical to the control on every cell) are
    # not measurements and must not consume Stage-B budget. They are reported in
    # their own section of the Stage-A record, not as triggers.
    inert = set()
    for r in csv.DictReader(open(os.path.join(os.path.dirname(a.main_effects), "arm_byte_identity.tsv")), delimiter="\t"):
        devs = [d for d in r["arm"].split("-")[3:] if d]
        if len(devs) == 1 and float(r["frac"]) == 1.0:
            inert.add(devs[0])
    if inert:
        print("structurally INERT knobs (byte-identical to control on 100% of cells):", sorted(inert))
    for r in main:                                    # B-1
        if r["knob"] in inert:
            continue
        med, iqr = abs(float(r["median_bd"])), float(r["iqr"])
        if med >= 1.5 or iqr >= 3.0:
            trig.append(dict(id="B-1", key=f"{r['knob']}@s{r['speed']}",
                             why=f"|median| {med:.2f}% (bar 1.5) / IQR {iqr:.2f}% (bar 3.0)",
                             follow_up="dense grid: 5 levels x 29-q x 32 img x speeds {4,6,7}"))
    for r in inter:                                   # B-2
        if r["k1"] in inert or r["k2"] in inert:
            continue
        fr = float(r["frac_images_ge_1pct"])
        if fr >= 0.25:
            trig.append(dict(id="B-2", key=f"({r['k1']},{r['k2']})@s{r['speed']}",
                             why=f"|resid| >= 1% on {fr*100:.1f}% of images (bar 25%), median resid {float(r['median_resid']):+.2f}%",
                             follow_up="full k1 x k2 grid, 3 levels each, 9-q, 32 img"))
    bycls = collections.defaultdict(dict)             # B-3
    for r in csv.DictReader(open(os.path.join(os.path.dirname(a.main_effects), "main_effects_by_class.tsv")), delimiter="\t"):
        bycls[(r["speed"], r["knob"])][r["class"]] = (float(r["median_bd"]), int(r["n"]))
    # §7.2 B-3 does not name a minimum class size, and this corpus has content
    # classes as small as n=1 (12 fine classes over 32 refs). A "median" of one
    # image is a single observation, so every firing carries its contributing
    # class sizes and any trigger with a class of n<3 is marked PROVISIONAL —
    # reported, never silently promoted or silently dropped.
    MIN_CLASS_N = 3
    for (sp, k), d in sorted(bycls.items()):
        strong = {c: (v, nn) for c, (v, nn) in d.items() if abs(v) >= 1.0}
        if len({v > 0 for v, _ in strong.values()}) == 2:
            pos = [f"{c} {v:+.2f}% (n={nn})" for c, (v, nn) in strong.items() if v > 0]
            neg = [f"{c} {v:+.2f}% (n={nn})" for c, (v, nn) in strong.items() if v < 0]
            small = [c for c, (v, nn) in strong.items() if nn < MIN_CLASS_N]
            tag = f" PROVISIONAL (class n<{MIN_CLASS_N}: {','.join(small)})" if small else ""
            trig.append(dict(id="B-3", key=f"{k}@s{sp}",
                             why=f"opposite-sign class medians >=1%: [{', '.join(neg)}] vs [{', '.join(pos)}]{tag}",
                             follow_up="content-stratified dense follow-up + explicit interaction term"))
    bysp = collections.defaultdict(dict)              # B-5
    for r in main:
        bysp[r["knob"]][int(r["speed"])] = float(r["median_bd"])
    for k, d in sorted(bysp.items()):
        if k in inert:
            continue
        if len(d) >= 2:
            v = [x for x in d.values() if abs(x) >= 1.0]
            if len(v) >= 2 and len({x > 0 for x in v}) == 2:
                trig.append(dict(id="B-5", key=k,
                                 why="median BD-rate inverts sign across presets: " +
                                     ", ".join(f"s{s} {x:+.2f}%" for s, x in sorted(d.items())),
                                 follow_up="fit with an explicit preset interaction; Stage-B grid at both presets"))
    for r in gate_rows:                               # B-6
        if r["knob"] in inert:
            continue
        if str(r["verdict"]).startswith("FAIL-T1"):
            trig.append(dict(id="B-6", key=r["knob"],
                             why=f"fails T1 (sign agreement {r.get('T1')}) — not screenable at reduced size",
                             follow_up="Stage-B grid at NATIVE size; A1/A2 numbers annotated size-conditional"))
    # ---- price every follow-up against §7.2's registered envelope -----------
    # B-1: 5 levels x 29 q x 32 img x speeds {4,6,7}          = 13,920 cells / knob
    # B-2: 3 levels each (3x3=9 combos) x 9 q x 32 img        =  2,592 cells / pair
    # B-3: content-stratified dense follow-up. §7.2 does not fix a grid for it;
    #      priced as B-1's grid restricted to the triggering classes' images,
    #      which is the cheapest reading of "content-stratified dense".
    # B-5: "Stage-B grid at both inverting presets" = B-1's grid, 2 presets not 3.
    # B-6: B-1's grid at native size.
    COST = {"B-1": 5 * 29 * 32 * 3, "B-2": 9 * 9 * 32, "B-5": 5 * 29 * 32 * 2, "B-6": 5 * 29 * 32 * 3}
    n_img_by_class = collections.Counter()
    for r in csv.DictReader(open(os.path.join(os.path.dirname(a.main_effects), "main_effects_by_class.tsv")), delimiter="\t"):
        n_img_by_class[r["class"]] = max(n_img_by_class[r["class"]], int(r["n"]))
    for r in trig:
        if r["id"] == "B-3":
            imgs = sum(n for c, n in n_img_by_class.items() if c in r["why"])
            r["cells"] = 5 * 29 * max(imgs, 1) * 3
        else:
            r["cells"] = COST.get(r["id"], 0)
    with open(f"{a.outdir}/stage_b_triggers.tsv", "w") as f:
        f.write("trigger\tkey\tcells\twhy\tfollow_up\n")
        for r in sorted(trig, key=lambda x: (x["id"], x["key"])):
            f.write(f"{r['id']}\t{r['key']}\t{r['cells']}\t{r['why']}\t{r['follow_up']}\n")
    by = collections.Counter(); cells = collections.Counter()
    for r in trig:
        by[r["id"]] += 1; cells[r["id"]] += r["cells"]
    total = sum(cells.values())
    ENVELOPE = 60000
    print("Stage-B triggers:", dict(by))
    print("Stage-B cell cost by trigger:", dict(cells))
    print(f"Stage-B TOTAL if every trigger is honoured: {total:,} cells "
          f"vs the §7.2 envelope of {ENVELOPE:,} -> {total/ENVELOPE:.1f}x "
          f"({'OVER' if total > ENVELOPE else 'within'} budget)")
    json.dump(dict(triggers=dict(by), cells=dict(cells), total_cells=total,
                   envelope_cells=ENVELOPE, over_by=total / ENVELOPE),
              open(f"{a.outdir}/stage_b_budget.json", "w"), indent=1)

if __name__ == "__main__":
    main()
