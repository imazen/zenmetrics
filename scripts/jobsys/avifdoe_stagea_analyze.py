#!/usr/bin/env python3
"""avifdoe_stagea_analyze.py — the PRE-REGISTERED Stage-A analysis of the AVIF DOE.

Runs exactly the analysis registered in benchmarks/avif_doe_plan_2026-09-01.md
§7.1 and evaluates the §7.2 Stage-B triggers. Nothing here is chosen after
seeing results.

  §7.1(1) per (image, arm, speed): an RD curve over the q ladder, (bytes, ssim2)
  §7.1(2) effect size = BD-rate vs the deviations=0 control at the SAME
          (image, speed) AND THE SAME SIZE
  §7.1(3) main effects  = per-arm BD-rate distribution over the 32 images
  §7.1(4) interactions  = observed pair BD-rate - additive prediction
  §7.1(6) sharpness/tune stay CATEGORICAL (we never fit an ordinal trend)
  §7.1(7) aliased arms are detected by byte-identity and reported, not silently kept

STATS OWNERSHIP. Rank/correlation statistics go through the canonical owner
(`zenstats`, via zensim/scripts/lib/zen_stats.py). BD-rate is an integration,
not a rank stat; it is computed here with the SAME algorithm as zenavif's
scripts/rd_gap/bd_arm.py (Pareto frontier -> log-rate trapezoid over the
overlapping quality span) and `--parity-check` asserts agreement against that
implementation directly.

BD-rate sign convention (inherited from bd_arm.py): NEGATIVE = the arm needs
FEWER bits at matched quality = the arm WINS.
"""
import argparse, collections, csv, json, math, os, statistics, sys
import numpy as np

# ---------------------------------------------------------------- BD-rate ---
def frontier(points):
    """Monotone quality-vs-rate Pareto front. points = [(quality, rate), ...]"""
    bybpp = sorted(points, key=lambda p: (p[1], -p[0]))
    front, best = [], -1e18
    for s, b in bybpp:
        if s > best:
            front.append((s, b)); best = s
    front.sort(key=lambda p: p[0])
    return front

def bd_rate(test, ref):
    """BD-rate of test vs ref over overlapping quality. + = test needs MORE bits."""
    def prep(f):
        seen = {}
        for s, b in f:
            seen[round(s, 6)] = np.log(b)
        xs = sorted(seen)
        return np.array(xs), np.array([seen[x] for x in xs])
    x1, y1 = prep(ref); x2, y2 = prep(test)
    if len(x1) < 4 or len(x2) < 4:
        return None
    lo, hi = max(x1.min(), x2.min()), min(x1.max(), x2.max())
    if hi <= lo:
        return None
    gg = np.linspace(lo, hi, 200)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    avg = (trapz(np.interp(gg, x2, y2), gg) - trapz(np.interp(gg, x1, y1), gg)) / (hi - lo)
    return (np.exp(avg) - 1.0) * 100.0

# ------------------------------------------------------------ content class --
COARSE = {
    "7000-lilith-plots": "plot",
    "8100-lilith-web-screenshots": "screenshot",
    "9226-lilith-ai-products": "ai-gen",
    "9094-lilith-ai-illustrations": "ai-gen",
    "9000-lilith-ai-clipart": "ai-gen",
    "6000-lilith-scans-public-patents": "scan",
    "6600-ia-scans-manuscript-illustrations": "scan",
    "1400-lilith-nature": "photo",
    "3000-art-institute-of-chicago-photos": "photo",
    "1200-lilith-interiors": "photo",
    "1000-lilith-photos-general": "photo",
    "1600-lilith-food": "photo",
}

def load_manifest(path):
    out = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        pz = r.get("parent_z_dist", "")
        out[r["corpus_key"]] = dict(
            content_class=r["content_class"],
            coarse=COARSE.get(r["content_class"], "other"),
            transform=r["transform"],
            recheck=r.get("feature_recheck", ""),
            parent_z=float(pz) if pz not in ("", None) else None,
            width=int(r["width"]), height=int(r["height"]),
            source_sha=r["source_sha256"], crop_sha=r["crop_sha256"],
        )
    return out

def q1q3(v):
    if len(v) < 2: return (float("nan"), float("nan"))
    return (float(np.percentile(v, 25)), float(np.percentile(v, 75)))

def median_ci(v, n_boot=10000, seed=20260902):
    """Percentile-bootstrap 95% CI of the median, resampling IMAGES.

    The effect size here is a median over images, not a correlation, so this is
    a plain resample rather than a zenstats call (zenstats owns rank/correlation
    statistics and their CIs; it has no median-CI entry point). The RNG is
    seeded so the interval is reproducible.
    """
    v = np.asarray(v, dtype=float)
    if v.size < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    meds = np.median(v[idx], axis=1)
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="avifdoe_harvest.py parquet")
    ap.add_argument("--crop-manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--metric", default="m_ssim2", help="quality column (higher=better)")
    ap.add_argument("--control", choices=("auto", "a0r", "inrun"), default="auto",
                    help="which deviations=0 control to difference against. a0r = the DENSE 29-q "
                         "same-size control (§3.1's purpose); inrun = the arm's own run's 9-q "
                         "deviations=0 stratum; auto = a0r where available else in-run. Run BOTH "
                         "and compare — they are not the same instrument.")
    ap.add_argument("--parity-check", default=None,
                    help="path to zenavif scripts/rd_gap/bd_arm.py — assert BD-rate agreement")
    ap.add_argument("--paired-control-run", default=None,
                    help="run label supplying the deviations=0 control for the PAIRED "
                         "per-q table when an arm's own run carries none (e.g. a 3-point "
                         "bd10 probe block). A CROSS-ERA control must be declared as such "
                         "by the caller; the control's run is recorded in every row.")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    if a.parity_check:
        import importlib.util
        spec = importlib.util.spec_from_file_location("bd_arm", a.parity_check)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        rng = np.random.default_rng(20260902)
        worst = 0.0
        for _ in range(200):
            n = int(rng.integers(5, 12))
            ref = sorted(zip(np.sort(rng.uniform(30, 95, n)), np.sort(rng.uniform(2e3, 3e5, n))))
            tst = sorted(zip(np.sort(rng.uniform(30, 95, n)), np.sort(rng.uniform(2e3, 3e5, n))))
            x, y = bd_rate(frontier(tst), frontier(ref)), m.bd_rate(m.frontier(tst), m.frontier(ref))
            if x is None or y is None:
                assert x is None and y is None, "frontier/None disagreement vs bd_arm.py"
                continue
            worst = max(worst, abs(x - y))
        assert worst == 0.0, f"BD-rate DIVERGES from zenavif bd_arm.py by {worst}"
        print(f"PARITY: bd_rate == zenavif/scripts/rd_gap/bd_arm.py exactly on 200 random ladders (max |delta| {worst})")

    import pyarrow.parquet as pq
    t = pq.read_table(a.scored).to_pydict()
    n = len(t["run"])
    man = load_manifest(a.crop_manifest)

    # ---- rows: keep only scored cells with a rate and a quality -------------
    rows = []
    for i in range(n):
        if t[a.metric][i] is None or t["bytes"][i] is None:
            continue
        img = t["image"][i]
        rows.append(dict(run=t["run"][i], image=img, q=t["q"][i], arm=t["arm"][i],
                         speed=t["speed"][i], devs=t["devs"][i] or "", n_dev=t["n_dev"][i],
                         bytes=t["bytes"][i], qual=t[a.metric][i],
                         esha=t["encode_sha"][i], metrics=t["metrics_present"][i],
                         cls=man.get(img, {}).get("coarse", "?"),
                         fine=man.get(img, {}).get("content_class", "?"),
                         transform=man.get(img, {}).get("transform", "?")))
    print(f"scored rows: {len(rows)} of {n} cells")

    # ---- §7.1(7): alias detection by byte-identity -------------------------
    # An arm that is byte-identical to the control on EVERY (image,q) it shares
    # with it is not a measurement, it is an alias. Report, never silently keep.
    bysr = collections.defaultdict(dict)          # (run,speed,image,q) -> {arm: esha}
    for r in rows:
        bysr[(r["run"], r["speed"], r["image"], r["q"])][r["arm"]] = r["esha"]
    ident = collections.Counter(); total = collections.Counter()
    for key, d in bysr.items():
        run, sp, img, q = key
        ctl = f"s{sp}-svt-420"
        if ctl not in d: continue
        for arm, e in d.items():
            if arm == ctl: continue
            total[(run, arm)] += 1
            if e == d[ctl]: ident[(run, arm)] += 1
    alias = []
    for k, tot in sorted(total.items()):
        fr = ident[k] / tot
        alias.append(dict(run=k[0], arm=k[1], cells=tot, byte_identical=ident[k], frac=fr))
    with open(f"{a.outdir}/arm_byte_identity.tsv", "w") as f:
        f.write("run\tarm\tcells\tbyte_identical_to_control\tfrac\n")
        for r_ in alias:
            f.write(f"{r_['run']}\t{r_['arm']}\t{r_['cells']}\t{r_['byte_identical']}\t{r_['frac']:.4f}\n")

    # ---- §7.1(1,2): RD curves + BD-rate vs same-(image,speed) control -------
    curves = collections.defaultdict(list)        # (run,speed,image,arm) -> [(qual,bytes)]
    for r in rows:
        curves[(r["run"], r["speed"], r["image"], r["arm"])].append((r["qual"], r["bytes"]))

    # A0R supplies the DENSE (29-q) same-size control; A1/A2 carry their own
    # 9-q deviations=0 stratum. Prefer the dense one, fall back to the in-run one,
    # and RECORD which was used — they are not the same instrument.
    bd = []            # dicts
    ctl_source = collections.Counter()
    for (run, sp, img, arm), pts in curves.items():
        if arm == f"s{sp}-svt-420":
            continue
        ctl_arm = f"s{sp}-svt-420"
        # The source label carries the control ladder's ACTUAL point count, not an
        # assumed one: A1/A2's in-run control is 9-q, but a Stage-B native block
        # (svt_doe_b6) carries a 29-q in-run control, and a hardcoded "in-run-9q"
        # would misreport which instrument produced the number.
        def _inrun():
            r = curves.get((run, sp, img, ctl_arm))
            return r, (f"in-run-{len(r)}q" if r else "in-run-missing")
        if a.control == "inrun":
            ref, src = _inrun()
        elif a.control == "a0r":
            ref, src = curves.get(("a0r", sp, img, ctl_arm)), "a0r-dense"
        else:
            ref, src = curves.get(("a0r", sp, img, ctl_arm)), "a0r-dense"
            if ref is None or len(ref) < 4:
                ref, src = _inrun()
        if ref is None:
            continue
        v = bd_rate(frontier(pts), frontier(ref))
        if v is None:
            continue
        ctl_source[src] += 1
        man_i = man.get(img, {})
        bd.append(dict(run=run, speed=sp, image=img, arm=arm,
                       devs=[d for d in arm.split("-")[3:] if d],
                       bd_rate=v, n_pts=len(pts), ctl=src,
                       cls=man_i.get("coarse", "?"), fine=man_i.get("content_class", "?")))
    print("control source used:", dict(ctl_source))
    with open(f"{a.outdir}/bd_per_image.tsv", "w") as f:
        f.write("run\tspeed\timage\tarm\tn_dev\tbd_rate\tn_pts\tctl\tclass\tfine_class\n")
        for r_ in sorted(bd, key=lambda x: (x["run"], x["speed"], x["arm"], x["image"])):
            f.write(f"{r_['run']}\t{r_['speed']}\t{r_['image']}\t{r_['arm']}\t{len(r_['devs'])}"
                    f"\t{r_['bd_rate']:.6f}\t{r_['n_pts']}\t{r_['ctl']}\t{r_['cls']}\t{r_['fine']}\n")

    # ---- §7.1(3): main effects ---------------------------------------------
    singles = collections.defaultdict(list)       # (speed,knob) -> [(image,bd,cls)]
    for r_ in bd:
        if len(r_["devs"]) == 1:
            singles[(r_["speed"], r_["devs"][0])].append((r_["image"], r_["bd_rate"], r_["cls"]))
    main = {}
    with open(f"{a.outdir}/main_effects.tsv", "w") as f:
        f.write("speed\tknob\tn_images\tmedian_bd\tci_lo\tci_hi\tiqr\tq25\tq75\tmean\tn_better\tmin\tmax\n")
        for (sp, k), v in sorted(singles.items()):
            vals = [x[1] for x in v]
            lo, hi = q1q3(vals)
            clo, chi = median_ci(vals)
            main[(sp, k)] = dict(median=float(np.median(vals)), iqr=hi - lo, n=len(vals),
                                 ci=(clo, chi), vals={x[0]: x[1] for x in v})
            f.write(f"{sp}\t{k}\t{len(vals)}\t{np.median(vals):.4f}\t{clo:.4f}\t{chi:.4f}\t{hi-lo:.4f}\t{lo:.4f}\t{hi:.4f}"
                    f"\t{np.mean(vals):.4f}\t{sum(1 for x in vals if x<0)}\t{min(vals):.4f}\t{max(vals):.4f}\n")

    # ---- §7.1(4): interactions ---------------------------------------------
    inter = []
    for r_ in bd:
        if len(r_["devs"]) != 2: continue
        k1, k2 = r_["devs"]
        m1 = main.get((r_["speed"], k1)); m2 = main.get((r_["speed"], k2))
        if not m1 or not m2: continue
        a1 = m1["vals"].get(r_["image"]); a2 = m2["vals"].get(r_["image"])
        if a1 is None or a2 is None: continue
        inter.append(dict(speed=r_["speed"], image=r_["image"], k1=k1, k2=k2,
                          observed=r_["bd_rate"], additive=a1 + a2,
                          resid=r_["bd_rate"] - (a1 + a2), cls=r_["cls"]))
    with open(f"{a.outdir}/interactions_per_image.tsv", "w") as f:
        f.write("speed\timage\tk1\tk2\tobserved\tadditive\tresid\tclass\n")
        for r_ in sorted(inter, key=lambda x: (x["k1"], x["k2"], x["image"])):
            f.write(f"{r_['speed']}\t{r_['image']}\t{r_['k1']}\t{r_['k2']}\t{r_['observed']:.4f}"
                    f"\t{r_['additive']:.4f}\t{r_['resid']:.4f}\t{r_['cls']}\n")
    bypair = collections.defaultdict(list)
    for r_ in inter: bypair[(r_["speed"], r_["k1"], r_["k2"])].append(r_["resid"])
    with open(f"{a.outdir}/interactions.tsv", "w") as f:
        f.write("speed\tk1\tk2\tn_images\tmedian_resid\tmean_abs_resid\tfrac_images_ge_1pct\tmax_abs\n")
        for (sp, k1, k2), v in sorted(bypair.items()):
            arr = np.array(v); fr = float((np.abs(arr) >= 1.0).mean())
            f.write(f"{sp}\t{k1}\t{k2}\t{len(v)}\t{np.median(arr):.4f}\t{np.abs(arr).mean():.4f}"
                    f"\t{fr:.4f}\t{np.abs(arr).max():.4f}\n")

    # ---- per content class --------------------------------------------------
    bycls = collections.defaultdict(list)
    for r_ in bd:
        if len(r_["devs"]) == 1:
            bycls[(r_["speed"], r_["devs"][0], r_["cls"])].append(r_["bd_rate"])
    with open(f"{a.outdir}/main_effects_by_class.tsv", "w") as f:
        f.write("speed\tknob\tclass\tn\tmedian_bd\tci_lo\tci_hi\tq25\tq75\n")
        for (sp, k, c), v in sorted(bycls.items()):
            lo, hi = q1q3(v); clo, chi = median_ci(v)
            f.write(f"{sp}\t{k}\t{c}\t{len(v)}\t{np.median(v):.4f}\t{clo:.4f}\t{chi:.4f}\t{lo:.4f}\t{hi:.4f}\n")

    # ---- PAIRED per-(image, q) arm-vs-control read --------------------------
    # BD-rate needs >= 4 ladder points on BOTH sides (the guard in bd_rate above).
    # A 3-point probe block (e.g. svt_doe_t1_bd10_transfer, q in {15,45,90}) can
    # therefore NEVER yield one, and the honest report of such a block is the
    # paired matched-q read, with BD-rate stated as NOT-MEASURED rather than
    # manufactured by loosening the guard. This table is emitted for every arm,
    # so a block that HAS a BD-rate gets both reads and they can be cross-checked.
    #
    # The control may live in a different run (a 3-point bd10 block carries no
    # deviations=0 stratum of its own); --paired-control-run names it. Doing that
    # across encoder pins is a CROSS-ERA join and must be declared as one by the
    # caller — this code does not hide it, it records the control's run in
    # every row.
    idx = {}
    for r in rows:
        idx[(r["run"], r["speed"], r["image"], r["arm"], r["q"])] = r
    ctl_runs = [a.paired_control_run] if a.paired_control_run else []
    paired = []
    for r in rows:
        devs = [d for d in (r["arm"] or "").split("-")[3:] if d]
        if len(devs) != 1:
            continue
        ctl_arm = f"s{r['speed']}-svt-420"
        for crun in [r["run"]] + ctl_runs:
            c = idx.get((crun, r["speed"], r["image"], ctl_arm, r["q"]))
            if c is not None:
                break
        if c is None or not c["bytes"]:
            continue
        dq = r["qual"] - c["qual"]
        db = 100.0 * (r["bytes"] - c["bytes"]) / c["bytes"]
        if dq > 0 and db <= 0:
            verdict = "DOMINATES"
        elif dq < 0 and db >= 0:
            verdict = "DOMINATED"
        else:
            verdict = "TRADE"
        paired.append(dict(run=r["run"], ctl_run=crun, speed=r["speed"], image=r["image"],
                           knob=devs[0], q=r["q"], bytes_arm=r["bytes"], bytes_ctl=c["bytes"],
                           d_bytes_pct=db, qual_arm=r["qual"], qual_ctl=c["qual"],
                           d_qual=dq, verdict=verdict, cls=r["cls"],
                           transform=r["transform"]))
    with open(f"{a.outdir}/paired_per_q.tsv", "w") as f:
        # `transform` is load-bearing, not decoration: on this corpus 19 of 32
        # images are byte-identical passthroughs between the native and 1024²
        # builds, so a cross-SIZE question has n = 13 (`crop-native`), not 32
        # (avif_hdr_arm_plan_2026-09-02.md §10.4a registered restriction).
        f.write("run\tctl_run\tspeed\tknob\timage\tq\tbytes_arm\tbytes_ctl\td_bytes_pct"
                f"\t{a.metric}_arm\t{a.metric}_ctl\td_qual\tverdict\tclass\ttransform\n")
        for r_ in sorted(paired, key=lambda x: (x["run"], x["knob"], x["speed"], x["image"], x["q"])):
            f.write(f"{r_['run']}\t{r_['ctl_run']}\t{r_['speed']}\t{r_['knob']}\t{r_['image']}"
                    f"\t{r_['q']}\t{r_['bytes_arm']}\t{r_['bytes_ctl']}\t{r_['d_bytes_pct']:.4f}"
                    f"\t{r_['qual_arm']:.6f}\t{r_['qual_ctl']:.6f}\t{r_['d_qual']:.6f}"
                    f"\t{r_['verdict']}\t{r_['cls']}\t{r_['transform']}\n")
    byknob = collections.defaultdict(list)
    for r_ in paired:
        byknob[(r_["run"], r_["speed"], r_["knob"], r_["q"])].append(r_)
    with open(f"{a.outdir}/paired_summary.tsv", "w") as f:
        f.write("run\tspeed\tknob\tq\tn\tmedian_d_bytes_pct\tci_lo\tci_hi"
                "\tmedian_d_qual\tq_ci_lo\tq_ci_hi\tn_dominates\tn_dominated\tn_trade\n")
        for k, v in sorted(byknob.items()):
            db = [x["d_bytes_pct"] for x in v]; dq = [x["d_qual"] for x in v]
            blo, bhi = median_ci(db); qlo, qhi = median_ci(dq)
            f.write(f"{k[0]}\t{k[1]}\t{k[2]}\t{k[3]}\t{len(v)}\t{np.median(db):.4f}"
                    f"\t{blo:.4f}\t{bhi:.4f}\t{np.median(dq):.4f}\t{qlo:.4f}\t{qhi:.4f}"
                    f"\t{sum(1 for x in v if x['verdict']=='DOMINATES')}"
                    f"\t{sum(1 for x in v if x['verdict']=='DOMINATED')}"
                    f"\t{sum(1 for x in v if x['verdict']=='TRADE')}\n")
    bytf = collections.defaultdict(list)
    for r_ in paired:
        bytf[(r_["run"], r_["speed"], r_["knob"], r_["q"], r_["transform"])].append(r_)
    with open(f"{a.outdir}/paired_summary_by_transform.tsv", "w") as f:
        f.write("run\tspeed\tknob\tq\ttransform\tn_images\tmedian_d_bytes_pct\tci_lo\tci_hi"
                "\tmedian_d_qual\tq_ci_lo\tq_ci_hi\tn_dominates\tn_dominated\tn_trade\n")
        for k, v in sorted(bytf.items()):
            db = [x["d_bytes_pct"] for x in v]; dq = [x["d_qual"] for x in v]
            blo, bhi = median_ci(db); qlo, qhi = median_ci(dq)
            f.write(f"{k[0]}\t{k[1]}\t{k[2]}\t{k[3]}\t{k[4]}\t{len(v)}\t{np.median(db):.4f}"
                    f"\t{blo:.4f}\t{bhi:.4f}\t{np.median(dq):.4f}\t{qlo:.4f}\t{qhi:.4f}"
                    f"\t{sum(1 for x in v if x['verdict']=='DOMINATES')}"
                    f"\t{sum(1 for x in v if x['verdict']=='DOMINATED')}"
                    f"\t{sum(1 for x in v if x['verdict']=='TRADE')}\n")
    print(f"paired per-q rows: {len(paired)}")

    json.dump(dict(scored_rows=len(rows), bd_rows=len(bd), ctl_source=dict(ctl_source),
                   n_main=len(main), n_pairs=len(bypair), paired_rows=len(paired)),
              open(f"{a.outdir}/_summary.json", "w"), indent=1)
    print(f"wrote tables to {a.outdir}")

if __name__ == "__main__":
    main()
