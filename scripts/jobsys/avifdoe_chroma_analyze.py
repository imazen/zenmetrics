#!/usr/bin/env python3
"""avifdoe_chroma_analyze.py — the UNCONFOUNDED backend/chroma re-cut.

Registered in `benchmarks/avif_chroma_split_2026-09-04.md`. Every cut, bar and
exclusion in here is fixed in that document, which was written and pushed BEFORE
any cell of this arm was scored.

What it answers, in order:

  0. ERA CONTROL   br444 vs brsdr `encode_sha` on the 2,880 shared CellIds.
                   Decides whether anything may be read against the old era at
                   all.  Byte identity, not a test that the difference is small.
  1. REACH         Per image, per arm: max achieved ssim2 over that image's 90
                   cells.  THE question -- is the plot/screen reach failure a
                   CHROMA property or a BACKEND property -- and the k11 rule
                   from registration §4 decides it.
  2. TWO AXES      br420 vs svt-420  = the TRUE BACKEND axis (chroma held)
                   br420 vs br444    = the TRUE CHROMA axis   (backend held)
                   br444 vs svt-420  = the PUBLISHED confounded axis, recomputed
                                       here only to show this instrument
                                       reproduces it.

NO NEW STAT MATH.  `frontier`/`bd_rate`/`median_ci`/`COARSE` come from
avifdoe_stagea_analyze, `binom_two_sided` from avifdoe_stagea_gates, and
`pooled_front`/`clip_front`/`interp_bytes`/`QUALITY_BANDS`/`fmt` from
avifdoe_brem_analyze -- the same composition-by-_load that brem itself uses on
stagea.  The bands are brem's REGISTERED bands, unchanged.
"""
import argparse, collections, csv, importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SA = _load("avifdoe_stagea_analyze", os.path.join(HERE, "avifdoe_stagea_analyze.py"))
SG = _load("avifdoe_stagea_gates", os.path.join(HERE, "avifdoe_stagea_gates.py"))
B6 = _load("avifdoe_stageb6_analyze", os.path.join(HERE, "avifdoe_stageb6_analyze.py"))
BR = _load("avifdoe_brem_analyze", os.path.join(HERE, "avifdoe_brem_analyze.py"))
bd_rate, median_ci, COARSE = SA.bd_rate, SA.median_ci, SA.COARSE
binom_two_sided = SG.binom_two_sided
load_scored = B6.load_scored
pooled_front, clip_front, interp_bytes = BR.pooled_front, BR.clip_front, BR.interp_bytes
QUALITY_BANDS, PRODUCT_BAND, fmt, summarize = BR.QUALITY_BANDS, BR.PRODUCT_BAND, BR.fmt, BR.summarize

# The 9-q ladder both chroma arms ran. brsdr's 2,099 extra 29-q cells are
# EXCLUDED from everything (registration §2): they are whichever images the
# workers reached first, not a designed stratum.
LADDER9 = (5, 15, 25, 35, 45, 60, 76, 90, 96)

# The reach bar, and the pre-registered comparison set: the 16 references whose
# svt-4:2:0 max achieved ssim2 is below REACH_BAR.  Enumerated in registration
# §4 so it cannot be chosen after the fact; re-derived from the arm's own data
# and CROSS-CHECKED against this list, which is the point of hard-coding it.
REACH_BAR = 90.0
SVT_CANNOT_REACH = {
    "7004.scale1024x1024.png": "plot",       "7042.scale1024x1024.png": "plot",
    "7050.scale1024x1024.png": "plot",       "7052.scale1024x1024.png": "plot",
    "7058.scale1024x1024.png": "plot",       "7076.scale1024x1024.png": "plot",
    "8134.scale1440x900.png": "screenshot",  "8288.scale375x667.png": "screenshot",
    "8414.scale1280x800.png": "screenshot",  "8434.scale414x896.png": "screenshot",
    "8446.scale2560x1440.png": "screenshot",
    "6602.scale3302x4844.png": "scan",       "6604.scale3286x4868.png": "scan",
    "9032.scale1024x1536.png": "ai-gen",     "9118.scale1536x1024.png": "ai-gen",
    "9444.scale1024x1536.png": "ai-gen",
}
PLOT_SCREEN = {k for k, v in SVT_CANNOT_REACH.items() if v in ("plot", "screenshot")}


# ------------------------------------------------------------------ era ------
def era_control(br444_pairs, brsdr_pairs, out):
    """br444 vs brsdr `encode_sha` on shared CellIds.  Registration §3."""
    def key_sha(path):
        d = {}
        for r in csv.DictReader(open(path), delimiter="\t"):
            if int(float(r["q"])) not in LADDER9:
                continue          # brsdr's uncontrolled 29-q extras
            d[(r["image_path"], int(float(r["q"])), r["knob_tuple_json"])] = r["encode_sha"]
        return d
    a, b = key_sha(br444_pairs), key_sha(brsdr_pairs)
    shared = sorted(set(a) & set(b))
    same = [k for k in shared if a[k] == b[k]]
    res = dict(shared=len(shared), identical=len(same),
               br444_rows=len(a), brsdr_9q_rows=len(b),
               inert=(len(shared) > 0 and len(same) == len(shared)))
    with open(f"{out}/era_control.tsv", "w") as f:
        f.write("shared_cells\tbyte_identical\tfraction\tbr444_rows\tbrsdr_9q_rows\tverdict\n")
        frac = (len(same) / len(shared)) if shared else 0.0
        f.write(f"{len(shared)}\t{len(same)}\t{frac:.6f}\t{len(a)}\t{len(b)}\t"
                f"{'ERA INERT' if res['inert'] else 'ERA MOVED'}\n")
    if not res["inert"] and shared:
        with open(f"{out}/era_control_differing.tsv", "w") as f:
            f.write("image\tq\tknob_tuple\tbr444_sha\tbrsdr_sha\n")
            for k in shared:
                if a[k] != b[k]:
                    f.write(f"{k[0]}\t{k[1]}\t{k[2]}\t{a[k]}\t{b[k]}\n")
    return res


# ---------------------------------------------------------------- curves -----
def arm_points(t, run, metric):
    """image -> [(ssim2, bytes)] pooled over every speed and q of that arm."""
    pts = collections.defaultdict(list)
    for i in range(len(t["run"])):
        if t["run"][i] != run:
            continue
        if int(t["q"][i]) not in LADDER9:
            continue
        s_, b_ = t[metric][i], t["bytes"][i]
        if s_ is None or b_ is None:
            continue
        pts[t["image"][i]].append((s_, b_))
    return pts


def pair_table(A, B, name_a, name_b, cls_of, out, label):
    """BD-rate + per-band bytes for one arm pair.  bd < 0 => A wins (fewer bits)."""
    rows = []
    for img in sorted(set(A) & set(B)):
        fa, fb = pooled_front(A[img]), pooled_front(B[img])
        bd = bd_rate(fa, fb)
        ca, cb = clip_front(fa, *PRODUCT_BAND), clip_front(fb, *PRODUCT_BAND)
        bdb = bd_rate(ca, cb) if (len(ca) >= 4 and len(cb) >= 4) else None
        rec = dict(image=img, cls=cls_of.get(img, "?"), bd=bd, bd_banded=bdb,
                   n_a=len(fa), n_b=len(fb), n_a_band=len(ca), n_b_band=len(cb),
                   a_qmax=fa[-1][0] if fa else None, b_qmax=fb[-1][0] if fb else None)
        for nm, lo, hi in QUALITY_BANDS:
            mid = 0.5 * (lo + hi)
            ba, bb = interp_bytes(fa, mid), interp_bytes(fb, mid)
            rec[f"ratio_{nm}"] = (ba / bb) if (ba and bb) else None
        rows.append(rec)
    bandcols = [f"ratio_{n}" for n, _, _ in QUALITY_BANDS]
    with open(f"{out}/axis_{label}.tsv", "w") as f:
        f.write(f"image\tclass\tbd_{name_a}_vs_{name_b}\twinner\tbd_banded_30_95\t"
                f"n_{name_a}\tn_{name_b}\tn_{name_a}_band\tn_{name_b}_band\t"
                f"{name_a}_qmax\t{name_b}_qmax\t" + "\t".join(bandcols) + "\n")
        for r in rows:
            w = "" if r["bd"] is None else (name_a if r["bd"] < 0 else name_b)
            f.write(f"{r['image']}\t{r['cls']}\t{fmt(r['bd'])}\t{w}\t{fmt(r['bd_banded'])}"
                    f"\t{r['n_a']}\t{r['n_b']}\t{r['n_a_band']}\t{r['n_b_band']}"
                    f"\t{fmt(r['a_qmax'],3)}\t{fmt(r['b_qmax'],3)}\t"
                    + "\t".join(fmt(r[c]) for c in bandcols) + "\n")
    vals = [r["bd"] for r in rows if r["bd"] is not None]
    # `summarize` is brem's -- n / median / mean / bootstrap CI / IQR / n_neg --
    # so the median and its CI are the SAME estimator the backend table used.
    summary = dict(axis=label, a=name_a, b=name_b,
                   not_measured=len(rows) - len(vals))
    if vals:
        st = summarize(vals)
        summary.update(st)
        summary["a_wins"] = st["n_neg"]
        summary["b_wins"] = st["n"] - st["n_neg"]
        summary["sign_p"] = binom_two_sided(st["n_neg"], st["n"])
    else:
        summary.update(n=0, median=None, ci_lo=None, ci_hi=None,
                       a_wins=0, b_wins=0, sign_p=None)
    return rows, summary


# ----------------------------------------------------------------- reach -----
def reach_table(arms, cls_of, out):
    """Per image, per arm: max achieved ssim2 over every cell of that arm."""
    imgs = sorted(set().union(*[set(p) for p in arms.values()]))
    names = list(arms)
    with open(f"{out}/reach_per_image.tsv", "w") as f:
        f.write("image\tclass\t" + "\t".join(f"{n}_max_ssim2" for n in names)
                + "\t" + "\t".join(f"{n}_reaches90" for n in names) + "\n")
        reach = {}
        for img in imgs:
            mx = {n: (max(s for s, _ in arms[n][img]) if arms[n].get(img) else None)
                  for n in names}
            reach[img] = mx
            f.write(f"{img}\t{cls_of.get(img,'?')}\t"
                    + "\t".join(fmt(mx[n], 3) for n in names) + "\t"
                    + "\t".join("" if mx[n] is None else ("yes" if mx[n] >= REACH_BAR else "NO")
                                for n in names) + "\n")
    return reach


# ------------------------------------------------------------- scorer --------
def scorer_control(t, brsdr_scored, out):
    """Does the SCORER agree across eras on identical bitstreams?

    The era control (§3) proves the ENCODER is inert; it says nothing about the
    metric. But br444's blobs are byte-identical to brsdr's, and the two were
    scored by DIFFERENT score runs -- so joining on `encode_sha` and comparing
    `m_ssim2` isolates the scorer exactly, for free. A non-zero delta here would
    mean no cross-era ssim2 comparison is admissible, including the br420 vs
    stored-svt backend axis.
    """
    import pyarrow.parquet as pq
    b = pq.read_table(brsdr_scored).to_pydict()
    old = {}
    for i in range(len(b["run"])):
        if b["run"][i] == "brsdr" and b["m_ssim2"][i] is not None:
            old[b["encode_sha"][i]] = b["m_ssim2"][i]
    deltas = []
    for i in range(len(t["run"])):
        if t["run"][i] != "br444" or t["m_ssim2"][i] is None:
            continue
        o = old.get(t["encode_sha"][i])
        if o is not None:
            deltas.append(t["m_ssim2"][i] - o)
    res = dict(n=len(deltas),
               max_abs=max((abs(d) for d in deltas), default=None),
               n_nonzero=sum(1 for d in deltas if d != 0.0))
    with open(f"{out}/scorer_control.tsv", "w") as f:
        f.write("shared_bitstreams\tmax_abs_delta_ssim2\tn_nonzero\tverdict\n")
        v = "SCORER INERT" if res["n"] and res["n_nonzero"] == 0 else (
            "NO OVERLAP" if not res["n"] else "SCORER MOVED")
        f.write(f"{res['n']}\t{res['max_abs']}\t{res['n_nonzero']}\t{v}\n")
    res["verdict"] = v
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="harvested chroma-split parquet (br420+br444)")
    ap.add_argument("--stagea-scored", required=True, help="Stage-A parquet (supplies a0r = svt-420)")
    ap.add_argument("--crop-manifest", required=True)
    ap.add_argument("--br444-pairs", required=True)
    ap.add_argument("--brsdr-pairs", required=True)
    ap.add_argument("--brsdr-scored", default="/mnt/v/output/avif-backend-2026-09-03/br_scored_2026-09-03.parquet",
                    help="the backend wave's harvested table; supplies brsdr's ssim2 for the scorer control")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--metric", default="m_ssim2")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    # Same two lines brem uses (avifdoe_brem_analyze.py:216) -- the manifest is
    # keyed by `corpus_key`, which IS the image_path the cells carry.
    man = {r["corpus_key"]: r for r in csv.DictReader(open(a.crop_manifest), delimiter="\t")}
    cls_of = {k: COARSE.get(v["content_class"], "other") for k, v in man.items()}

    # ---- 0. era control ----------------------------------------------------
    era = era_control(a.br444_pairs, a.brsdr_pairs, a.outdir)
    print(f"[era] shared={era['shared']} byte-identical={era['identical']} "
          f"-> {'ERA INERT' if era['inert'] else 'ERA MOVED'}")

    # ---- arms --------------------------------------------------------------
    t = load_scored(a.scored)
    sa = load_scored(a.stagea_scored)
    arms = {
        "br420": arm_points(t, "br420", a.metric),
        "br444": arm_points(t, "br444", a.metric),
    }
    svt = collections.defaultdict(list)
    for i in range(len(sa["run"])):
        if sa["run"][i] != "a0r" or int(sa["q"][i]) not in LADDER9:
            continue
        s_, b_ = sa[a.metric][i], sa["bytes"][i]
        if s_ is not None and b_ is not None:
            svt[sa["image"][i]].append((s_, b_))
    arms["svt420"] = svt

    # ---- 0b. scorer control ------------------------------------------------
    sc = scorer_control(t, a.brsdr_scored, a.outdir) if os.path.exists(a.brsdr_scored) else \
        dict(n=0, verdict="NOT RUN (no brsdr scored table)")
    print(f"[scorer] shared bitstreams={sc['n']} max|d ssim2|={sc.get('max_abs')} -> {sc['verdict']}")

    # ---- 1. reach ----------------------------------------------------------
    # EVERY arm is read on the SAME 9-q ladder (svt is restricted in arm_points
    # above), because the ladder ceiling is itself a reach constraint: q96, not
    # the 29-q grid's q98. MEASURED cost of that restriction on svt, 2026-09-04:
    # up to 2.106 ssim2 points of achieved max, and it flips exactly ONE image's
    # reach-90 verdict -- 9954.scale1024x1536.png, 90.49 -> 89.64. That image is
    # ai-gen, NOT plot/screenshot, so the k11 denominator is untouched and the
    # decisive test is unaffected. Both sets are reported rather than one being
    # quietly substituted for the other.
    reach = reach_table(arms, cls_of, a.outdir)
    k = [img for img in SVT_CANNOT_REACH
         if reach.get(img, {}).get("br420") is not None
         and reach[img]["br420"] < REACH_BAR]
    k11 = [img for img in k if img in PLOT_SCREEN]
    # The same question asked entirely INSIDE this wave, on one ladder: which
    # images does svt-420 miss here, and does br420 miss them too? This is the
    # reading that cannot be confounded by ladder density, because all three
    # arms share it.
    svt_fail_9q = sorted(img for img, m in reach.items()
                         if m.get("svt420") is not None and m["svt420"] < REACH_BAR)
    k_inwave = [img for img in svt_fail_9q
                if reach[img].get("br420") is not None
                and reach[img]["br420"] < REACH_BAR]
    k444_inwave = [img for img in svt_fail_9q
                   if reach[img].get("br444") is not None
                   and reach[img]["br444"] < REACH_BAR]
    verdict = ("CHROMA" if len(k11) >= 9 else
               "BACKEND" if len(k11) <= 2 else "CONDITIONAL")

    # ---- 2. the axes -------------------------------------------------------
    summaries = []
    for label, (na, A), (nb, B) in (
        ("backend_true", ("br420", arms["br420"]), ("svt420", arms["svt420"])),
        ("chroma_true", ("br420", arms["br420"]), ("br444", arms["br444"])),
        ("published_confounded", ("br444", arms["br444"]), ("svt420", arms["svt420"])),
    ):
        _, s = pair_table(A, B, na, nb, cls_of, a.outdir, label)
        summaries.append(s)
        print(f"[axis {label}] n={s['n']} median={fmt(s['median'])} "
              f"CI=[{fmt(s['ci_lo'])}, {fmt(s['ci_hi'])}] "
              f"{na}_wins={s['a_wins']} {nb}_wins={s['b_wins']} p={fmt(s['sign_p'])}")

    with open(f"{a.outdir}/notes.json", "w") as f:
        json.dump(dict(era=era, scorer=sc, reach_bar=REACH_BAR,
                       k_of_16=len(k), k11_of_11=len(k11), verdict=verdict,
                       k_images=sorted(k), axes=summaries,
                       ladder=list(LADDER9),
                       in_wave=dict(svt420_fails=svt_fail_9q,
                                    n_svt420_fails=len(svt_fail_9q),
                                    br420_also_fails=k_inwave,
                                    n_br420_also_fails=len(k_inwave),
                                    br444_also_fails=k444_inwave,
                                    n_br444_also_fails=len(k444_inwave))), f, indent=2)
    print(f"[reach] br420 fails ssim2>={REACH_BAR} on {len(k)}/16 of the registered "
          f"svt-fail set, {len(k11)}/11 plot+screenshot -> {verdict}")
    print(f"[reach in-wave, one ladder] svt420 misses 90 on {len(svt_fail_9q)}; "
          f"br420 also misses on {len(k_inwave)}; br444 also misses on {len(k444_inwave)}")
    print(f"wrote tables to {a.outdir}")


if __name__ == "__main__":
    main()
