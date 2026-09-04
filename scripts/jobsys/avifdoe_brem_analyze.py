#!/usr/bin/env python3
"""avifdoe_brem_analyze.py — the Stage-B REMAINDER analysis (brnat + brsdr).

Registered in `benchmarks/avif_stageB_remainder_2026-09-03.md`. Answers, in the
registered order:

  Q1  QM x sharpness AT NATIVE — the complete 4x3 factorial the B-6 gap named
      (§17.4), with the crop-vs-native A/B against A2's matched budget-size arms.
  Q2  THE BACKEND TABLE — the first per-image SDR cross-backend comparison
      (zenrav1e `brsdr` vs svt `a0r`), matched in QUALITY space, never q space,
      then joined to the speed instrument's wall-time coefficients.
  Q3  feeds the decision-surface statement in the report (this script emits the
      tables; the prose is the report's).

STATS OWNERSHIP — nothing is re-implemented here. `frontier`, `bd_rate`,
`median_ci`, `q1q3` and the content-class map are IMPORTED from
`avifdoe_stagea_analyze.py`; `binom_two_sided` from `avifdoe_stagea_gates.py`;
`curves` / `bd_table` shape-helpers from `avifdoe_stageb6_analyze.py`. The
interaction residual is the Stage-A owner's construction verbatim
(`resid = observed_pair_BD - (BD_k1 + BD_k2)`, per image, in BD-rate points).

BD-rate sign convention (inherited from zenavif `scripts/rd_gap/bd_arm.py`):
NEGATIVE = the arm needs FEWER bits at matched quality = the arm WINS.

QUALITY SPACE, NOT q SPACE. q values do not align across backends (registration
§3.2). Every cross-backend number here is either a BD-rate over the OVERLAPPING
achieved-ssim2 span, or a byte reading interpolated on the achieved-ssim2
frontier at a fixed ssim2 target. No comparison is ever made at equal q.

zensim is emitted by the scorer as a 720-wide FEATURE vector, not a scalar
(harvest docstring), so `m_ssim2` is the only corpus-wide scalar quality
response and is the matching axis. That is a stated limitation, not a choice.
"""
import argparse, collections, csv, importlib.util, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SA = _load("avifdoe_stagea_analyze", os.path.join(HERE, "avifdoe_stagea_analyze.py"))
SG = _load("avifdoe_stagea_gates", os.path.join(HERE, "avifdoe_stagea_gates.py"))
B6 = _load("avifdoe_stageb6_analyze", os.path.join(HERE, "avifdoe_stageb6_analyze.py"))
frontier, bd_rate, median_ci, q1q3, COARSE = SA.frontier, SA.bd_rate, SA.median_ci, SA.q1q3, SA.COARSE
binom_two_sided = SG.binom_two_sided
curves, load_scored = B6.curves, B6.load_scored

# The 9 q points brnat ran. A2 ran the SAME 9 on the same 32 images with the same
# 26 strata, so restricting A2 to this set makes the crop-vs-native A/B an EXACT
# q match and isolates SIZE from ladder density -- B-6's BUDGET_Q, same values.
BRNAT_Q = (5, 15, 25, 35, 45, 60, 76, 90, 96)

# The 4x3 factorial's axes, as the plan spells them.
QML_LEVELS = ("qml1.2.10", "qml1.4.10", "qml1.8.15")
SHP_LEVELS = ("shp3", "shp7")

# Cross-backend quality bands, REGISTERED HERE before the numbers are read.
# Chosen on the product argument, not on the data: web delivery lives in the
# mid/high bands, and the near-lossless band is where the metric is weakest and
# where byte differences are largest in absolute terms.
QUALITY_BANDS = (("low", 30.0, 50.0), ("mid", 50.0, 70.0),
                 ("high", 70.0, 85.0), ("nearlossless", 85.0, 95.0))


def interp_bytes(front, target):
    """Bytes at a target quality on a monotone (quality, bytes) frontier, or None
    if the target is outside the frontier's achieved span. NEVER extrapolates --
    an unreachable quality is reported as unreachable, not as a guessed size."""
    if len(front) < 2:
        return None
    xs = np.array([p[0] for p in front], dtype=float)
    ys = np.array([p[1] for p in front], dtype=float)
    if target < xs[0] or target > xs[-1]:
        return None
    return float(np.exp(np.interp(target, xs, np.log(ys))))


def pooled_front(pts):
    """Pareto frontier over points pooled across every dial position."""
    return frontier(pts)


# The PRODUCT band. A full-overlap BD-rate on these ladders integrates from
# ssim2 ~= -60 (q1/q5 cells) upward, so the deep-negative tail -- where nobody
# ships and where the metric is least trustworthy -- carries most of the
# integration weight. MEASURED consequence: the widest cross-backend BD-rates in
# this wave (+156 %, +118 %) are dominated by that tail. So every cross-backend
# BD-rate is reported TWICE: the mechanical full-overlap figure, and this
# band-restricted one. Neither replaces the other; the banded one is the product
# reading and the full one is the honest default.
PRODUCT_BAND = (30.0, 95.0)


def clip_front(front, lo, hi):
    """Restrict a monotone (quality, bytes) frontier to [lo, hi], interpolating
    the two endpoints in log-rate. Returns [] if the frontier does not reach the
    band. Feeds the OWNER's bd_rate -- the integrator itself is not re-written."""
    if len(front) < 2:
        return []
    xs = [p[0] for p in front]
    if xs[-1] < lo or xs[0] > hi:
        return []
    out = []
    a_lo, a_hi = max(lo, xs[0]), min(hi, xs[-1])
    if a_hi <= a_lo:
        return []
    b = interp_bytes(front, a_lo)
    if b is not None:
        out.append((a_lo, b))
    out.extend((q, by) for q, by in front if a_lo < q < a_hi)
    b = interp_bytes(front, a_hi)
    if b is not None:
        out.append((a_hi, b))
    return out


def fmt(x, nd=4):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


# ------------------------------------------------------------------- Q1 ------
def bd_vs_control(cur, run, control_arm):
    """BD-rate of every arm vs its own (image) control within one run."""
    out = []
    for (r, sp, img, arm), pts in cur.items():
        if r != run or arm == control_arm:
            continue
        ref = cur.get((r, sp, img, control_arm))
        if not ref:
            continue
        v = bd_rate(frontier(pts), frontier(ref))
        if v is None:
            continue
        devs = [d for d in arm.split("-")[3:] if d]
        out.append(dict(run=r, speed=sp, image=img, arm=arm, devs=devs,
                        knob="|".join(devs), n_dev=len(devs), bd_rate=v, n_pts=len(pts)))
    return out


def main_effect_map(bd):
    """(speed, knob) -> {image: bd_rate} for single-deviation arms only."""
    m = collections.defaultdict(dict)
    for r in bd:
        if len(r["devs"]) == 1:
            m[(r["speed"], r["devs"][0])][r["image"]] = r["bd_rate"]
    return m


def interactions(bd, main, want=None):
    """Stage-A's construction verbatim: resid = observed - (a1 + a2), per image."""
    out = []
    for r in bd:
        if len(r["devs"]) != 2:
            continue
        k1, k2 = r["devs"]
        if want is not None and (k1, k2) not in want and (k2, k1) not in want:
            continue
        a1 = main.get((r["speed"], k1), {}).get(r["image"])
        a2 = main.get((r["speed"], k2), {}).get(r["image"])
        if a1 is None or a2 is None:
            continue
        out.append(dict(speed=r["speed"], image=r["image"], k1=k1, k2=k2,
                        observed=r["bd_rate"], a1=a1, a2=a2, additive=a1 + a2,
                        resid=r["bd_rate"] - (a1 + a2)))
    return out


def summarize(vals):
    a = np.asarray(vals, dtype=float)
    lo, hi = q1q3(a) if a.size >= 2 else (float("nan"), float("nan"))
    clo, chi = median_ci(a)
    return dict(n=int(a.size), median=float(np.median(a)), mean=float(np.mean(a)),
                ci_lo=clo, ci_hi=chi, iqr=float(hi - lo), q1=float(lo), q3=float(hi),
                n_neg=int((a < 0).sum()), amin=float(a.min()), amax=float(a.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--br-scored", required=True, help="harvested brnat+brsdr parquet")
    ap.add_argument("--stagea-scored", required=True, help="Stage-A parquet (supplies a2 + a0r)")
    ap.add_argument("--crop-manifest", required=True)
    ap.add_argument("--native-dims", required=True, help="TSV image/width/height for the NATIVE corpus")
    ap.add_argument("--speed-coef", required=True, help="speed_alpha_beta.tsv")
    ap.add_argument("--speed-coef-per-source", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--metric", default="m_ssim2")
    ap.add_argument("--parity-check", default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    notes = []

    # ---- gate 0: BD-rate parity against the house implementation ------------
    if a.parity_check:
        m = _load("bd_arm", a.parity_check)
        rng = np.random.default_rng(20260903)
        worst = 0.0
        for _ in range(200):
            n = int(rng.integers(5, 12))
            ref = sorted(zip(np.sort(rng.uniform(30, 95, n)), np.sort(rng.uniform(2e3, 3e5, n))))
            tst = sorted(zip(np.sort(rng.uniform(30, 95, n)), np.sort(rng.uniform(2e3, 3e5, n))))
            x, y = bd_rate(frontier(tst), frontier(ref)), m.bd_rate(m.frontier(tst), m.frontier(ref))
            if x is None or y is None:
                assert x is None and y is None
                continue
            worst = max(worst, abs(x - y))
        assert worst == 0.0, f"BD-rate DIVERGES from zenavif bd_arm.py by {worst}"
        print(f"GATE bd-parity: identical to zenavif bd_arm.py on 200 ladders (max |d| {worst})")
        notes.append(dict(gate="bd_rate_parity", result="PASS", detail="max |delta| 0.0 on 200 random ladders"))

    man = {r["corpus_key"]: r for r in csv.DictReader(open(a.crop_manifest), delimiter="\t")}
    cls_of = {k: COARSE.get(v["content_class"], "other") for k, v in man.items()}
    passthru = {k for k, v in man.items() if v["transform"] == "native"}
    crop_px = {k: int(v["width"]) * int(v["height"]) for k, v in man.items()}
    nat_px = {}
    for r in csv.DictReader(open(a.native_dims), delimiter="\t"):
        nat_px[r["image"]] = int(r["width"]) * int(r["height"])

    br = load_scored(a.br_scored)
    sa = load_scored(a.stagea_scored)

    # ---- gate 1: cross-run control byte identity ----------------------------
    # For the 19 passthrough references the budget corpus IS the native pixels,
    # so A2's s6-svt-420 encode must be the SAME BITSTREAM as brnat's at every
    # shared q. Anything else means the two legs differ in configuration rather
    # than in pixels, which would confound the whole crop-vs-native A/B.
    def sha_index(t, run, arm, qset):
        d = {}
        for i in range(len(t["run"])):
            if t["run"][i] == run and t["arm"][i] == arm and t["q"][i] in qset:
                d[(t["image"][i], t["q"][i])] = t["encode_sha"][i]
        return d
    n_sha = sha_index(br, "brnat", "s6-svt-420", set(BRNAT_Q))
    a_sha = sha_index(sa, "a2", "s6-svt-420", set(BRNAT_Q))
    shared = [k for k in n_sha if k in a_sha and k[0] in passthru]
    ident = sum(1 for k in shared if n_sha[k] == a_sha[k])
    cropped_shared = [k for k in n_sha if k in a_sha and k[0] not in passthru]
    cropped_ident = sum(1 for k in cropped_shared if n_sha[k] == a_sha[k])
    print(f"GATE control-identity: passthrough refs {ident}/{len(shared)} byte-identical "
          f"(brnat vs a2 control); cropped refs {cropped_ident}/{len(cropped_shared)} "
          f"(expected 0 -- different pixels)")
    notes.append(dict(gate="control_byte_identity",
                      result="PASS" if (len(shared) and ident == len(shared) and cropped_ident == 0) else "CHECK",
                      detail=f"passthrough {ident}/{len(shared)} identical; cropped {cropped_ident}/{len(cropped_shared)} identical"))

    # ---- gate 2: per-stratum byte-identity census (the "inert here" signal) --
    # Registration §4.2: 22.3 % of brnat cells are byte-identical to some other
    # cell. Counted PER (stratum, image) -- never pooled -- it is an upper bound
    # on how much of the grid can carry any interaction signal at all.
    ctl_sha = {}
    for i in range(len(br["run"])):
        if br["run"][i] == "brnat" and br["arm"][i] == "s6-svt-420":
            ctl_sha[(br["image"][i], br["q"][i])] = br["encode_sha"][i]
    inert = collections.defaultdict(lambda: [0, 0])
    for i in range(len(br["run"])):
        if br["run"][i] != "brnat" or br["arm"][i] == "s6-svt-420":
            continue
        c = ctl_sha.get((br["image"][i], br["q"][i]))
        if c is None:
            continue
        k = (br["arm"][i], br["image"][i])
        inert[k][1] += 1
        if br["encode_sha"][i] == c:
            inert[k][0] += 1
    with open(f"{a.outdir}/brnat_inert_per_stratum_image.tsv", "w") as f:
        f.write("arm\timage\tclass\tn_q\tn_identical_to_control\tfrac\n")
        for (arm, img), (nid, tot) in sorted(inert.items()):
            f.write(f"{arm}\t{img}\t{cls_of.get(img,'?')}\t{tot}\t{nid}\t{nid/tot:.4f}\n")
    by_arm = collections.defaultdict(lambda: [0, 0])
    for (arm, img), (nid, tot) in inert.items():
        by_arm[arm][0] += nid; by_arm[arm][1] += tot
    with open(f"{a.outdir}/brnat_inert_per_stratum.tsv", "w") as f:
        f.write("arm\tn_cells\tn_identical_to_control\tfrac_inert\tn_images_fully_inert\n")
        for arm in sorted(by_arm):
            nid, tot = by_arm[arm]
            full = sum(1 for (ar, im), (i2, t2) in inert.items() if ar == arm and i2 == t2)
            f.write(f"{arm}\t{tot}\t{nid}\t{nid/tot:.4f}\t{full}\n")

    # ================================================================= Q1 =====
    cur_nat = curves(br, a.metric)
    cur_bud = curves(sa, a.metric, qset=set(BRNAT_Q))
    bd_nat = bd_vs_control(cur_nat, "brnat", "s6-svt-420")
    strata = {r["arm"] for r in bd_nat}
    bd_bud = [r for r in bd_vs_control(cur_bud, "a2", "s6-svt-420") if r["arm"] in strata]
    main_nat, main_bud = main_effect_map(bd_nat), main_effect_map(bd_bud)

    with open(f"{a.outdir}/bd_per_image.tsv", "w") as f:
        f.write("leg\tspeed\timage\tclass\tarm\tknob\tn_dev\tbd_rate\tn_pts\n")
        for leg, rows in (("native", bd_nat), ("budget", bd_bud)):
            for r in sorted(rows, key=lambda x: (x["arm"], x["image"])):
                f.write(f"{leg}\t{r['speed']}\t{r['image']}\t{cls_of.get(r['image'],'?')}\t{r['arm']}"
                        f"\t{r['knob']}\t{r['n_dev']}\t{r['bd_rate']:.4f}\t{r['n_pts']}\n")

    def eff_table(rows, path, cls_filter=None):
        byk = collections.defaultdict(list)
        for r in rows:
            if cls_filter and cls_of.get(r["image"]) != cls_filter:
                continue
            byk[r["arm"]].append(r["bd_rate"])
        with open(path, "w") as f:
            f.write("arm\tn\tmedian\tci_lo\tci_hi\tiqr\tq1\tq3\tmean\tn_neg\tmin\tmax\n")
            for arm in sorted(byk):
                s = summarize(byk[arm])
                f.write(f"{arm}\t{s['n']}\t{s['median']:.4f}\t{fmt(s['ci_lo'])}\t{fmt(s['ci_hi'])}"
                        f"\t{s['iqr']:.4f}\t{s['q1']:.4f}\t{s['q3']:.4f}\t{s['mean']:.4f}"
                        f"\t{s['n_neg']}\t{s['amin']:.4f}\t{s['amax']:.4f}\n")
        return byk
    eff_table(bd_nat, f"{a.outdir}/arm_effects_native.tsv")
    eff_table(bd_bud, f"{a.outdir}/arm_effects_budget.tsv")

    want_pairs = {(q, s) for q in QML_LEVELS for s in SHP_LEVELS}
    int_nat = interactions(bd_nat, main_nat, want=None)
    int_bud = interactions(bd_bud, main_bud, want=None)
    with open(f"{a.outdir}/interactions_per_image.tsv", "w") as f:
        f.write("leg\tspeed\timage\tclass\tk1\tk2\tobserved\ta1\ta2\tadditive\tresid\tis_qml_x_shp\n")
        for leg, rows in (("native", int_nat), ("budget", int_bud)):
            for r in sorted(rows, key=lambda x: (x["k1"], x["k2"], x["image"])):
                isf = (r["k1"], r["k2"]) in want_pairs or (r["k2"], r["k1"]) in want_pairs
                f.write(f"{leg}\t{r['speed']}\t{r['image']}\t{cls_of.get(r['image'],'?')}\t{r['k1']}\t{r['k2']}"
                        f"\t{r['observed']:.4f}\t{r['a1']:.4f}\t{r['a2']:.4f}\t{r['additive']:.4f}"
                        f"\t{r['resid']:.4f}\t{int(isf)}\n")

    def pair_table(rows, path):
        byp = collections.defaultdict(list)
        for r in rows:
            byp[(r["k1"], r["k2"])].append(r["resid"])
        with open(path, "w") as f:
            f.write("k1\tk2\tis_qml_x_shp\tn\tmedian_resid\tci_lo\tci_hi\tiqr\tmean\t"
                    "n_neg\tfrac_neg\tp_sign_two_sided\tfrac_abs_ge_1pt\tmin\tmax\n")
            for (k1, k2), v in sorted(byp.items()):
                s = summarize(v)
                arr = np.asarray(v)
                p = binom_two_sided(s["n_neg"], s["n"])
                isf = int((k1, k2) in want_pairs or (k2, k1) in want_pairs)
                f.write(f"{k1}\t{k2}\t{isf}\t{s['n']}\t{s['median']:.4f}\t{fmt(s['ci_lo'])}\t{fmt(s['ci_hi'])}"
                        f"\t{s['iqr']:.4f}\t{s['mean']:.4f}\t{s['n_neg']}\t{s['n_neg']/s['n']:.4f}"
                        f"\t{p:.5g}\t{float((np.abs(arr)>=1.0).mean()):.4f}"
                        f"\t{s['amin']:.4f}\t{s['amax']:.4f}\n")
        return byp
    pn = pair_table(int_nat, f"{a.outdir}/interactions_native.tsv")
    pb = pair_table(int_bud, f"{a.outdir}/interactions_budget.tsv")

    # crop-vs-native A/B on the INTERACTION itself, paired per (pair, image)
    bud_idx = {(r["k1"], r["k2"], r["image"]): r["resid"] for r in int_bud}
    ab = []
    for r in int_nat:
        b = bud_idx.get((r["k1"], r["k2"], r["image"]))
        if b is not None:
            ab.append(dict(k1=r["k1"], k2=r["k2"], image=r["image"], native=r["resid"],
                           budget=b, delta=r["resid"] - b))
    with open(f"{a.outdir}/interaction_size_ab_per_image.tsv", "w") as f:
        f.write("k1\tk2\timage\tclass\tresid_native\tresid_budget\tdelta_native_minus_budget\tis_qml_x_shp\n")
        for r in sorted(ab, key=lambda x: (x["k1"], x["k2"], x["image"])):
            isf = int((r["k1"], r["k2"]) in want_pairs or (r["k2"], r["k1"]) in want_pairs)
            f.write(f"{r['k1']}\t{r['k2']}\t{r['image']}\t{cls_of.get(r['image'],'?')}"
                    f"\t{r['native']:.4f}\t{r['budget']:.4f}\t{r['delta']:.4f}\t{isf}\n")
    # ⛔ THE SIZE A/B IS ONLY MEANINGFUL ON THE 13 CROPPED REFERENCES. 19 of the
    # 32 budget-corpus references are NATIVE PASSTHROUGHS -- same pixels, hence
    # byte-identical encodes (gate 1 above: 171/171) -- so their native-minus-
    # budget delta is IDENTICALLY ZERO by construction. Pooling all 32 would
    # dilute a real size effect toward zero with 19 self-comparisons and read as
    # "size does not matter". The passthrough scope is emitted anyway, as a NULL
    # CONTROL: it must be exactly 0.0000 on every cell, and if it is not, the two
    # legs differ in configuration and nothing else in this section is safe.
    SCOPES = (("cropped", lambda im: im not in passthru),
              ("passthrough_NULLCTL", lambda im: im in passthru),
              ("all32_DILUTED", lambda im: True))
    with open(f"{a.outdir}/interaction_size_ab.tsv", "w") as f:
        f.write("scope\tk1\tk2\tis_qml_x_shp\tn\tmedian_delta\tci_lo\tci_hi\tn_neg\tp_sign_two_sided\tmin\tmax\n")
        for sname, keep in SCOPES:
            byp = collections.defaultdict(list)
            for r in ab:
                if keep(r["image"]):
                    byp[(r["k1"], r["k2"])].append(r["delta"])
            for (k1, k2), v in sorted(byp.items()):
                s = summarize(v)
                isf = int((k1, k2) in want_pairs or (k2, k1) in want_pairs)
                f.write(f"{sname}\t{k1}\t{k2}\t{isf}\t{s['n']}\t{s['median']:.4f}\t{fmt(s['ci_lo'])}\t{fmt(s['ci_hi'])}"
                        f"\t{s['n_neg']}\t{binom_two_sided(s['n_neg'], s['n']):.5g}"
                        f"\t{s['amin']:.4f}\t{s['amax']:.4f}\n")
        # pooled over all six factorial cells, per scope -- the headline A/B
        for sname, keep in SCOPES:
            v = [r["delta"] for r in ab if keep(r["image"]) and
                 ((r["k1"], r["k2"]) in want_pairs or (r["k2"], r["k1"]) in want_pairs)]
            if len(v) < 3:
                continue
            s = summarize(v)
            f.write(f"{sname}\tPOOLED_qml_x_shp\t-\t1\t{s['n']}\t{s['median']:.4f}\t{fmt(s['ci_lo'])}"
                    f"\t{fmt(s['ci_hi'])}\t{s['n_neg']}\t{binom_two_sided(s['n_neg'], s['n']):.5g}"
                    f"\t{s['amin']:.4f}\t{s['amax']:.4f}\n")

    # per content class, native leg, factorial cells only
    with open(f"{a.outdir}/interactions_native_by_class.tsv", "w") as f:
        f.write("class\tk1\tk2\tn\tmedian_resid\tci_lo\tci_hi\tn_neg\tmin\tmax\n")
        byc = collections.defaultdict(list)
        for r in int_nat:
            if (r["k1"], r["k2"]) in want_pairs or (r["k2"], r["k1"]) in want_pairs:
                byc[(cls_of.get(r["image"], "?"), r["k1"], r["k2"])].append(r["resid"])
        for (c, k1, k2), v in sorted(byc.items()):
            s = summarize(v)
            f.write(f"{c}\t{k1}\t{k2}\t{s['n']}\t{s['median']:.4f}\t{fmt(s['ci_lo'])}\t{fmt(s['ci_hi'])}"
                    f"\t{s['n_neg']}\t{s['amin']:.4f}\t{s['amax']:.4f}\n")
    # ...and the pooled-over-factorial view (all 6 qml x shp cells together)
    with open(f"{a.outdir}/factorial_pooled.tsv", "w") as f:
        f.write("leg\tscope\tn\tmedian_resid\tci_lo\tci_hi\tn_neg\tfrac_neg\tp_sign_two_sided\tmin\tmax\n")
        for leg, rows in (("native", int_nat), ("budget", int_bud)):
            # cropped-only view FIRST: the two legs share pixels on 19/32 refs,
            # so an all-32 native-vs-budget median compares 19 rows to themselves.
            vc = [r["resid"] for r in rows
                  if r["image"] not in passthru and
                  ((r["k1"], r["k2"]) in want_pairs or (r["k2"], r["k1"]) in want_pairs)]
            if len(vc) >= 3:
                sc = summarize(vc)
                f.write(f"{leg}\tqml_x_shp_CROPPED13\t{sc['n']}\t{sc['median']:.4f}\t{fmt(sc['ci_lo'])}"
                        f"\t{fmt(sc['ci_hi'])}\t{sc['n_neg']}\t{sc['n_neg']/sc['n']:.4f}"
                        f"\t{binom_two_sided(sc['n_neg'], sc['n']):.5g}\t{sc['amin']:.4f}\t{sc['amax']:.4f}\n")
            v = [r["resid"] for r in rows
                 if (r["k1"], r["k2"]) in want_pairs or (r["k2"], r["k1"]) in want_pairs]
            s = summarize(v)
            f.write(f"{leg}\tqml_x_shp_all6\t{s['n']}\t{s['median']:.4f}\t{fmt(s['ci_lo'])}\t{fmt(s['ci_hi'])}"
                    f"\t{s['n_neg']}\t{s['n_neg']/s['n']:.4f}\t{binom_two_sided(s['n_neg'], s['n']):.5g}"
                    f"\t{s['amin']:.4f}\t{s['amax']:.4f}\n")
            for c in sorted({cls_of.get(r["image"], "?") for r in rows}):
                v = [r["resid"] for r in rows
                     if (cls_of.get(r["image"], "?") == c) and
                     ((r["k1"], r["k2"]) in want_pairs or (r["k2"], r["k1"]) in want_pairs)]
                if len(v) < 3:
                    continue
                s = summarize(v)
                f.write(f"{leg}\tqml_x_shp_{c}\t{s['n']}\t{s['median']:.4f}\t{fmt(s['ci_lo'])}\t{fmt(s['ci_hi'])}"
                        f"\t{s['n_neg']}\t{s['n_neg']/s['n']:.4f}\t{binom_two_sided(s['n_neg'], s['n']):.5g}"
                        f"\t{s['amin']:.4f}\t{s['amax']:.4f}\n")

    # ================================================================= Q2 =====
    # zenrav1e (brsdr) vs svt (a0r), BUDGET corpus, matched in QUALITY space.
    rav = collections.defaultdict(list)   # (image, speed) -> [(ssim2, bytes)]
    for i in range(len(br["run"])):
        if br["run"][i] != "brsdr":
            continue
        s_, b_ = br[a.metric][i], br["bytes"][i]
        if s_ is None or b_ is None:
            continue
        rav[(br["image"][i], br["dial"][i])].append((s_, b_))
    svt = collections.defaultdict(list)
    for i in range(len(sa["run"])):
        if sa["run"][i] != "a0r":
            continue
        s_, b_ = sa[a.metric][i], sa["bytes"][i]
        if s_ is None or b_ is None:
            continue
        svt[(sa["image"][i], sa["speed"][i])].append((s_, b_))

    rav_all = collections.defaultdict(list)
    for (img, sp), pts in rav.items():
        rav_all[img].extend(pts)
    svt_all = collections.defaultdict(list)
    for (img, sp), pts in svt.items():
        svt_all[img].extend(pts)

    # (a) best-over-speeds RD frontier per backend -> BD-rate + per-band bytes
    rows = []
    for img in sorted(set(rav_all) & set(svt_all)):
        fr, fs = pooled_front(rav_all[img]), pooled_front(svt_all[img])
        bd = bd_rate(fr, fs)   # NEGATIVE = zenrav1e wins
        cr, cs = clip_front(fr, *PRODUCT_BAND), clip_front(fs, *PRODUCT_BAND)
        bdb = bd_rate(cr, cs) if (len(cr) >= 4 and len(cs) >= 4) else None
        rec = dict(image=img, cls=cls_of.get(img, "?"), px=crop_px.get(img),
                   bd_rav_vs_svt=bd, bd_banded=bdb,
                   n_rav_band=len(cr), n_svt_band=len(cs),
                   n_rav=len(fr), n_svt=len(fs),
                   rav_qmax=fr[-1][0] if fr else None, svt_qmax=fs[-1][0] if fs else None,
                   rav_qmin=fr[0][0] if fr else None, svt_qmin=fs[0][0] if fs else None)
        for name, lo, hi in QUALITY_BANDS:
            mid = 0.5 * (lo + hi)
            br_ = interp_bytes(fr, mid); bs_ = interp_bytes(fs, mid)
            rec[f"rav_bytes_{name}"] = br_
            rec[f"svt_bytes_{name}"] = bs_
            rec[f"ratio_{name}"] = (br_ / bs_) if (br_ and bs_) else None
        rows.append(rec)
    bandcols = [c for n, _, _ in QUALITY_BANDS for c in
                (f"rav_bytes_{n}", f"svt_bytes_{n}", f"ratio_{n}")]
    with open(f"{a.outdir}/backend_per_image.tsv", "w") as f:
        f.write("image\tclass\tpixels\tbd_rav_vs_svt\twinner_full\tbd_banded_30_95\twinner_banded\t"
                "n_rav_front\tn_svt_front\tn_rav_band\tn_svt_band\t"
                "rav_qmin\trav_qmax\tsvt_qmin\tsvt_qmax\t" + "\t".join(bandcols) + "\n")
        for r in rows:
            w = "" if r["bd_rav_vs_svt"] is None else ("zenrav1e" if r["bd_rav_vs_svt"] < 0 else "svt")
            wb = "" if r["bd_banded"] is None else ("zenrav1e" if r["bd_banded"] < 0 else "svt")
            f.write(f"{r['image']}\t{r['cls']}\t{r['px']}\t{fmt(r['bd_rav_vs_svt'])}\t{w}"
                    f"\t{fmt(r['bd_banded'])}\t{wb}"
                    f"\t{r['n_rav']}\t{r['n_svt']}\t{r['n_rav_band']}\t{r['n_svt_band']}"
                    f"\t{fmt(r['rav_qmin'],3)}\t{fmt(r['rav_qmax'],3)}"
                    f"\t{fmt(r['svt_qmin'],3)}\t{fmt(r['svt_qmax'],3)}\t"
                    + "\t".join(fmt(r[c], 1 if "bytes" in c else 4) for c in bandcols) + "\n")

    with open(f"{a.outdir}/backend_summary.tsv", "w") as f:
        f.write("scope\tstat\tn\tmedian\tci_lo\tci_hi\tn_rav_wins\tfrac_rav_wins\tp_sign_two_sided\tmin\tmax\n")
        def emit(scope, vals, lower_is_rav_win=True):
            v = [x for x in vals if x is not None and np.isfinite(x)]
            if len(v) < 3:
                f.write(f"{scope}\t-\t{len(v)}\tNOT-MEASURED (n<3)\t\t\t\t\t\t\t\n"); return
            s = summarize(v)
            nw = sum(1 for x in v if (x < 0 if lower_is_rav_win else x < 1.0))
            f.write(f"{scope}\tvalue\t{s['n']}\t{s['median']:.4f}\t{fmt(s['ci_lo'])}\t{fmt(s['ci_hi'])}"
                    f"\t{nw}\t{nw/s['n']:.4f}\t{binom_two_sided(nw, s['n']):.5g}"
                    f"\t{s['amin']:.4f}\t{s['amax']:.4f}\n")
        emit("bd_rav_vs_svt|all", [r["bd_rav_vs_svt"] for r in rows])
        for c in sorted({r["cls"] for r in rows}):
            emit(f"bd_rav_vs_svt|{c}", [r["bd_rav_vs_svt"] for r in rows if r["cls"] == c])
        emit("bd_banded_30_95|all", [r["bd_banded"] for r in rows])
        for c in sorted({r["cls"] for r in rows}):
            emit(f"bd_banded_30_95|{c}", [r["bd_banded"] for r in rows if r["cls"] == c])
        for n, _, _ in QUALITY_BANDS:
            emit(f"bytes_ratio_rav_over_svt|{n}", [r[f"ratio_{n}"] for r in rows], lower_is_rav_win=False)
            for c in sorted({r["cls"] for r in rows}):
                emit(f"bytes_ratio_rav_over_svt|{n}|{c}",
                     [r[f"ratio_{n}"] for r in rows if r["cls"] == c], lower_is_rav_win=False)

    # (b) per-SPEED cross-backend frontier (the operating-point view)
    with open(f"{a.outdir}/backend_per_image_speed.tsv", "w") as f:
        f.write("image\tclass\tbackend\tspeed\tn_pts\tqmin\tqmax\t"
                + "\t".join(f"bytes_{n}" for n, _, _ in QUALITY_BANDS) + "\n")
        for label, src in (("zenrav1e", rav), ("svt", svt)):
            for (img, sp), pts in sorted(src.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1)):
                fr = frontier(pts)
                cells = [fmt(interp_bytes(fr, 0.5 * (lo + hi)), 1) for _, lo, hi in QUALITY_BANDS]
                f.write(f"{img}\t{cls_of.get(img,'?')}\t{label}\t{sp}\t{len(fr)}"
                        f"\t{fmt(fr[0][0],3) if fr else ''}\t{fmt(fr[-1][0],3) if fr else ''}\t"
                        + "\t".join(cells) + "\n")

    # ---- Q2c: the THIRD AXIS -- join the speed instrument -------------------
    # LIMITATIONS, carried into the table itself rather than only into prose:
    #  * every alpha/beta is q45-SPECIFIC (instrument §7.2)
    #  * POOLED beta is unreliable (R^2 0.62-0.91); PER-SOURCE beta is clean
    #    (0.9929-0.9997) but exists for 5 sources only
    #  * beta is single-threaded WALL time (conservative ~27% for large threaded
    #    svt frames)
    #  * content-class splits (S1c) are NOT MEASURED -- never interpolated
    coef = {}
    for r in csv.DictReader(open(a.speed_coef), delimiter="\t"):
        coef[(r["backend"], int(r["speed"]))] = (float(r["alpha_ms"]), float(r["beta_ms_per_mp"]),
                                                 float(r["r2"]), float(r["per_source_r2_median"]))
    coef_src = {}
    for r in csv.DictReader(open(a.speed_coef_per_source), delimiter="\t"):
        coef_src[(r["backend"], int(r["speed"]), r["source"])] = (float(r["alpha_ms"]),
                                                                  float(r["beta_ms_per_mp"]),
                                                                  float(r["r2"]))
    def src_of(img):
        return img.split(".", 1)[0]

    with open(f"{a.outdir}/backend_speed_cost.tsv", "w") as f:
        f.write("image\tclass\tpixels\tmegapixels\tbackend\tspeed\tcoef_source\t"
                "alpha_ms\tbeta_ms_per_mp\tfit_r2\tpred_encode_ms\tq_regime\tnote\n")
        for label, be in (("zenrav1e", "zenravif"), ("svt", "svt-rs")):
            for img in sorted(set(rav_all) & set(svt_all)):
                px = crop_px.get(img)
                if px is None:
                    continue
                mp = px / 1e6
                for sp in range(1, 11):
                    ps = coef_src.get((be, sp, src_of(img)))
                    if ps is not None:
                        al, bt, r2 = ps; which = f"per-source:{src_of(img)}"
                        note = "per-source fit (clean, R2>=0.99)"
                    else:
                        c = coef.get((be, sp))
                        if c is None:
                            continue
                        al, bt, r2, _psr2 = c
                        which = "pooled"
                        note = ("POOLED fit -- unreliable, beta varies up to 24.33x with "
                                "content; use BANDED/qualitative only")
                    f.write(f"{img}\t{cls_of.get(img,'?')}\t{px}\t{mp:.4f}\t{label}\t{sp}\t{which}"
                            f"\t{al:.4f}\t{bt:.4f}\t{r2:.6f}\t{al + bt*mp:.2f}\tq45-ONLY"
                            f"\t{note}\n")

    # the three-axis operating picture: at each quality band, the cheapest-bytes
    # backend and the wall-time it costs at its own cheapest speed that reaches
    # the band. Speeds are enumerated; the time column is q45-anchored.
    with open(f"{a.outdir}/backend_three_axis.tsv", "w") as f:
        f.write("image\tclass\tmegapixels\tband\tband_mid_ssim2\t"
                "rav_best_bytes\trav_best_speed\trav_ms_at_best\t"
                "svt_best_bytes\tsvt_best_speed\tsvt_ms_at_best\t"
                "bytes_winner\tbytes_ratio_rav_over_svt\tms_ratio_rav_over_svt\tcoef_source\n")
        for img in sorted(set(rav_all) & set(svt_all)):
            px = crop_px.get(img)
            if px is None:
                continue
            mp = px / 1e6
            csrc = "per-source" if (("svt-rs", 1, src_of(img)) in coef_src) else "pooled"
            for name, lo, hi in QUALITY_BANDS:
                mid = 0.5 * (lo + hi)
                out = {}
                for label, be, src in (("rav", "zenravif", rav), ("svt", "svt-rs", svt)):
                    best = None
                    for (im, sp), pts in src.items():
                        if im != img or sp is None:
                            continue
                        b_ = interp_bytes(frontier(pts), mid)
                        if b_ is None:
                            continue
                        if best is None or b_ < best[0]:
                            best = (b_, sp)
                    if best is None:
                        out[label] = (None, None, None); continue
                    b_, sp = best
                    ps = coef_src.get((be, sp, src_of(img)))
                    if ps is not None:
                        al, bt = ps[0], ps[1]
                    else:
                        c = coef.get((be, sp))
                        al, bt = (c[0], c[1]) if c else (float("nan"), float("nan"))
                    out[label] = (b_, sp, al + bt * mp)
                rb, rs, rm = out["rav"]; sb, ss, sm = out["svt"]
                w = "" if (rb is None or sb is None) else ("zenrav1e" if rb < sb else "svt")
                f.write(f"{img}\t{cls_of.get(img,'?')}\t{mp:.4f}\t{name}\t{mid:.1f}"
                        f"\t{fmt(rb,1)}\t{rs if rs is not None else ''}\t{fmt(rm,1)}"
                        f"\t{fmt(sb,1)}\t{ss if ss is not None else ''}\t{fmt(sm,1)}"
                        f"\t{w}\t{fmt((rb/sb) if (rb and sb) else None)}"
                        f"\t{fmt((rm/sm) if (rm and sm) else None)}\t{csrc}\n")

    # ---- Q2d: the ISO-TIME view -- the one a product actually chooses on -----
    # The bytes-optimal speed is speed 1-2 for BOTH backends (slowest = smallest),
    # which is not an operating point anybody ships. So: at a wall-time budget T,
    # each backend may use its cheapest-bytes speed that (i) reaches the quality
    # band and (ii) fits the budget. The time model is the speed instrument's,
    # with ALL of its stated limitations -- q45-anchored, single-threaded wall
    # time, pooled beta where no per-source fit exists. Budgets are absolute ms
    # for the actual crop size, so a big image legitimately fails a small budget.
    TIME_BUDGETS_MS = (100.0, 500.0, 2000.0, 10000.0)
    def pred_ms(be, sp, img, mp):
        ps = coef_src.get((be, sp, src_of(img)))
        if ps is not None:
            return ps[0] + ps[1] * mp, "per-source"
        c = coef.get((be, sp))
        if c is None:
            return None, None
        return c[0] + c[1] * mp, "pooled"
    with open(f"{a.outdir}/backend_iso_time.tsv", "w") as f:
        f.write("image\tclass\tmegapixels\tband\tbudget_ms\t"
                "rav_bytes\trav_speed\trav_ms\tsvt_bytes\tsvt_speed\tsvt_ms\t"
                "winner\tbytes_ratio_rav_over_svt\tcoef_source\tq_regime\n")
        for img in sorted(set(rav_all) & set(svt_all)):
            px = crop_px.get(img)
            if px is None:
                continue
            mp = px / 1e6
            for name, lo, hi in QUALITY_BANDS:
                mid = 0.5 * (lo + hi)
                cand = {"rav": [], "svt": []}
                for label, be, src in (("rav", "zenravif", rav), ("svt", "svt-rs", svt)):
                    for (im, sp), pts in src.items():
                        if im != img or sp is None:
                            continue
                        b_ = interp_bytes(frontier(pts), mid)
                        if b_ is None:
                            continue
                        ms, which = pred_ms(be, sp, img, mp)
                        if ms is None:
                            continue
                        cand[label].append((b_, sp, ms, which))
                # TWO DIFFERENT FAILURES, never merged into one label: a backend
                # can fail to REACH the quality band at any speed (a capability
                # limit of its ladder) or reach it but not inside the time budget
                # (a cost limit). Conflating them reads a capability gap as a
                # speed problem -- MEASURED here: svt's default a0r ladder cannot
                # reach ssim2 90 on many references at ANY speed, which no time
                # budget would fix.
                for T in TIME_BUDGETS_MS:
                    pick, reach = {}, {}
                    for label in ("rav", "svt"):
                        reach[label] = bool(cand[label])
                        ok = [c for c in cand[label] if c[2] <= T]
                        pick[label] = min(ok, key=lambda c: c[0]) if ok else (None, None, None, None)
                    rb, rs, rm, rw = pick["rav"]; sb, ss, sm, sw = pick["svt"]
                    def why(label):
                        return "no-band-reach" if not reach[label] else "over-budget"
                    if rb is None and sb is None:
                        w = f"NEITHER (rav {why('rav')}, svt {why('svt')})"
                    elif rb is None:
                        w = f"svt (rav {why('rav')})"
                    elif sb is None:
                        w = f"zenrav1e (svt {why('svt')})"
                    else:
                        w = "zenrav1e" if rb < sb else "svt"
                    f.write(f"{img}\t{cls_of.get(img,'?')}\t{mp:.4f}\t{name}\t{T:.0f}"
                            f"\t{fmt(rb,1)}\t{rs if rs is not None else ''}\t{fmt(rm,1)}"
                            f"\t{fmt(sb,1)}\t{ss if ss is not None else ''}\t{fmt(sm,1)}"
                            f"\t{w}\t{fmt((rb/sb) if (rb and sb) else None)}"
                            f"\t{rw or sw or ''}\tq45-ONLY\n")

    with open(f"{a.outdir}/notes.json", "w") as f:
        json.dump(dict(
            metric=a.metric,
            quality_bands=[dict(name=n, lo=lo, hi=hi) for n, lo, hi in QUALITY_BANDS],
            brnat_q=list(BRNAT_Q), qml_levels=list(QML_LEVELS), shp_levels=list(SHP_LEVELS),
            gates=notes,
            n_rows=dict(brnat=sum(1 for x in br["run"] if x == "brnat"),
                        brsdr=sum(1 for x in br["run"] if x == "brsdr"),
                        a2=sum(1 for x in sa["run"] if x == "a2"),
                        a0r=sum(1 for x in sa["run"] if x == "a0r")),
            limitations=[
                "s6 only on brnat (svt_doe_pairwise has a single preset)",
                "m_ssim2 is the only corpus-wide scalar quality response; zensim is a feature vector",
                "speed coefficients are q45-SPECIFIC and single-threaded WALL time",
                "pooled beta unreliable (R2 0.62-0.91); per-source clean but 5 sources only",
                "content-class speed splits (S1c) NOT MEASURED -- never interpolated",
                "svt a0r covers speeds 1-7; zenrav1e brsdr covers 1-10",
            ]), f, indent=1)
    print(f"wrote tables to {a.outdir}")


if __name__ == "__main__":
    main()
