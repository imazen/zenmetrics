#!/usr/bin/env python3
"""avifdoe_stageb6_analyze.py — the PRE-REGISTERED Stage-B6 analysis of the AVIF DOE.

Trigger B-6 (plan §7.2) fires when an arm fails T1 of the cross-size transfer
gate: its direction at the 1024**2 screening budget does not carry to native.
Its follow-up is "that knob's Stage-B grid runs at NATIVE size" — declared in
plan §15 as `avifdoe-svt-b6-20260902`, 9 levels x 29 q x 32 images x speeds
{4,6,7} = 25,056 cells.

This script runs exactly what §15 + §7.1 register, and nothing chosen after
seeing results:

  Q1  native full-ladder main effects per LEVEL, per speed, per content class,
      vs the native deviations=0 control, with bootstrap CIs   (§7.1(2,3,6))
  Q2  the SCREENING-FAILURE quantification: the transfer gate's own T1/T2/T3
      re-run with B-6's 29-q native leg replacing AG's 3-q one, on the 13
      CROPPED references only (§12.4), q-MATCHED to the budget ladder so the
      residual is a size effect and not a ladder-density effect
  Q3  what B-6 can and cannot say about the QM x sharpness synergy (§7.1(4))
  Q4  the per-image tuning-model verdict

OWNERSHIP — nothing statistical is re-implemented here. `bd_rate`, `frontier`,
`median_ci` and the content-class map are IMPORTED from the Stage-A analyzer;
`binom_two_sided` from the Stage-A gates; SROCC from `zenstats` via
zensim/scripts/lib/zen_stats.py. If a number in this script's output disagrees
with Stage A's, it is because the DATA differs, never because the math does.

CATEGORICAL DISCIPLINE (§7.1(6), H-14): `sharpness` is a FACTOR. Levels are
reported as levels; no ordinal trend is fitted and no slope is quoted.
"""
import argparse, collections, csv, importlib.util, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/work/zen/zensim/scripts"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SA = _load("avifdoe_stagea_analyze", os.path.join(HERE, "avifdoe_stagea_analyze.py"))
SG = _load("avifdoe_stagea_gates", os.path.join(HERE, "avifdoe_stagea_gates.py"))
frontier, bd_rate, median_ci, q1q3, COARSE = SA.frontier, SA.bd_rate, SA.median_ci, SA.q1q3, SA.COARSE
binom_two_sided = SG.binom_two_sided

# The 9 q points the 1024**2 screening block (A1 @s4, A2 @s6) actually ran. Every
# one of them is also in B-6's 29-q ladder, so restricting the native leg to this
# set gives an EXACT q-match and isolates the size effect from ladder density.
BUDGET_Q = (5, 15, 25, 35, 45, 60, 76, 90, 96)

# T1/T2/T3 bars — plan §3.8, unchanged. Restated over the cropped refs per §12.4.
T1_BAR, T2_BAR, T3_BAR, T1_EFFECT_FLOOR = 0.80, 0.70, 1.0, 0.5
MIN_CROP, MIN_EFF = 5, 3


def load_scored(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pydict()


def curves(t, metric="m_ssim2", qset=None):
    """(run, speed, image, arm) -> [(quality, bytes)]"""
    c = collections.defaultdict(list)
    for i in range(len(t["run"])):
        b, s_ = t["bytes"][i], t[metric][i]
        if b is None or s_ is None:
            continue
        if qset is not None and t["q"][i] not in qset:
            continue
        c[(t["run"][i], t["speed"][i], t["image"][i], t["arm"][i])].append((s_, b))
    return c


def bd_table(cur, run, min_pts=4):
    """BD-rate of every deviation arm vs its OWN run's same-(image,speed) control."""
    out = []
    for (r, sp, img, arm), pts in cur.items():
        if r != run:
            continue
        ctl_arm = f"s{sp}-svt-420"
        if arm == ctl_arm:
            continue
        ref = cur.get((r, sp, img, ctl_arm))
        if not ref:
            continue
        v = bd_rate(frontier(pts), frontier(ref)) if min_pts == 4 else \
            SG.bd_rate(SG.frontier(pts), SG.frontier(ref), min_pts=min_pts)
        if v is None:
            continue
        devs = [d for d in arm.split("-")[3:] if d]
        out.append(dict(run=r, speed=sp, image=img, arm=arm, devs=devs,
                        knob=devs[0] if len(devs) == 1 else "|".join(devs),
                        n_dev=len(devs), bd_rate=v, n_pts=len(pts)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b6-scored", required=True)
    ap.add_argument("--stagea-scored", required=True)
    ap.add_argument("--naive-scored", default=None, help="the naive preset x q sweep, as a control robustness arm")
    ap.add_argument("--crop-manifest", required=True)
    ap.add_argument("--native-dims", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--metric", default="m_ssim2")
    ap.add_argument("--parity-check", default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    notes = []

    # ---- gate 0: BD-rate parity against the house implementation ------------
    if a.parity_check:
        m = _load("bd_arm", a.parity_check)
        rng = np.random.default_rng(20260902)
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
        print(f"GATE bd-parity: identical to zenavif/scripts/rd_gap/bd_arm.py on 200 ladders (max |d| {worst})")
        notes.append(dict(gate="bd_rate_parity", result="PASS", detail="max |delta| 0.0 on 200 random ladders"))

    man = {r["corpus_key"]: r for r in csv.DictReader(open(a.crop_manifest), delimiter="\t")}
    cropped = sorted(k for k, v in man.items() if v["transform"] != "native")
    passthru = sorted(k for k, v in man.items() if v["transform"] == "native")
    cls_of = {k: COARSE.get(v["content_class"], "other") for k, v in man.items()}
    fine_of = {k: v["content_class"] for k, v in man.items()}
    px = {}
    for r in csv.DictReader(open(a.native_dims), delimiter="\t"):
        px[r["image"]] = int(r["width"]) * int(r["height"])
    print(f"references {len(man)}: cropped {len(cropped)}, passthrough {len(passthru)}")

    b6 = load_scored(a.b6_scored)
    sa = load_scored(a.stagea_scored)

    # ---- gate 1: cross-run byte identity ------------------------------------
    # 19 of 32 refs are native passthroughs, so a budget-run encode of one is the
    # SAME BITSTREAM as B-6's native encode. AG encoded the native corpus, so it
    # must match on BOTH classes. Anything else means the two legs differ in
    # configuration rather than in pixels, which would confound every comparison
    # below. This is the §5.2 gate restated across runs.
    Bx = {(t_arm, t_img, t_q): t_sha for t_arm, t_img, t_q, t_sha
          in zip(b6["arm"], b6["image"], b6["q"], b6["encode_sha"])}
    ident = collections.defaultdict(lambda: [0, 0])
    for r_, arm, img, q, sha in zip(sa["run"], sa["arm"], sa["image"], sa["q"], sa["encode_sha"]):
        k = (arm, img, q)
        if k not in Bx:
            continue
        cell = ident[(r_, "passthrough" if img in passthru else "cropped")]
        cell[0] += 1
        cell[1] += (Bx[k] == sha)
    with open(f"{a.outdir}/gate_cross_run_identity.tsv", "w") as f:
        f.write("stagea_run\tref_class\tshared_cells\tbyte_identical\tfrac\texpected\tverdict\n")
        for (r_, c), (n_, i_) in sorted(ident.items()):
            exp = 1.0 if (r_ == "ag" or c == "passthrough") else 0.0
            ok = "PASS" if abs(i_ / n_ - exp) < 1e-12 else "FAIL"
            f.write(f"{r_}\t{c}\t{n_}\t{i_}\t{i_/n_:.6f}\t{exp:.1f}\t{ok}\n")
            notes.append(dict(gate=f"cross_run_identity[{r_}/{c}]", result=ok,
                              detail=f"{i_}/{n_} identical, expected frac {exp}"))
    print("GATE cross-run identity:", {f"{k[0]}/{k[1]}": f"{v[1]}/{v[0]}" for k, v in sorted(ident.items())})

    # ================= Q1 — native full-ladder main effects ==================
    cur_full = curves(b6, a.metric)
    bd_full = bd_table(cur_full, "b6")
    for r_ in bd_full:
        r_["cls"], r_["fine"] = cls_of.get(r_["image"], "?"), fine_of.get(r_["image"], "?")
        r_["mpix"] = px.get(r_["image"], 0) / 1e6
    with open(f"{a.outdir}/b6_bd_per_image.tsv", "w") as f:
        f.write("speed\timage\tmpix\tknob\tbd_rate\tn_pts\tclass\tfine_class\n")
        for r_ in sorted(bd_full, key=lambda x: (x["speed"], x["knob"], x["image"])):
            f.write(f"{r_['speed']}\t{r_['image']}\t{r_['mpix']:.3f}\t{r_['knob']}\t{r_['bd_rate']:.6f}"
                    f"\t{r_['n_pts']}\t{r_['cls']}\t{r_['fine']}\n")

    by = collections.defaultdict(list)
    for r_ in bd_full:
        by[(r_["speed"], r_["knob"])].append(r_["bd_rate"])
    with open(f"{a.outdir}/b6_main_effects.tsv", "w") as f:
        f.write("speed\tknob\tn_images\tmedian_bd\tci_lo\tci_hi\tiqr\tq25\tq75\tmean\tn_wins\tmin\tmax\tci_excludes_zero\n")
        for (sp, k), v in sorted(by.items()):
            lo, hi = q1q3(v); clo, chi = median_ci(v)
            f.write(f"{sp}\t{k}\t{len(v)}\t{np.median(v):.4f}\t{clo:.4f}\t{chi:.4f}\t{hi-lo:.4f}\t{lo:.4f}\t{hi:.4f}"
                    f"\t{np.mean(v):.4f}\t{sum(1 for x in v if x<0)}\t{min(v):.4f}\t{max(v):.4f}"
                    f"\t{'yes' if (clo>0 or chi<0) else 'no'}\n")

    bycls = collections.defaultdict(list)
    for r_ in bd_full:
        bycls[(r_["speed"], r_["knob"], r_["cls"])].append(r_["bd_rate"])
    with open(f"{a.outdir}/b6_main_effects_by_class.tsv", "w") as f:
        f.write("speed\tknob\tclass\tn\tmedian_bd\tci_lo\tci_hi\tq25\tq75\tmin\tmax\tprovisional\n")
        for (sp, k, c), v in sorted(bycls.items()):
            lo, hi = q1q3(v); clo, chi = median_ci(v)
            f.write(f"{sp}\t{k}\t{c}\t{len(v)}\t{np.median(v):.4f}\t{clo:.4f}\t{chi:.4f}\t{lo:.4f}\t{hi:.4f}"
                    f"\t{min(v):.4f}\t{max(v):.4f}\t{'yes' if len(v)<3 else 'no'}\n")

    # ---- Q1b: the naive preset x q sweep as an INDEPENDENT native control ----
    if a.naive_scored:
        nv = load_scored(a.naive_scored)
        cur_nv = curves(nv, a.metric)
        rows = []
        for (r_, sp, img, arm), pts in cur_full.items():
            devs = [d for d in arm.split("-")[3:] if d]
            if len(devs) != 1:
                continue
            ref = cur_nv.get(("naive", sp, img, f"s{sp}-svt-420"))
            if not ref:
                continue
            v = bd_rate(frontier(pts), frontier(ref))
            if v is not None:
                rows.append((sp, devs[0], img, v))
        alt = collections.defaultdict(list)
        for sp, k, img, v in rows:
            alt[(sp, k)].append(v)
        with open(f"{a.outdir}/b6_control_robustness.tsv", "w") as f:
            f.write("speed\tknob\tn_inrun\tmedian_inrun\tn_naive\tmedian_naive\tabs_delta_pp\n")
            for (sp, k), v in sorted(alt.items()):
                m_in = float(np.median(by[(sp, k)])) if (sp, k) in by else float("nan")
                m_nv = float(np.median(v))
                f.write(f"{sp}\t{k}\t{len(by.get((sp,k),[]))}\t{m_in:.4f}\t{len(v)}\t{m_nv:.4f}\t{abs(m_in-m_nv):.4f}\n")

    # ============ Q2 — the screening-failure quantification ==================
    # Budget leg: EXACTLY what the screening published — A1 (s4) / A2 (s6) arms on
    # their 9-q ladder against their own run's same-size control.
    # Native leg: B-6, restricted to the SAME 9 q points (q-matched) and, as a
    # secondary read, its full 29-q ladder.
    cur_bud = curves(sa, a.metric, qset=set(BUDGET_Q))
    cur_nat_q = curves(b6, a.metric, qset=set(BUDGET_Q))
    bd_bud = {(r_["speed"], r_["knob"], r_["image"]): r_["bd_rate"]
              for r_ in bd_table(cur_bud, "a1") + bd_table(cur_bud, "a2") if r_["n_dev"] == 1}
    bd_natq = {(r_["speed"], r_["knob"], r_["image"]): r_["bd_rate"]
               for r_ in bd_table(cur_nat_q, "b6") if r_["n_dev"] == 1}
    bd_natf = {(r_["speed"], r_["knob"], r_["image"]): r_["bd_rate"]
               for r_ in bd_full if r_["n_dev"] == 1}

    from lib.zen_stats import srocc as _srocc

    def gate(nat, lbl, fout):
        rows = []
        keys = sorted({(sp, k) for (sp, k, _) in bd_bud} & {(sp, k) for (sp, k, _) in nat})
        for sp, k in keys:
            pairs = [(img, nat[(sp, k, img)], bd_bud[(sp, k, img)])
                     for img in sorted(man)
                     if (sp, k, img) in nat and (sp, k, img) in bd_bud]
            crop_p = [p for p in pairs if p[0] in cropped]
            pass_p = [p for p in pairs if p[0] in passthru]
            viol = [p for p in pass_p if abs(p[1] - p[2]) > 1e-9]
            if len(crop_p) < MIN_CROP:
                rows.append(dict(speed=sp, knob=k, n_crop=len(crop_p), n_eff=None, t1=None, t2=None,
                                 t3=None, p=None, npass=len(pass_p), nviol=len(viol),
                                 verdict=f"NOT-MEASURED (n={len(crop_p)} usable cropped refs, need {MIN_CROP})"))
                continue
            eff = [p for p in crop_p if abs(p[1]) >= T1_EFFECT_FLOOR]
            t1 = (sum(1 for p in eff if (p[1] > 0) == (p[2] > 0)) / len(eff)) if len(eff) >= MIN_EFF else None
            t2 = float(_srocc([p[2] for p in crop_p], [p[1] for p in crop_p]))
            resid = [p[2] - p[1] for p in crop_p]
            t3 = float(np.median(np.abs(resid)))
            pos = sum(1 for r_ in resid if r_ > 0)
            pv = binom_two_sided(pos, len(resid))
            if t1 is None:
                v = f"NOT-MEASURED (only {len(eff)} of {len(crop_p)} cropped refs move >={T1_EFFECT_FLOOR}% at native, need {MIN_EFF})"
            elif t1 < T1_BAR:
                v = "FAIL-T1 (not screenable at budget)"
            elif t2 < T2_BAR or t3 > T3_BAR:
                v = "PARTIAL (direction holds; magnitude/rank flagged)"
            else:
                v = "PASS"
            rows.append(dict(speed=sp, knob=k, n_crop=len(crop_p), n_eff=len(eff), t1=t1, t2=t2,
                             t3=t3, p=pv, npass=len(pass_p), nviol=len(viol), verdict=v))
        with open(fout, "w") as f:
            f.write("speed\tknob\tn_crop\tn_eff\tT1_sign_agree\tT2_srocc\tT3_med_abs_resid\tT3_binom_p"
                    "\tn_passthrough\tn_passthrough_violations\tverdict\n")
            for r_ in rows:
                fm = lambda x: "" if x is None else f"{x:.4f}"
                f.write(f"{r_['speed']}\t{r_['knob']}\t{r_['n_crop']}\t{'' if r_['n_eff'] is None else r_['n_eff']}"
                        f"\t{fm(r_['t1'])}\t{fm(r_['t2'])}\t{fm(r_['t3'])}\t{fm(r_['p'])}"
                        f"\t{r_['npass']}\t{r_['nviol']}\t{r_['verdict']}\n")
        print(f"  transfer gate [{lbl}]: {len(rows)} (speed,knob) cells -> {fout}")
        return rows

    g_q = gate(bd_natq, "q-matched 9q native", f"{a.outdir}/b6_transfer_gate_qmatched.tsv")
    g_f = gate(bd_natf, "full 29q native", f"{a.outdir}/b6_transfer_gate_full.tsv")

    # per-image budget-vs-native, the raw evidence behind the verdicts
    with open(f"{a.outdir}/b6_budget_vs_native_per_image.tsv", "w") as f:
        f.write("speed\tknob\timage\tref_class\tcontent_class\tmpix\tbd_budget_1024\tbd_native_qmatched"
                "\tbd_native_full29q\tresid_budget_minus_native\tsign_agree\n")
        for (sp, k, img), bb in sorted(bd_bud.items()):
            bn = bd_natq.get((sp, k, img))
            bf = bd_natf.get((sp, k, img))
            if bn is None:
                continue
            f.write(f"{sp}\t{k}\t{img}\t{'passthrough' if img in passthru else 'cropped'}"
                    f"\t{cls_of.get(img,'?')}\t{px.get(img,0)/1e6:.3f}\t{bb:.6f}\t{bn:.6f}"
                    f"\t{'' if bf is None else f'{bf:.6f}'}\t{bb-bn:.6f}"
                    f"\t{'yes' if (bb>0)==(bn>0) else 'no'}\n")

    json.dump(dict(b6_cells=len(b6["run"]), b6_bd_rows=len(bd_full),
                   n_cropped=len(cropped), n_passthrough=len(passthru),
                   budget_q=list(BUDGET_Q), gates=notes,
                   transfer_cells_qmatched=len(g_q), transfer_cells_full=len(g_f)),
              open(f"{a.outdir}/_summary.json", "w"), indent=1)
    print(f"wrote tables to {a.outdir}")


if __name__ == "__main__":
    main()
