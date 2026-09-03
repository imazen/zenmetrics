#!/usr/bin/env python3
"""avifhbd_t2_analyze.py — the pre-registered Track-T2 (HDR-10 RD) reduction.

Answers ONLY the two questions frozen in `benchmarks/avif_hdr_arm_plan_2026-09-02.md`
§5.1 for this track:

  Q5  What is the rate/quality curve of 10-bit AVIF on HDR stills?  (T2-a — a
      BASELINE. No PASS/FAIL bar, exactly like the gain-map census.)
  Q6  How do the two wired HDR AVIF arms differ?  (T2-b — a CONTRAST, not a
      controlled comparison: the arms differ in backend AND chroma AND matrix.)

STATS / BD-RATE OWNERSHIP. Nothing is re-implemented here. `frontier`, `bd_rate`,
`median_ci` and `q1q3` are IMPORTED from `avifdoe_stagea_analyze.py`, the DOE's
BD-rate owner (itself parity-gated against zenavif `scripts/rd_gap/bd_arm.py`);
`--parity-check` runs that same assertion. Rank/correlation statistics are not
computed here at all — that is `zenstats`' surface.

SIGN CONVENTION (inherited): NEGATIVE BD-rate = the test arm needs FEWER bits at
matched quality = the test arm WINS.

WHAT THIS TOOL REFUSES TO DO.
  * It will not map a zenav1-svt `preset` onto a zenavif `speed`. There is no
    such mapping across backends; the dials are separate axes and are reported
    as a full arm x arm matrix plus a per-backend Pareto ENVELOPE.
  * It will not draw a content-class conclusion without the primaries cross-tab
    (plan §4.3 registered restriction) — `--picks` is required and the cross-tab
    is always written.
  * It will not compute a BD-rate from fewer than 4 ladder points (the owner's
    own guard); such a cell is NOT-MEASURED, never a zero.
"""
import argparse, collections, importlib.util, json, os, re, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "avifdoe_stagea_analyze", os.path.join(_HERE, "avifdoe_stagea_analyze.py"))
_owner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_owner)
frontier, bd_rate, median_ci, q1q3 = (
    _owner.frontier, _owner.bd_rate, _owner.median_ci, _owner.q1q3)

BACKEND_OF = {"zenav1svt": "zenav1-svt", "zenavif": "zenavif"}

# The imazen-26 variant convention ends every name with `_<W>x<H>`. Parsed
# STRICTLY: a name that does not match yields no pixel count and the bpp column
# is left empty rather than filled with a guess.
_DIMS = re.compile(r"_(\d{2,5})x(\d{2,5})$")


def pixels_of(variant):
    m = _DIMS.search(variant)
    return int(m.group(1)) * int(m.group(2)) if m else None


def backend_of(arm):
    for tag, be in BACKEND_OF.items():
        if arm and f"-{tag}-" in arm:
            return be
    return "?"


def load_picks(path):
    import csv
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[r["variant"]] = dict(
                category=r.get("category", "?"),
                primaries=r.get("primaries_name") or r.get("primaries", "?"),
                peak_nits=float(r["display_peak_nits"]) if r.get("display_peak_nits") else None,
                headroom=float(r["hdr_headroom_stops"]) if r.get("hdr_headroom_stops") else None,
                eff_depth=r.get("effective_bit_depth", "?"),
            )
    return out


def img_key(image_path):
    """The harvest's `image` is the reference basename; picks key on `variant`."""
    base = image_path.rsplit("/", 1)[-1]
    for suf in (".hdr.png", ".png"):
        if base.endswith(suf):
            return base[: -len(suf)]
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="avifdoe_harvest.py parquet (T2 runs)")
    ap.add_argument("--picks", required=True, help="benchmarks/avif_hdr_t2_picks_k16_*.tsv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--metric", default="m_ssim2", help="quality column (higher=better)")
    ap.add_argument("--second-metric", default="m_zensim_score")
    ap.add_argument("--ref-arm", default=None,
                    help="arm used as the BD-rate reference (default: the svt arm whose "
                         "preset is the plan's default stratum, p4)")
    ap.add_argument("--parity-check", default=None,
                    help="path to zenavif scripts/rd_gap/bd_arm.py — assert BD-rate agreement")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    if a.parity_check:
        spec = importlib.util.spec_from_file_location("bd_arm", a.parity_check)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        rng = np.random.default_rng(20260903)
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
        print(f"PARITY: bd_rate == zenavif/scripts/rd_gap/bd_arm.py exactly (max |delta| {worst})")

    import pyarrow.parquet as pq
    t = pq.read_table(a.scored).to_pydict()
    picks = load_picks(a.picks)
    n = len(t["run"])

    rows, unscored = [], 0
    for i in range(n):
        if t[a.metric][i] is None or t["bytes"][i] is None:
            unscored += 1
            continue
        img = img_key(t["image"][i])
        meta = picks.get(img, {})
        rows.append(dict(run=t["run"][i], arm=t["arm"][i], dial=t["dial"][i],
                         dial_kind=t["dial_kind"][i], codec=t["codec"][i],
                         backend=backend_of(t["arm"][i]),
                         image=img, q=t["q"][i], bytes=t["bytes"][i],
                         qual=t[a.metric][i],
                         qual2=t.get(a.second_metric, [None] * n)[i],
                         category=meta.get("category", "?"),
                         primaries=meta.get("primaries", "?"),
                         peak_nits=meta.get("peak_nits")))
    print(f"scored rows {len(rows)} of {n} cells (unscored {unscored})")
    if not rows:
        sys.exit("no scored rows — refusing to write an empty analysis")

    # ---- G0.5: the primaries x content cross-tab, ALWAYS written ------------
    seen = {}
    for r in rows:
        seen[r["image"]] = (r["category"], r["primaries"])
    ct = collections.Counter(seen.values())
    with open(f"{a.outdir}/t2_primaries_content_crosstab.tsv", "w") as f:
        f.write("category\tprimaries\tn_refs\n")
        for (c, p), v in sorted(ct.items()):
            f.write(f"{c}\t{p}\t{v}\n")

    # ---- Q5: the RD curves --------------------------------------------------
    with open(f"{a.outdir}/t2_rd_points.tsv", "w") as f:
        f.write("run\tbackend\tarm\tdial_kind\tdial\timage\tq\tbytes\t"
                f"{a.metric}\t{a.second_metric}\tcategory\tprimaries\n")
        for r in sorted(rows, key=lambda x: (x["run"], x["arm"], x["image"], x["q"])):
            q2 = "" if r["qual2"] is None else f"{r['qual2']:.6f}"
            f.write(f"{r['run']}\t{r['backend']}\t{r['arm']}\t{r['dial_kind']}\t{r['dial']}"
                    f"\t{r['image']}\t{r['q']}\t{r['bytes']}\t{r['qual']:.6f}\t{q2}"
                    f"\t{r['category']}\t{r['primaries']}\n")

    # ---- Q5: the reference RD curve, per arm x q -----------------------------
    # This IS the T2-a deliverable: the baseline every future HDR arm differences
    # against. Medians over the 16 references, with bpp where the variant name
    # carries dimensions.
    byaq = collections.defaultdict(list)
    for r in rows:
        byaq[(r["arm"], r["q"])].append(r)
    with open(f"{a.outdir}/t2_rd_baseline.tsv", "w") as f:
        f.write(f"arm\tbackend\tdial_kind\tdial\tq\tn_refs\tmedian_bytes\tmedian_bpp"
                f"\tmedian_{a.metric}\tmedian_{a.second_metric}\n")
        for (arm, q), v in sorted(byaq.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            b = [x["bytes"] for x in v]
            bpp = [8.0 * x["bytes"] / pixels_of(x["image"])
                   for x in v if pixels_of(x["image"])]
            q1 = [x["qual"] for x in v]
            q2 = [x["qual2"] for x in v if x["qual2"] is not None]
            fields = [arm, backend_of(arm), v[0]["dial_kind"], v[0]["dial"], q, len(v),
                      f"{np.median(b):.0f}",
                      f"{np.median(bpp):.4f}" if bpp else "",
                      f"{np.median(q1):.4f}",
                      f"{np.median(q2):.4f}" if q2 else ""]
            f.write("\t".join(str(x) for x in fields) + "\n")

    curves = collections.defaultdict(list)          # (arm,image) -> [(qual,bytes)]
    for r in rows:
        curves[(r["arm"], r["image"])].append((r["qual"], r["bytes"]))

    # per-arm ladder summary: coverage + the achieved-quality span, which is what
    # bounds every BD-rate below.
    arm_imgs = collections.defaultdict(set)
    for (arm, img) in curves:
        arm_imgs[arm].add(img)
    with open(f"{a.outdir}/t2_arm_summary.tsv", "w") as f:
        f.write("arm\tbackend\tn_images\tn_points_median\tqual_min_median\tqual_max_median\t"
                "bytes_min_median\tbytes_max_median\n")
        for arm in sorted(arm_imgs):
            per = [curves[(arm, im)] for im in sorted(arm_imgs[arm])]
            npts = [len(p) for p in per]
            qmin = [min(q for q, _ in p) for p in per]
            qmax = [max(q for q, _ in p) for p in per]
            bmin = [min(b for _, b in p) for p in per]
            bmax = [max(b for _, b in p) for p in per]
            f.write(f"{arm}\t{backend_of(arm)}\t{len(per)}\t{np.median(npts):.0f}"
                    f"\t{np.median(qmin):.3f}\t{np.median(qmax):.3f}"
                    f"\t{np.median(bmin):.0f}\t{np.median(bmax):.0f}\n")

    arms = sorted(arm_imgs)
    ref_arm = a.ref_arm or next((x for x in arms if x.startswith("p4-")), arms[0])
    if ref_arm not in arms:
        sys.exit(f"--ref-arm {ref_arm} not present; have {arms}")

    # ---- Q6 (a): full arm x arm BD-rate matrix, per image -------------------
    per_pair = collections.defaultdict(list)
    notmeasured = collections.Counter()
    with open(f"{a.outdir}/t2_bd_per_image.tsv", "w") as f:
        f.write("test_arm\tref_arm\timage\tbd_rate\tn_test_pts\tn_ref_pts\tcategory\tprimaries\n")
        for test in arms:
            for ref in arms:
                if test == ref:
                    continue
                for img in sorted(arm_imgs[test] & arm_imgs[ref]):
                    tp, rp = curves[(test, img)], curves[(ref, img)]
                    v = bd_rate(frontier(tp), frontier(rp))
                    if v is None:
                        notmeasured[(test, ref)] += 1
                        continue
                    per_pair[(test, ref)].append(v)
                    meta = seen.get(img, ("?", "?"))
                    f.write(f"{test}\t{ref}\t{img}\t{v:.6f}\t{len(tp)}\t{len(rp)}"
                            f"\t{meta[0]}\t{meta[1]}\n")
    with open(f"{a.outdir}/t2_bd_matrix.tsv", "w") as f:
        f.write("test_arm\tref_arm\tn_images\tmedian_bd\tci_lo\tci_hi\tq25\tq75\tmean\t"
                "n_test_wins\tmin\tmax\tn_not_measured\n")
        for (test, ref), v in sorted(per_pair.items()):
            lo, hi = q1q3(v); clo, chi = median_ci(v)
            f.write(f"{test}\t{ref}\t{len(v)}\t{np.median(v):.4f}\t{clo:.4f}\t{chi:.4f}"
                    f"\t{lo:.4f}\t{hi:.4f}\t{np.mean(v):.4f}"
                    f"\t{sum(1 for x in v if x < 0)}\t{min(v):.4f}\t{max(v):.4f}"
                    f"\t{notmeasured.get((test, ref), 0)}\n")

    # ---- Q6 (b): per-BACKEND Pareto envelope --------------------------------
    # The product-level contrast that does NOT require a preset<->speed mapping:
    # pool every dial setting a backend offers on that image and take the front.
    env = collections.defaultdict(list)             # (backend,image) -> points
    for r in rows:
        env[(r["backend"], r["image"])].append((r["qual"], r["bytes"]))
    backends = sorted({b for b, _ in env})
    env_pairs = collections.defaultdict(list)
    with open(f"{a.outdir}/t2_envelope_per_image.tsv", "w") as f:
        f.write("test_backend\tref_backend\timage\tbd_rate\tcategory\tprimaries\n")
        for tb in backends:
            for rb in backends:
                if tb == rb:
                    continue
                imgs = {i for b, i in env if b == tb} & {i for b, i in env if b == rb}
                for img in sorted(imgs):
                    v = bd_rate(frontier(env[(tb, img)]), frontier(env[(rb, img)]))
                    if v is None:
                        continue
                    env_pairs[(tb, rb)].append(v)
                    meta = seen.get(img, ("?", "?"))
                    f.write(f"{tb}\t{rb}\t{img}\t{v:.6f}\t{meta[0]}\t{meta[1]}\n")
    with open(f"{a.outdir}/t2_envelope_contrast.tsv", "w") as f:
        f.write("test_backend\tref_backend\tn_images\tmedian_bd\tci_lo\tci_hi\tq25\tq75\t"
                "n_test_wins\tmin\tmax\n")
        for (tb, rb), v in sorted(env_pairs.items()):
            lo, hi = q1q3(v); clo, chi = median_ci(v)
            f.write(f"{tb}\t{rb}\t{len(v)}\t{np.median(v):.4f}\t{clo:.4f}\t{chi:.4f}"
                    f"\t{lo:.4f}\t{hi:.4f}\t{sum(1 for x in v if x < 0)}"
                    f"\t{min(v):.4f}\t{max(v):.4f}\n")

    # ---- per-category, with primaries carried (never dropped) ---------------
    # Re-read the per-image file rather than re-deriving: one source of truth for
    # every envelope number, and the class labels are already joined there.
    bycat = collections.defaultdict(list)
    import csv as _csv
    with open(f"{a.outdir}/t2_envelope_per_image.tsv") as f:
        for r in _csv.DictReader(f, delimiter="\t"):
            bycat[(r["test_backend"], r["ref_backend"], r["category"], r["primaries"])].append(
                float(r["bd_rate"]))
    with open(f"{a.outdir}/t2_envelope_by_class.tsv", "w") as f:
        f.write("test_backend\tref_backend\tcategory\tprimaries\tn\tmedian_bd\tmin\tmax\n")
        for k, v in sorted(bycat.items()):
            f.write(f"{k[0]}\t{k[1]}\t{k[2]}\t{k[3]}\t{len(v)}\t{np.median(v):.4f}"
                    f"\t{min(v):.4f}\t{max(v):.4f}\n")

    json.dump(dict(scored_rows=len(rows), unscored_cells=unscored, arms=arms,
                   ref_arm=ref_arm, backends=backends,
                   images=len(seen), metric=a.metric, second_metric=a.second_metric,
                   not_measured_pairs={f"{k[0]}|{k[1]}": v for k, v in notmeasured.items()}),
              open(f"{a.outdir}/_t2_summary.json", "w"), indent=1)
    print(f"wrote T2 tables to {a.outdir}")


if __name__ == "__main__":
    main()
