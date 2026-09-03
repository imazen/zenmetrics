#!/usr/bin/env python3
"""Fit the AVIF speed model from the timing instrument's pass TSVs.

Record: benchmarks/avif_speed_instrument_2026-09-03.md (protocol in section 2,
analysis plan in section 5).

RELATIONSHIP TO THE EXISTING FITTER, so this is not read as a duplicate:
`avifdoe_stagea_gates.py` fits `alpha + beta*pixels` for BYTES as a TWO-POINT
exact solve (native vs budget, two pixel counts per image).  This is a different
computation on a different quantity -- an over-determined least-squares fit for
TIME over a 5-7 rung ladder, plus the nonlinearity check that a 2-point solve
cannot make (two points always fit a line exactly, so a 2-point fit can never
detect that the relationship is not one).

Estimator, per the perf discipline:
  * per cell, the reported statistic is the MIN over passes (separate process
    starts, ASLR on -- min kills both interference and unlucky layout);
  * the pass-to-pass spread is reported BESIDE it as the drift control, never
    averaged away.

Usage:
  avif_speed_analyze.py --s1a run/s1a_pass*.tsv [--s1b run/s1b_pass*.tsv]
                        --out-dir DIR
"""
import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

# A linear fit this far from 1.0 is not describing the data.
R2_OK = 0.98

CROP_RE = re.compile(r"\.crop(\d+)\.png$")


def parse_rows(paths):
    """-> {(image, backend, speed, q): [encode_ms per pass]}"""
    acc = defaultdict(list)
    meta = {}
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                ms = r.get("encode_ms")
                if not ms:
                    continue
                img = os.path.basename(r["image_path"])
                kt = json.loads(r["knob_tuple_json"])
                key = (img, kt["backend"], int(kt["speed"]), int(r["q"]))
                acc[key].append(float(ms))
                meta[key] = int(r["encoded_bytes"]) if r.get("encoded_bytes") else None
    return acc, meta


def pixels_of(image):
    m = CROP_RE.search(image)
    if not m:
        return None
    side = int(m.group(1))
    return side * side


def ols(xs, ys):
    """Plain least squares y = a + b x.  Returns (a, b, r2)."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


def powerfit(xs, ys):
    """log-log fit y = c * x^g -> (c, g, r2_log).  Used only when the linear
    fit returns a NEGATIVE intercept, which is the linear model failing (the
    zensim perf work found per-pixel cost RISING with size), not a saving."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 2:
        return None
    lx = [p[0] for p in pts]
    ly = [p[1] for p in pts]
    f = ols(lx, ly)
    if not f:
        return None
    a, g, r2 = f
    return math.exp(a), g, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1a", nargs="+", required=True)
    ap.add_argument("--s1b", nargs="*", default=[])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    s1a_paths = [p for g in a.s1a for p in sorted(glob.glob(g))]
    acc, _ = parse_rows(s1a_paths)
    n_passes = max((len(v) for v in acc.values()), default=0)
    print(f"S1a: {len(acc)} cells from {len(s1a_paths)} pass file(s); "
          f"max passes per cell = {n_passes}")

    # ---- drift control: per-cell spread across passes ----------------------
    spreads = []
    for k, v in acc.items():
        if len(v) >= 2 and min(v) > 0:
            spreads.append((max(v) - min(v)) / min(v))
    spreads.sort()
    drift = {}
    if spreads:
        drift = {
            "n_cells_with_repeats": len(spreads),
            "median_rel_spread": spreads[len(spreads) // 2],
            "p90_rel_spread": spreads[int(0.90 * (len(spreads) - 1))],
            "max_rel_spread": spreads[-1],
        }
        print("drift control (max-min)/min across passes: "
              "median %.4f  p90 %.4f  max %.4f  (n=%d)"
              % (drift["median_rel_spread"], drift["p90_rel_spread"],
                 drift["max_rel_spread"], drift["n_cells_with_repeats"]))
    else:
        print("drift control: NOT MEASURABLE (no cell has >= 2 passes yet)")

    # ---- alpha + beta*pixels per (backend, speed) --------------------------
    by_arm = defaultdict(list)          # (backend, speed) -> [(px, ms, image)]
    by_arm_src = defaultdict(list)      # (backend, speed, source) -> [(px, ms)]
    for (img, backend, speed, q), v in acc.items():
        px = pixels_of(img)
        if px is None:
            continue
        ms = min(v)
        by_arm[(backend, speed)].append((px, ms, img))
        by_arm_src[(backend, speed, img.split(".")[0])].append((px, ms))

    rows = []
    for (backend, speed), pts in sorted(by_arm.items()):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        f = ols(xs, ys)
        if not f:
            continue
        alpha, beta, r2 = f
        rec = dict(backend=backend, speed=speed, n=len(pts),
                   alpha_ms=alpha, beta_ms_per_px=beta,
                   beta_ms_per_mp=beta * 1e6, r2=r2,
                   px_min=min(xs), px_max=max(xs))
        # TWO ways the linear model fails, and both must fire the power fit.
        #   * a NEGATIVE intercept -- the classic tell (per-pixel cost RISING
        #     with size makes the line dive under the origin); and
        #   * a poor R^2 with a positive intercept, which is the SAME failure
        #     wearing a respectable-looking alpha.  MEASURED on the first pass:
        #     every arm had alpha > 0 and R^2 ran 0.60-0.95, so an alpha<0-only
        #     test would have reported 20 of 20 arms as clean linear fits and
        #     shipped an intercept nobody should quote.
        rec["linear_model_failed"] = alpha < 0 or r2 < R2_OK
        rec["fail_reason"] = ("negative_intercept" if alpha < 0
                              else (f"r2<{R2_OK}" if r2 < R2_OK else ""))
        if rec["linear_model_failed"]:
            pf = powerfit(xs, ys)
            if pf:
                rec["power_c"], rec["power_gamma"], rec["power_r2_log"] = pf
        # WHICH failed -- the MODEL, or the POOLING?  A pooled fit over several
        # sources can look terrible while the model is exactly right, because
        # beta is a function of CONTENT and pooling averages incompatible
        # slopes.  MEASURED on pass 1: pooled R^2 ran 0.60-0.95 while the SAME
        # arms fit per source at R^2 0.989-1.0000 (median 0.999), and beta
        # spanned 5.4x between two photographs at svt speed 1.  Reporting only
        # the pooled number would have condemned a model that fits ~perfectly.
        src_r2 = []
        src_beta = []
        for (bk, sp, src), spts in by_arm_src.items():
            if bk != backend or sp != speed or len(spts) < 5:
                continue
            sf = ols([q[0] for q in spts], [q[1] for q in spts])
            if sf:
                src_r2.append(sf[2])
                src_beta.append(sf[1])
        if src_r2:
            src_r2.sort()
            rec["per_source_r2_median"] = src_r2[len(src_r2) // 2]
            rec["n_source_fits"] = len(src_r2)
            if src_beta and min(src_beta) > 0:
                rec["beta_spread_across_sources"] = max(src_beta) / min(src_beta)
            # The diagnosis the caller actually needs.
            if rec["linear_model_failed"] and rec["per_source_r2_median"] >= R2_OK:
                rec["fail_reason"] = "POOLING_NOT_MODEL (per-source fit is clean)"
        rows.append(rec)

    out_tsv = os.path.join(a.out_dir, "speed_alpha_beta.tsv")
    cols = ["backend", "speed", "n", "px_min", "px_max", "alpha_ms",
            "beta_ms_per_mp", "r2", "per_source_r2_median", "n_source_fits",
            "beta_spread_across_sources", "linear_model_failed", "fail_reason",
            "power_c", "power_gamma", "power_r2_log"]
    with open(out_tsv, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(
                ("" if r.get(c) is None else
                 (f"{r[c]:.6g}" if isinstance(r.get(c), float) else str(r.get(c, ""))))
                for c in cols) + "\n")
    print(f"wrote {out_tsv}  ({len(rows)} arms)")

    # ---- per-source beta: content is a factor, not noise --------------------
    src_tsv = os.path.join(a.out_dir, "speed_alpha_beta_per_source.tsv")
    with open(src_tsv, "w") as fh:
        fh.write("backend\tspeed\tsource\tn\talpha_ms\tbeta_ms_per_mp\tr2\n")
        for (backend, speed, src), pts in sorted(by_arm_src.items()):
            f = ols([p[0] for p in pts], [p[1] for p in pts])
            if not f:
                continue
            al, be, r2 = f
            fh.write(f"{backend}\t{speed}\t{src}\t{len(pts)}\t{al:.6g}"
                     f"\t{be*1e6:.6g}\t{r2:.6g}\n")
    print(f"wrote {src_tsv}")

    # ---- q-flatness verdict (S1b) ------------------------------------------
    qv = {}
    if a.s1b:
        s1b_paths = [p for g in a.s1b for p in sorted(glob.glob(g))]
        bacc, _ = parse_rows(s1b_paths)
        by_cell = defaultdict(dict)   # (img, backend, speed) -> {q: min_ms}
        for (img, backend, speed, q), v in bacc.items():
            by_cell[(img, backend, speed)][q] = min(v)
        per_backend = defaultdict(list)
        qrows = []
        for (img, backend, speed), qmap in sorted(by_cell.items()):
            if len(qmap) < 2:
                continue
            lo, hi = min(qmap.values()), max(qmap.values())
            rel = (hi - lo) / lo if lo > 0 else float("nan")
            qrows.append((img, backend, speed, len(qmap), lo, hi, rel))
            per_backend[backend].append(rel)
        qtsv = os.path.join(a.out_dir, "q_flatness.tsv")
        with open(qtsv, "w") as fh:
            fh.write("image\tbackend\tspeed\tn_q\tmin_ms\tmax_ms\trel_spread\n")
            for r in qrows:
                fh.write("\t".join(str(x) for x in r[:4]) +
                         f"\t{r[4]:.4f}\t{r[5]:.4f}\t{r[6]:.6f}\n")
        print(f"wrote {qtsv}  ({len(qrows)} cells)")
        for backend, v in sorted(per_backend.items()):
            v.sort()
            qv[backend] = {"n": len(v), "median": v[len(v) // 2], "max": v[-1]}
            print("q-flatness %-9s median %.4f  max %.4f  (n=%d)"
                  % (backend, qv[backend]["median"], qv[backend]["max"], len(v)))
    else:
        print("q-flatness: NOT MEASURED (no --s1b files given)")

    with open(os.path.join(a.out_dir, "summary.json"), "w") as fh:
        json.dump({"n_passes_max": n_passes, "drift": drift,
                   "arms": rows, "q_flatness": qv}, fh, indent=2)


if __name__ == "__main__":
    main()
