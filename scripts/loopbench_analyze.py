#!/usr/bin/env python3
"""Task #163 — fit the encoder-search-loop cost model from loop-wall TSVs.

Reads one or more `loop-wall` output TSVs (repeat runs of the same grid) and
emits the analysis TSV that backs the recommendation:

  * per (metric, mode, size): the MINIMUM across runs.
    Min, not mean: every run on this box shares the machine with other agents,
    so each sample is `true_cost + contention >= true_cost`. The minimum is the
    least-contended observation and the only estimator that converges to the
    uncontended cost as runs accumulate. A mean would converge to the
    contention level, which is not a property of the code.

  * an ordinary-least-squares fit of `ms = alpha + beta * megapixels` per
    (metric, mode), reporting BOTH terms. CLAUDE.md sweep discipline: a
    "ms/MP" figure with no intercept is meaningless, because at 64x64 the
    intercept is the entire cost and at 4096x4096 the slope is.

  * the per-candidate cost and the break-even candidate count N* at which a
    warm reference beats repeating the cold one-shot:

        cold(N)  = N * oneshot
        warm(N)  = precompute + N * warm_per_candidate
        N*       = precompute / (oneshot - warm_per_candidate)

    N* is the smallest integer N with warm(N) < cold(N).

Usage: loopbench_analyze.py <out_tsv> <in_tsv> [<in_tsv> ...]
"""

import csv
import sys
from collections import defaultdict

DIMS = {
    "64": (64, 64),
    "128": (128, 128),
    "256": (256, 256),
    "512": (512, 512),
    "1024": (1024, 1024),
    "2K": (2048, 2048),
    "4096": (4096, 4096),
}
ORDER = list(DIMS)
METRICS = ["butter", "zensim", "ssim2"]
MODES = [
    "oneshot",
    "precompute",
    "warm_strided",
    "warm_zeroalloc",
    "warm_tight_plus_copy",
    "copy_only",
    "loop_n5",
]


def load(paths):
    """min-of-runs per (metric, size, mode); parity rows collected separately."""
    best = defaultdict(dict)
    nruns = defaultdict(lambda: defaultdict(int))
    parity = defaultdict(dict)
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                key = (r["metric"], r["size_label"])
                if r["mode"].startswith("PARITY"):
                    parity[key][r["mode"][7:]] = r["score"]
                    continue
                v = float(r["mean_ms"])
                cur = best[key].get(r["mode"])
                if cur is None or v < cur:
                    best[key][r["mode"]] = v
                nruns[key][r["mode"]] += 1
    return best, nruns, parity


def ols(xs, ys):
    """Least squares y = a + b x. Returns (a, b, r2)."""
    n = len(xs)
    if n < 2:
        return (float("nan"), float("nan"), float("nan"))
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return (float("nan"), float("nan"), float("nan"))
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return (a, b, r2)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out_path, in_paths = sys.argv[1], sys.argv[2:]
    best, nruns, parity = load(in_paths)

    rows = []
    for m in METRICS:
        for s in ORDER:
            d = best.get((m, s))
            if not d:
                continue
            w, h = DIMS[s]
            mp = w * h / 1.0e6
            one = d.get("oneshot")
            pre = d.get("precompute")
            warm = d.get("warm_strided")
            ing = d.get("copy_only", 0.0)
            n5 = d.get("loop_n5")
            alt = d.get("warm_zeroalloc", d.get("warm_tight_plus_copy"))
            # Additivity check: does the measured 5-candidate loop match
            # precompute + 5 * per-candidate? If not, the cost model is wrong
            # and every projection built on it would be wrong too.
            pred = (pre + 5 * warm) if (pre is not None and warm is not None) else None
            addit = (100.0 * (n5 - pred) / pred) if (n5 and pred) else None
            # Break-even candidate count.
            nstar = ""
            if one and warm is not None and pre is not None:
                gain = one - warm
                if gain <= 0:
                    # The warm compare is not cheaper than a cold one-shot at
                    # this size — no candidate count makes the precompute pay.
                    nstar = "never"
                else:
                    # smallest N with pre + N*warm < N*one  <=>  N > pre/gain
                    n = 1
                    while pre + n * warm >= n * one:
                        n += 1
                        if n > 10000:
                            break
                    nstar = str(n)
            rows.append(
                dict(
                    metric=m,
                    size_label=s,
                    w=w,
                    h=h,
                    mp=f"{mp:.4f}",
                    oneshot_ms=f"{one:.5f}" if one else "-",
                    precompute_ms=f"{pre:.5f}" if pre is not None else "-",
                    warm_per_cand_ms=f"{warm:.5f}" if warm is not None else "-",
                    ingress_ms=f"{ing:.5f}",
                    ingress_pct_of_warm=f"{100.0 * ing / warm:.2f}" if warm else "-",
                    warm_alt_ms=f"{alt:.5f}" if alt is not None else "-",
                    warm_over_oneshot=f"{warm / one:.4f}" if (warm is not None and one) else "-",
                    loop_n5_measured_ms=f"{n5:.5f}" if n5 else "-",
                    loop_n5_predicted_ms=f"{pred:.5f}" if pred else "-",
                    loop_n5_model_err_pct=f"{addit:+.2f}" if addit is not None else "-",
                    breakeven_candidates=nstar,
                    n_runs=nruns[(m, s)].get("warm_strided", 0),
                )
            )

    with open(out_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        wtr.writeheader()
        wtr.writerows(rows)

    # ---- fits + parity to stdout (captured into the .md writeup) ----
    # `alpha + beta*MP` fitted over the WHOLE ladder is dominated by the 16 MP
    # point and comes back with a NEGATIVE intercept for every cell — an
    # artifact, not a fixed cost. These kernels are close to pure per-pixel
    # with mild superlinearity at the top end (cache), so unweighted OLS tilts
    # the line and pushes alpha below zero.
    #
    # So report two things instead of one misleading line:
    #   * `alpha` fitted on the SMALL end only (64..512), where a per-call
    #     fixed cost is actually observable;
    #   * `beta` fitted on the LARGE end only (1024..4096), the asymptotic
    #     per-pixel slope;
    #   * and the measured ms/MP at every size, so the reader can see the
    #     intercept's influence decay instead of trusting a single number.
    SMALL = ["64", "128", "256", "512"]
    LARGE = ["1024", "2K", "4096"]

    def fit(m, mode, sizes):
        xs, ys = [], []
        for s in sizes:
            d = best.get((m, s), {})
            if mode in d:
                w, h = DIMS[s]
                xs.append(w * h / 1.0e6)
                ys.append(d[mode])
        return ols(xs, ys) if len(xs) >= 2 else (float("nan"),) * 3

    print("== cost model: alpha from the small end (64-512), beta from the large end (1024-4096) ==")
    print(
        f"{'metric':8}{'mode':>22}{'alpha_ms':>10}{'r2_small':>10}"
        f"{'beta_ms/MP':>12}{'r2_large':>10}{'alpha_as_%_of_64sq':>19}"
    )
    for m in METRICS:
        for mode in MODES:
            a_s, b_s, r2s = fit(m, mode, SMALL)
            a_l, b_l, r2l = fit(m, mode, LARGE)
            base = best.get((m, "64"), {}).get(mode)
            pct = f"{100.0 * a_s / base:.1f}" if (base and a_s == a_s) else "-"
            if a_s != a_s:
                continue
            print(
                f"{m:8}{mode:>22}{a_s:10.4f}{r2s:10.4f}{b_l:12.3f}{r2l:10.4f}{pct:>19}"
            )

    print("\n== measured ms per megapixel at each size (shows the intercept decaying) ==")
    hdr = f"{'metric':8}{'mode':>22}" + "".join(f"{s:>10}" for s in ORDER)
    print(hdr)
    for m in METRICS:
        for mode in ["oneshot", "precompute", "warm_strided"]:
            cells = ""
            for s in ORDER:
                d = best.get((m, s), {})
                w, h = DIMS[s]
                cells += f"{d[mode] / (w * h / 1.0e6):10.2f}" if mode in d else f"{'-':>10}"
            print(f"{m:8}{mode:>22}{cells}")

    print("\n== per-candidate cost and warm-reference break-even ==")
    print(
        f"{'metric':8}{'size':>6}{'oneshot':>10}{'precomp':>10}{'warm':>10}"
        f"{'warm/one':>10}{'ingress%':>10}{'N*':>6}{'n5 err%':>9}"
    )
    for r in rows:
        print(
            f"{r['metric']:8}{r['size_label']:>6}{r['oneshot_ms']:>10}{r['precompute_ms']:>10}"
            f"{r['warm_per_cand_ms']:>10}{r['warm_over_oneshot']:>10}"
            f"{r['ingress_pct_of_warm']:>10}{r['breakeven_candidates']:>6}"
            f"{r['loop_n5_model_err_pct']:>9}"
        )

    print("\n== numeric parity (abs delta; 0 == bit-identical) ==")
    seen = set()
    for (m, s), v in parity.items():
        for k, val in v.items():
            seen.add((m, k, val))
    agg = defaultdict(set)
    for m, k, val in seen:
        agg[(m, k)].add(val)
    for (m, k), vals in sorted(agg.items()):
        allz = all(float(x) == 0.0 for x in vals)
        print(f"{m:8} {k:32} {'ALL ZERO (bit-identical)' if allz else sorted(vals)}")

    print(f"\nwrote {out_path}  ({len(rows)} rows from {len(in_paths)} run(s))")


if __name__ == "__main__":
    main()
