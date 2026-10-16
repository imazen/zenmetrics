#!/usr/bin/env python3
"""Black-box convention search for the published AIC-4 `A-FINE` column.

Input 1: `--dump` — the TSV written by the `afine::tests::aic_dump` test in
`crates/zenmetrics-nn/src/afine.rs` (branch afine-aic): per
(pair, variant) rows of every A-FINE sub-score our port computes —
`n_r`/`n_d` raw `AFINEQhead` naturalness, `fid` raw `AFINEDhead`
fidelity, `s_r`/`s_d`/`s_f` after the NLM scalers, `lambda` the
AFINELearnLambda adaptive weight, `afine_all` the assembled score.

Input 2: `--site` — the AIC-4 site CSV with the published `A-FINE`
column (`~/tmp/aic-site/site_fullres_subset61.csv`) and `--config` —
`~/tmp/aic-site/config.json` carrying the site's JND mapping
(`jnd(x) = a·sign(x−b)·|x−b|^c`; for A-FINE a=3.9897, b=2, c=0.6167,
sign=+1: all scores sit below b so the map is `a·|x−b|^c` from the
b-anchor — monotone over the observed range).

Candidate space, per preprocessing variant:
  * each raw term, each plausible structured combination (adapter terms,
    naturalness deltas, mean terms), under identity and fitted affine
    (`pub ≈ a·x+b`) and one extra λ-grid for `exp(k·(s_r−s_d))·s_d + s_f`;
  * swapped ref/dist (fid is symmetric; (s_r−s_d) and λ flip sign).

Reports every candidate's residual vs the published column in raw units
(median/max |Δ|) and site JND (median/max), plus SRCC. Sorted by JND max.
"""

import argparse
import csv
import itertools
import json
import math
import sys


def load_dump(path):
    """dump tsv -> {variant: {pair: {col: float}}}"""
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(
            (ln for ln in f if not ln.startswith("#")), delimiter="\t"
        ):
            d = {}
            for k in ("n_r", "n_d", "fid", "s_r", "s_d", "s_f", "lambda", "afine_all"):
                v = r.get(k) or ""
                d[k] = float(v) if v.strip() else float("nan")
            rows.setdefault(r["variant"], {})[r["pair"]] = d
    return rows


def load_site(path, col="A-FINE"):
    out = {}
    for r in csv.DictReader(open(path)):
        d = r["distorted"].replace("PTC_", "")
        out[d] = float(r[col])
    return out


def jnd_map(m):
    a, b, c = m["a"], m["b"], m["c"]

    def g(x):
        d = x - b
        return a * math.copysign(abs(d) ** c, d)

    return g


def srcc(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r

    n = len(xs)
    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return num / den if den else float("nan")


def lstsq1(xs, ys):
    """fit y ≈ a·x + b -> (a, b)"""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx if sxx else 0.0
    return a, my - a * mx


def lstsq2(us, vs, ys):
    """fit y ≈ a·u + b·v + c via normal equations (numpy-free)."""
    n = len(ys)
    s = [
        [sum(u * u for u in us), sum(u * v for u, v in zip(us, vs)), sum(us)],
        [0.0, sum(v * v for v in vs), sum(vs)],
        [0.0, 0.0, float(n)],
    ]
    s[1][0] = s[0][1]
    s[2][0] = s[0][2]
    s[2][1] = s[1][2]
    rhs = [sum(u * y for u, y in zip(us, ys)), sum(v * y for v, y in zip(vs, ys)), sum(ys)]
    # gaussian elimination
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(s[r][i]))
        if abs(s[piv][i]) < 1e-12:
            return None
        s[i], s[piv] = s[piv], s[i]
        rhs[i], rhs[piv] = rhs[piv], rhs[i]
        for r in range(i + 1, 3):
            t = s[r][i] / s[i][i]
            for c in range(3):
                s[r][c] -= t * s[i][c]
            rhs[r] -= t * rhs[i]
    x = [0.0] * 3
    for i in (2, 1, 0):
        x[i] = (rhs[i] - sum(s[i][j] * x[j] for j in range(i + 1, 3))) / s[i][i]
    return x


def candidates(row):
    """All structured formulas evaluated on one dump row."""
    nr_, nd, fid = row["n_r"], row["n_d"], row["fid"]
    sr, sd, sf, lam, all_ = row["s_r"], row["s_d"], row["s_f"], row["lambda"], row["afine_all"]
    return {
        "afine_all": all_,
        "s_f": sf,
        "s_d": sd,
        "s_r": sr,
        "fid_raw": fid,
        "n_d": nd,
        "n_r": nr_,
        "lambda": lam,
        "lam*sd": lam * sd,
        "lam*sr": lam * sr,
        "sf+sd": sf + sd,
        "sf-sd": sf - sd,
        "sf+sr": sf + sr,
        "sf-sr": sf - sr,
        "(sd+sf)/2": (sd + sf) / 2,
        "sf+(sr-sd)": sf + (sr - sd),
        "sf+(sd-sr)": sf + (sd - sr),
        "sf+(nr-nd)": sf + (nr_ - nd),
        "sf+(nd-nr)": sf + (nd - nr_),
        "nr-nd": nr_ - nd,
        "nd-nr": nd - nr_,
        "sr-sd": sr - sd,
        "sd-sr": sd - sr,
        "lam*sd+sf(=all)": lam * sd + sf,
        "lam*sr+sf": lam * sr + sf,
        "(1-lam)*sd+sf": (1 - lam) * sd + sf,
        "sd+sf-lam*sd": sd + sf - lam * sd,
        "sf*lam+sd": sf * lam + sd,
        "sf/lam+sd": sf / lam + sd if lam else float("nan"),
        "clip_all_-2..2": min(2.0, max(-2.0, all_)),
        "all/(1+|all|)": all_ / (1.0 + abs(all_)),
        "2*tanh(all/2)": 2.0 * math.tanh(all_ / 2.0),
        "nlm(all)": 4.0 / (1.0 + math.exp(-(-(all_ - 0.5) / 0.16))) - 2.0,
        "sf+lam*(sr-sd)": sf + lam * (sr - sd),
        "sf*exp(nd-nr)": sf * math.exp(max(-20.0, min(20.0, nd - nr_))),
        "sf*(1-(sr-sd))": sf * (1.0 - (sr - sd)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="write full residual TSV")
    args = ap.parse_args()

    dump = load_dump(args.dump)
    pub = load_site(args.site)
    g = jnd_map(json.load(open(args.config))["mappings"]["A-FINE"])

    results = []
    detail_rows = []
    for variant, rows in sorted(dump.items()):
        pairs = [p for p in rows if p in pub]
        pubs = [pub[p] for p in pairs]
        cnames = None
        for p in pairs:
            cs = candidates(rows[p])
            cnames = cnames or list(cs)
        def emit(preds_all, name, fit, params):
            """Skip pairs whose prediction is NaN (e.g. missing sub-scores)."""
            sel = [(p, x) for p, x in zip(pairs, preds_all) if not math.isnan(x)]
            if len(sel) < 10:
                return
            ps = [p for p, _ in sel]
            preds = [x for _, x in sel]
            pbs = [pub[p] for p in ps]
            d = [abs(a - b) for a, b in zip(preds, pbs)]
            dj = [abs(g(a) - g(b)) for a, b in zip(preds, pbs)]
            results.append(
                (sorted(dj)[len(dj) // 2], max(dj), sorted(d)[len(d) // 2], max(d), srcc(preds, pbs), variant, name, fit, params + f" n={len(ps)}")
            )

        for name in cnames or []:
            preds = [candidates(rows[p])[name] for p in pairs]
            emit(preds, name, "raw", "")
            sel = [(p, x) for p, x in zip(pairs, preds) if not math.isnan(x)]
            if len(sel) < 10:
                continue
            pr = [x for _, x in sel]
            pb = [pub[p] for p, _ in sel]
            a, b = lstsq1(pr, pb)
            emit([a * x + b for x in preds], name, "affine", f"a={a:.6g} b={b:.6g}")
        # two-term affine fits over every pair of candidate columns
        for u, v in itertools.combinations(cnames or [], 2):
            us = [candidates(rows[p])[u] for p in pairs]
            vs = [candidates(rows[p])[v] for p in pairs]
            sel = [
                (p, x, y)
                for p, x, y in zip(pairs, us, vs)
                if not (math.isnan(x) or math.isnan(y))
            ]
            if len(sel) < 10:
                continue
            uu = [x for _, x, _ in sel]
            vv = [y for _, _, y in sel]
            coef = lstsq2(uu, vv, [pub[p] for p, _, _ in sel])
            if not coef:
                continue
            a, b, c = coef
            emit(
                [a * x + b * y + c for x, y in zip(us, vs)],
                f"{u}+{v}",
                "2term",
                f"a={a:.4g} b={b:.4g} c={c:.4g}",
            )
        # lambda-grid on the adapter form exp(k*(sr-sd))*sd + sf
        for kk in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0):
            preds = [
                math.exp(kk * (r["s_r"] - r["s_d"])) * r["s_d"] + r["s_f"]
                if not (math.isnan(r["s_r"]) or math.isnan(r["s_d"]) or math.isnan(r["s_f"]))
                else float("nan")
                for r in (rows[p] for p in pairs)
            ]
            emit(preds, f"exp({kk}*(sr-sd))*sd+sf", "raw", "")
    results.sort()
    w = csv.writer(sys.stdout, delimiter="\t")
    w.writerow(["jnd_med", "jnd_max", "raw_med", "raw_max", "srcc", "variant", "formula", "fit", "params"])
    for r in results:
        w.writerow([f"{r[0]:.4g}", f"{r[1]:.4g}", f"{r[2]:.4g}", f"{r[3]:.4g}", f"{r[4]:.4f}", r[5], r[6], r[7], r[8]])


if __name__ == "__main__":
    main()
