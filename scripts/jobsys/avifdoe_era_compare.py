#!/usr/bin/env python3
"""avifdoe_era_compare.py — CROSS-ERA cell identity for the AVIF DOE.

The question this exists for: when the encoder pin moves, does a re-run of an
UNCHANGED plan reproduce the old bytes? Because DOE cells are content-addressed,
that is answerable exactly — no statistics, no BD-rate, no tolerance — by
joining two runs' `zenfleet-ctl pairs` tables on the cell identity
`(image_path, q, knob_tuple.cell)` and comparing `encode_sha`.

WHY A NEW TOOL AND NOT AN EXISTING ONE. `avifdoe_stagea_analyze.py` already
writes `arm_byte_identity.tsv`, but that is *arm vs its own run's control* (the
alias detector of plan §7.1(7)) — a WITHIN-run question. Nothing answered the
BETWEEN-run/BETWEEN-era one. This script computes no statistic that another
owner owns: BD-rate stays in `avifdoe_stagea_analyze.py`, rank/correlation stats
stay in `zenstats`. It reshapes two pairs tables into an identity verdict, and
(with --scored) separates ENCODER drift from SCORER drift by re-reading the two
waves' scores for cells whose bytes are provably identical.

ERA DISCIPLINE. This tool never pools two eras' rows into one population. Every
output row is a PAIRED comparison of the same cell identity across two runs, or
a count of such pairs. Effect-level comparison (median BD-rate at pin X vs pin Y)
is the caller's job and must be stated as effect-vs-effect, never as one sample.

Usage:
  avifdoe_era_compare.py --old a1=OLD.parquet --new a1=NEW.parquet \
      [--old-scored doe_scored.parquet --new-scored era_scored.parquet] \
      [--label-old 'ef0b122b' --label-new '2ca060f4'] --outdir DIR
"""
import argparse, collections, json, os, sys

import pyarrow.parquet as pq


def load_pairs(path):
    """-> {(image_path, q, cell): encode_sha}, plus the plan name(s) seen."""
    t = pq.read_table(path).to_pydict()
    d, plans = {}, collections.Counter()
    for img, q, kt, es in zip(t["image_path"], t["q"], t["knob_tuple_json"], t["encode_sha"]):
        k = json.loads(kt)
        cell = k.get("cell")
        if cell is None:                       # naive sweeps: {backend,speed}
            cell = f"s{k.get('speed')}-{k.get('backend')}"
        plans[k.get("plan", "?")] += 1
        d[(img, int(q), cell)] = es.rsplit("/", 1)[-1]
    return d, plans


def load_scores(path, metric_cols=("m_ssim2",)):
    """harvest parquet -> {encode_sha_basename: {metric: value}}."""
    t = pq.read_table(path).to_pydict()
    out = {}
    for i, es in enumerate(t["encode_sha"]):
        if es is None:
            continue
        key = es.rsplit("/", 1)[-1]
        row = {}
        for m in metric_cols:
            if m in t and t[m][i] is not None:
                row[m] = t[m][i]
        if "bytes" in t and t["bytes"][i] is not None:
            row["bytes"] = t["bytes"][i]
        if row:
            out[key] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", action="append", required=True, help="label=pairs.parquet (old era)")
    ap.add_argument("--new", action="append", required=True, help="label=pairs.parquet (new era)")
    ap.add_argument("--old-scored", default=None, help="harvest parquet for the OLD era")
    ap.add_argument("--new-scored", default=None, help="harvest parquet for the NEW era")
    ap.add_argument("--metric", default="m_ssim2")
    ap.add_argument("--label-old", default="old-pin")
    ap.add_argument("--label-new", default="new-pin")
    ap.add_argument("--effects-old", default=None,
                    help="OLD era's main_effects.tsv (from avifdoe_stagea_analyze.py)")
    ap.add_argument("--effects-new", default=None,
                    help="NEW era's main_effects.tsv — MUST be the same instrument "
                         "(same --control choice) or the comparison confounds "
                         "instrument with era")
    ap.add_argument("--move-threshold", type=float, default=1.0,
                    help="pp of BD-rate that counts as MOVED (registered: 1.0)")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    olds = dict(s.split("=", 1) for s in a.old)
    news = dict(s.split("=", 1) for s in a.new)
    labels = [l for l in news if l in olds]
    if not labels:
        sys.exit(f"no shared label between --old {sorted(olds)} and --new {sorted(news)}")

    summary, diff_rows, strat_rows = {}, [], []
    for lab in labels:
        old, oplans = load_pairs(olds[lab])
        new, nplans = load_pairs(news[lab])
        shared = sorted(set(old) & set(new))
        same = [k for k in shared if old[k] == new[k]]
        diff = [k for k in shared if old[k] != new[k]]
        per = collections.defaultdict(lambda: [0, 0])
        for k in shared:
            per[k[2]][0] += 1
            if old[k] == new[k]:
                per[k[2]][1] += 1
        for cell, (tot, ident) in sorted(per.items()):
            strat_rows.append((lab, cell, tot, ident, tot - ident, ident / tot))
        for k in diff:
            diff_rows.append((lab, k[0], k[1], k[2], old[k][:16], new[k][:16]))
        summary[lab] = dict(
            old_cells=len(old), new_cells=len(new), shared=len(shared),
            byte_identical=len(same), differing=len(diff),
            frac_identical=(len(same) / len(shared)) if shared else None,
            old_only=len(set(old) - set(new)), new_only=len(set(new) - set(old)),
            plans_old=dict(oplans), plans_new=dict(nplans),
            differing_images=sorted({k[0] for k in diff}),
            differing_strata=sorted({k[2] for k in diff}),
        )
        print(f"[{lab}] shared={len(shared)} identical={len(same)} "
              f"differ={len(diff)} ({100*len(same)/max(1,len(shared)):.2f}% identical)")

    with open(f"{a.outdir}/era_stratum_identity.tsv", "w") as f:
        f.write("label\tcell\tshared_cells\tbyte_identical\tdiffering\tfrac_identical\n")
        for r in strat_rows:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{r[5]:.6f}\n")
    with open(f"{a.outdir}/era_cell_diffs.tsv", "w") as f:
        f.write(f"label\timage\tq\tcell\tsha_{a.label_old}\tsha_{a.label_new}\n")
        for r in diff_rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    # ---- scorer-era separation ------------------------------------------------
    # For cells whose BYTES are identical, any score difference is the SCORER's,
    # not the encoder's. This is the only clean way to tell the two apart.
    if a.old_scored and a.new_scored:
        os_ = load_scores(a.old_scored, (a.metric,))
        ns_ = load_scores(a.new_scored, (a.metric,))
        rows, nmax, bmax = [], 0.0, 0
        n_cmp = 0
        for lab in labels:
            old, _ = load_pairs(olds[lab]); new, _ = load_pairs(news[lab])
            for k in set(old) & set(new):
                if old[k] != new[k]:
                    continue
                sha = old[k]
                if sha not in os_ or sha not in ns_:
                    continue
                ov, nv = os_[sha].get(a.metric), ns_[sha].get(a.metric)
                ob, nb = os_[sha].get("bytes"), ns_[sha].get("bytes")
                if ov is None or nv is None:
                    continue
                n_cmp += 1
                d = abs(ov - nv); nmax = max(nmax, d)
                if ob is not None and nb is not None:
                    bmax = max(bmax, abs(ob - nb))
                if d > 0:
                    rows.append((lab, k[0], k[1], k[2], ov, nv, d))
        summary["_scorer_check"] = dict(
            metric=a.metric, cells_compared=n_cmp, cells_differing=len(rows),
            max_abs_delta=nmax, max_abs_bytes_delta=bmax)
        print(f"[scorer] identical-byte cells compared={n_cmp} differing={len(rows)} "
              f"max|delta {a.metric}|={nmax:g} max|delta bytes|={bmax}")
        with open(f"{a.outdir}/era_score_drift.tsv", "w") as f:
            f.write(f"label\timage\tq\tcell\t{a.metric}_{a.label_old}\t{a.metric}_{a.label_new}\tabs_delta\n")
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")

    # ---- effect-vs-effect stability verdict -----------------------------------
    # NEVER pools the two eras. Reads each era's OWN main_effects.tsv (produced by
    # the BD-rate owner, avifdoe_stagea_analyze.py, on that era's own rows with
    # that era's own in-run control) and differences the two EFFECTS.
    if a.effects_old and a.effects_new:
        import csv as _csv

        def _eff(path):
            out = {}
            with open(path) as f:
                for r in _csv.DictReader(f, delimiter="\t"):
                    out[(int(r["speed"]), r["knob"])] = {
                        k: (float(v) if k not in ("knob",) else v) for k, v in r.items()}
            return out

        eo, en = _eff(a.effects_old), _eff(a.effects_new)
        # A cell is HELD-EXACT only when the identity join PROVED every one of its
        # contributing bitstreams reproduced. That is a byte fact, not a tolerance.
        exact_cells = set()
        for lab in labels:
            old, _ = load_pairs(olds[lab]); new, _ = load_pairs(news[lab])
            per = collections.defaultdict(lambda: [0, 0])
            for k in set(old) & set(new):
                per[k[2]][0] += 1
                if old[k] == new[k]:
                    per[k[2]][1] += 1
            for cell, (tot, ident) in per.items():
                if tot and tot == ident:
                    exact_cells.add(cell)

        rows = []
        for key in sorted(set(eo) | set(en)):
            sp, knob = key
            o, nw = eo.get(key), en.get(key)
            cell = f"s{sp}-svt-420-{knob}"
            if o is None or nw is None:
                verdict = "NOT-MEASURED-old" if o is None else "NOT-MEASURED-new"
                rows.append((sp, knob, "" if o is None else f"{o['median_bd']:.4f}",
                             "" if nw is None else f"{nw['median_bd']:.4f}",
                             "", "", verdict))
                continue
            d = nw["median_bd"] - o["median_bd"]
            if cell in exact_cells and abs(d) == 0.0:
                verdict = "HELD-EXACT (all bitstreams byte-identical)"
            elif abs(d) >= a.move_threshold:
                verdict = f"MOVED by {d:+.2f} pp"
            else:
                inside = (o["ci_lo"] <= nw["median_bd"] <= o["ci_hi"])
                verdict = ("HELD (<%.1f pp, new median inside old CI)" % a.move_threshold
                           if inside else
                           "HELD-magnitude (<%.1f pp) but outside old CI" % a.move_threshold)
            rows.append((sp, knob, f"{o['median_bd']:.4f}", f"{nw['median_bd']:.4f}",
                         f"{d:+.4f}",
                         f"[{o['ci_lo']:.3f},{o['ci_hi']:.3f}] vs [{nw['ci_lo']:.3f},{nw['ci_hi']:.3f}]",
                         verdict))
        with open(f"{a.outdir}/era_effect_stability.tsv", "w") as f:
            f.write(f"speed\tknob\tmedian_bd_{a.label_old}\tmedian_bd_{a.label_new}"
                    "\tdelta_pp\tci_old_vs_new\tverdict\n")
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")
        vc = collections.Counter(r[-1].split(" (")[0].split(" by ")[0] for r in rows)
        summary["_effect_stability"] = dict(cells=len(rows), verdicts=dict(vc),
                                            move_threshold_pp=a.move_threshold)
        print("[effects] verdicts:", dict(vc))

    summary["_meta"] = dict(label_old=a.label_old, label_new=a.label_new,
                            old=olds, new=news)
    json.dump(summary, open(f"{a.outdir}/era_identity_summary.json", "w"), indent=1)
    print(f"wrote {a.outdir}/era_stratum_identity.tsv, era_cell_diffs.tsv, era_identity_summary.json")


if __name__ == "__main__":
    main()
