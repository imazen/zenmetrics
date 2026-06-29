#!/usr/bin/env python3
"""Aggregate fleet-LOO per-box result JSONs into a ranked feature-importance table
and the pair-aware safe-to-drop analysis.

Reads every results/box-*.json the fleet wrote (round 1 single-feature drops AND/OR
round 2 group/pairwise drops can coexist in one dir), keyed by `tag`. The baseline is
the variant with drop=[].

LOO importance (round 1):
    delta_f = overhead(drop f) - overhead(baseline)        [percentage POINTS]
    delta_f > 0  -> dropping f HURTS -> f carries marginal RD signal -> KEEP
    delta_f ~ 0  -> f is (first-order) dead weight -> "droppable-looking" (verify jointly!)

PAIR-AWARENESS (round 2) — single-feature LOO UNDER-estimates features valuable only in
pairs (redundant A,B: each reads droppable alone because the other covers; you cannot
drop both). So a feature is NOT declared droppable from its single-LOO delta alone. The
verified safe-to-drop set is proven by JOINTLY dropping the droppable-looking set:
    group_delta(S) = overhead(drop all of S) - overhead(baseline)
    group_delta(S) small  -> S is jointly safe to drop
    group_delta(S) large  -> S contains redundant/pair value -> bisect to the largest safe subset
Pairwise interaction (on suspicious pairs):
    interaction(A,B) = group_delta({A,B}) - (delta_A + delta_B)
    >> 0  -> redundancy: keep at least one (dropping both removes shared signal)
    << 0  -> substitution/saturation

Usage:
    loo_collect.py --results-dir <dir> --codec <c> --metric <m> --out-dir <dir>
                   [--keep-threshold-pp 0.05] [--metric-axis val|test]
"""
import argparse
import glob
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--codec", required=True)
    ap.add_argument("--metric", default="ssim2")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep-threshold-pp", type=float, default=0.05,
                    help="features with val delta <= this (pp) are 'droppable-looking' "
                         "(candidates for round-2 joint-drop verification)")
    ap.add_argument("--metric-axis", choices=["val", "test"], default="val",
                    help="rank on val (default) or held-out test overhead delta")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    by_tag = {}
    meta = {}
    for fp in sorted(glob.glob(os.path.join(args.results_dir, "box-*.json"))):
        try:
            doc = json.load(open(fp))
        except Exception as e:  # noqa: BLE001
            print(f"  WARN skipping {fp}: {e}")
            continue
        for k in ("codec_config", "pareto_stem", "picker_target", "seed",
                  "n_keep_features", "pareto_cells"):
            if doc.get(k) is not None:
                meta.setdefault(k, doc.get(k))
        for r in doc.get("results", []):
            r["_box"] = doc.get("box_id")
            tag = r["tag"]
            prev = by_tag.get(tag)
            if prev is None or (prev.get("val_overhead") is None and r.get("val_overhead") is not None):
                by_tag[tag] = r

    if not by_tag:
        raise SystemExit(f"no box-*.json results in {args.results_dir}")

    base = by_tag.get("baseline")
    if base is None or base.get("val_overhead") is None:
        raise SystemExit("no usable baseline variant (tag='baseline', drop=[]) — cannot compute deltas")
    bval = base["val_overhead"]
    btest = base.get("test_overhead")
    axis = args.metric_axis

    def ov(r, which):  # overhead on the chosen axis
        return r.get(f"{which}_overhead")

    # ── single-feature LOO ranking ────────────────────────────────────────────
    singles = []  # (feature, row)
    groups = []   # (tag, row) for |drop|>=2
    for tag, r in by_tag.items():
        if tag == "baseline":
            continue
        drop = r.get("drop") or []
        if len(drop) == 1:
            singles.append((drop[0], r))
        elif len(drop) >= 2:
            groups.append((tag, r))

    rank_rows = []
    for feat, r in singles:
        vd = (r["val_overhead"] - bval) if r.get("val_overhead") is not None else None
        td = (r["test_overhead"] - btest) if (r.get("test_overhead") is not None and btest is not None) else None
        rank_rows.append({
            "feature": feat,
            "val_baseline": bval, "val_dropped": r.get("val_overhead"), "val_delta_pp": vd,
            "test_baseline": btest, "test_dropped": r.get("test_overhead"), "test_delta_pp": td,
            "n_features": r.get("n_features"), "rc": r.get("rc"),
        })
    keyaxis = "val_delta_pp" if axis == "val" else "test_delta_pp"
    rank_rows.sort(key=lambda x: (x[keyaxis] is None, -(x[keyaxis] or -1e9)))

    tsv = Path(args.out_dir) / f"loo_{args.codec}_{args.metric}.tsv"
    with open(tsv, "w") as f:
        f.write("rank\tfeature\tval_baseline\tval_dropped\tval_delta_pp\t"
                "test_baseline\ttest_dropped\ttest_delta_pp\tn_features\tverdict\trc\n")
        for i, x in enumerate(rank_rows, 1):
            d = x[keyaxis]
            verdict = "KEEP" if (d is not None and d > args.keep_threshold_pp) else "droppable?"
            def fmt(v):
                return f"{v:.4f}" if isinstance(v, float) else ("NA" if v is None else str(v))
            f.write(f"{i}\t{x['feature']}\t{fmt(x['val_baseline'])}\t{fmt(x['val_dropped'])}\t"
                    f"{fmt(x['val_delta_pp'])}\t{fmt(x['test_baseline'])}\t{fmt(x['test_dropped'])}\t"
                    f"{fmt(x['test_delta_pp'])}\t{fmt(x['n_features'])}\t{verdict}\t{fmt(x['rc'])}\n")

    droppable_looking = [x["feature"] for x in rank_rows
                         if x[keyaxis] is not None and x[keyaxis] <= args.keep_threshold_pp]
    must_keep = [x["feature"] for x in rank_rows
                 if x[keyaxis] is not None and x[keyaxis] > args.keep_threshold_pp]

    # ── round-2 group / pairwise analysis (if those variants are present) ───────
    single_delta = {x["feature"]: x[keyaxis] for x in rank_rows}
    group_rows = []
    for tag, r in groups:
        drop = r.get("drop") or []
        gov = ov(r, axis)
        gd = (gov - (bval if axis == "val" else btest)) if gov is not None else None
        interaction = None
        if len(drop) == 2:
            a, b = drop
            if single_delta.get(a) is not None and single_delta.get(b) is not None and gd is not None:
                interaction = gd - (single_delta[a] + single_delta[b])
        group_rows.append({"tag": tag, "drop": drop, "n_dropped": len(drop),
                           "group_overhead": gov, "group_delta_pp": gd,
                           "interaction_pp": interaction})

    if group_rows:
        g_tsv = Path(args.out_dir) / f"loo_{args.codec}_{args.metric}_round2.tsv"
        with open(g_tsv, "w") as f:
            f.write("tag\tn_dropped\tgroup_overhead\tgroup_delta_pp\tinteraction_pp\tdrop\n")
            for g in sorted(group_rows, key=lambda x: (x["group_delta_pp"] is None,
                                                       -(x["group_delta_pp"] or -1e9))):
                def fmt(v):
                    return f"{v:.4f}" if isinstance(v, float) else ("NA" if v is None else str(v))
                f.write(f"{g['tag']}\t{g['n_dropped']}\t{fmt(g['group_overhead'])}\t"
                        f"{fmt(g['group_delta_pp'])}\t{fmt(g['interaction_pp'])}\t{','.join(g['drop'])}\n")

    # ── verified safe-to-drop set: the LARGEST group whose joint drop didn't hurt ─
    # (largest |drop| with group_delta_pp <= keep_threshold). The fleet supplies the
    # candidate groups/bisection subsets; we pick the largest verified-safe one.
    verified_safe = None
    safe_groups = [g for g in group_rows
                   if g["group_delta_pp"] is not None and g["group_delta_pp"] <= args.keep_threshold_pp]
    if safe_groups:
        verified_safe = max(safe_groups, key=lambda g: g["n_dropped"])

    # ── human summary ───────────────────────────────────────────────────────────
    summ = Path(args.out_dir) / f"loo_{args.codec}_{args.metric}_summary.md"
    with open(summ, "w") as f:
        f.write(f"# Fleet-LOO feature ablation — {args.codec} / {args.metric}\n\n")
        f.write(f"- config: `{meta.get('codec_config')}`  picker_target: `{meta.get('picker_target')}`  "
                f"seed: {meta.get('seed')}  features: {meta.get('n_keep_features')}  "
                f"pareto_cells: {meta.get('pareto_cells')}\n")
        f.write(f"- baseline overhead: val {bval:.3f}%"
                + (f"  test {btest:.3f}%" if btest is not None else "") + "\n")
        f.write(f"- ranked on **{axis}** delta; keep-threshold {args.keep_threshold_pp}pp\n")
        f.write(f"- single-feature variants: {len(rank_rows)}  "
                f"(must-keep {len(must_keep)} / droppable-looking {len(droppable_looking)})\n")
        if group_rows:
            f.write(f"- round-2 group/pairwise variants: {len(group_rows)}\n")
        f.write("\n## Top features that MATTER (highest LOO importance)\n\n")
        f.write("| rank | feature | val Δpp | test Δpp |\n|---|---|---|---|\n")
        for i, x in enumerate(rank_rows[:20], 1):
            vd = x["val_delta_pp"]; td = x["test_delta_pp"]
            f.write(f"| {i} | {x['feature']} | "
                    f"{'NA' if vd is None else f'{vd:+.3f}'} | "
                    f"{'NA' if td is None else f'{td:+.3f}'} |\n")
        f.write("\n## Droppable-looking tail (single-LOO ~0 — MUST be joint-verified, "
                "not dropped on single-LOO alone)\n\n")
        f.write("```\n" + " ".join(droppable_looking) + "\n```\n")
        if group_rows:
            f.write("\n## Round-2 verification (pair-aware)\n\n")
            if verified_safe is not None:
                f.write(f"- **VERIFIED safe-to-drop** (largest joint drop with Δ ≤ "
                        f"{args.keep_threshold_pp}pp): {verified_safe['n_dropped']} features, "
                        f"group Δ {verified_safe['group_delta_pp']:+.3f}pp\n")
                f.write("```\n" + " ".join(verified_safe["drop"]) + "\n```\n")
            big_int = sorted([g for g in group_rows if g["interaction_pp"] is not None],
                             key=lambda g: -abs(g["interaction_pp"]))[:15]
            if big_int:
                f.write("\n### Strongest pairwise interactions (|interaction|): "
                        ">>0 redundant→keep≥1, <<0 substitution\n\n")
                f.write("| pair | interaction pp | group Δpp |\n|---|---|---|\n")
                for g in big_int:
                    f.write(f"| {','.join(g['drop'])} | {g['interaction_pp']:+.3f} | "
                            f"{g['group_delta_pp']:+.3f} |\n")

    print(f"wrote {tsv}")
    if group_rows:
        print(f"wrote {Path(args.out_dir) / f'loo_{args.codec}_{args.metric}_round2.tsv'}")
    print(f"wrote {summ}")
    print(f"baseline val {bval:.3f}%" + (f" test {btest:.3f}%" if btest is not None else ""))
    print(f"must-keep {len(must_keep)} / droppable-looking {len(droppable_looking)} "
          f"(of {len(rank_rows)} single-feature variants)")
    if verified_safe is not None:
        print(f"VERIFIED safe-to-drop (joint): {verified_safe['n_dropped']} features, "
              f"group Δ {verified_safe['group_delta_pp']:+.3f}pp")


if __name__ == "__main__":
    main()
