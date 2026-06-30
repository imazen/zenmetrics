#!/usr/bin/env python3
"""Canonical support-aware picker DATA LAYER — THE place cross-codec picker oracle/training
tables are built. Every picker trainer MUST go through this (don't hand-roll `min(bytes)`).

WHY THIS EXISTS (2026-06-30, after the cross-codec coverage-bias bug)
--------------------------------------------------------------------
Two codecs can NEVER have the same sample density at every achieved quality: the sweep dials a
generic `q` that each codec resolves to its own native param (quality_to_quantizer /
resolve_distance_for_quality), so equal-q is not equal-quality, and the q->achieved map is
continuous + codec-specific. You cannot re-sweep your way to identical density. So:

  1. Resample every codec's RD curve onto a COMMON achieved-quality target grid (zq or ssim2).
  2. A codec is present at a target ONLY where it has MEASURED SUPPORT — i.e. the target lies
     inside its measured [min,max] achieved-quality range so `bytes_at` is an interpolation,
     never an extrapolation. Outside that range the codec is ABSENT (None), not guessed.
  3. Build the oracle/label ONLY on cells whose required support set is COMPLETE; otherwise
     EXCLUDE the cell and count it by which codec was missing. A `min` over an incomplete
     support set is a biased label (it can't see the absent codec) — that is the exact bug
     that made AVIF "win" above zq90.

This decouples the training distribution AND the oracle from any codec's q-sampling density, so
the cross-codec min-bytes comparison is valid. Pair with `check_quality_coverage.py` (the gate)
which refuses to proceed when high-band coverage is too asymmetric to even build complete cells.
"""
import collections
import numpy as np
import pandas as pd


def load_rd(base, families, split, score_col="score_zensim", lossless=False):
    """rd[variant][codec] = sorted [(achieved_quality, bytes)] measured points. `families` is a
    list of (codec_label, dataset_dir). For lossless pass `lossless=True` to keep only provably
    lossless rows (score>=99.999) — see lossless-data-filter-score100."""
    rd = collections.defaultdict(lambda: collections.defaultdict(list))
    for fam, d in families:
        df = pd.read_parquet(f"{base}/{d}/{split}.parquet",
                             columns=["variant_name", score_col, "encoded_bytes"]).dropna()
        if lossless:
            df = df[df[score_col] >= 99.999]
        for v, s, b in zip(df.variant_name.values, df[score_col].values, df.encoded_bytes.values):
            rd[v][fam].append((float(s), float(b)))
    for v in rd:
        for f in rd[v]:
            rd[v][f] = sorted(rd[v][f])
    return rd


def bytes_at(pts, t):
    """Bytes to hit achieved quality `t`, interpolated WITHIN the measured range only. `None`
    when `t` is outside [min,max] measured quality — the codec has NO support there, and we
    never extrapolate (extrapolation is what silently biased the oracle)."""
    if not pts or t < pts[0][0] or t > pts[-1][0]:
        return None
    for i in range(1, len(pts)):
        z0, b0 = pts[i - 1]; z1, b1 = pts[i]
        if z0 <= t <= z1:
            return b0 if z1 == z0 else b0 + (b1 - b0) * (t - z0) / (z1 - z0)
    return pts[-1][1]


def supported(pts, t):
    """Whether a codec has measured support at target `t` (target inside its measured range)."""
    return bool(pts) and pts[0][0] <= t <= pts[-1][0]


def oracle_rows(rd, families, targets, require="all"):
    """Support-aware oracle cells on the common `targets` grid.

    require="all" (default, for unbiased LABELS): emit a cell only if EVERY family has measured
    support at that target; otherwise exclude it and tally which codec(s) were missing.
    require=K (int): emit if >=K families supported (oracle among the supported set — use only
    when you accept a possibly-biased label and have flagged it).

    Returns (rows, excluded) where each row = {variant, target, oracle, support(tuple), bytes(dict)}
    and `excluded` is a Counter keyed by the missing-codec tuple."""
    names = [f for f, _ in families]
    need = len(names) if require == "all" else int(require)
    rows, excluded = [], collections.Counter()
    for v in rd:
        for t in targets:
            sup = {f: bytes_at(rd[v].get(f, []), t) for f in names}
            sup = {f: b for f, b in sup.items() if b is not None}
            if len(sup) < need:
                excluded[tuple(sorted(set(names) - set(sup)))] += 1
                continue
            rows.append({"variant": v, "target": t, "oracle": min(sup, key=sup.get),
                         "support": tuple(sorted(sup)), "bytes": dict(sup)})
    return rows, excluded


def coverage_report(rd, families, targets):
    """Per target, per codec: fraction of variants with measured support. The gate's input."""
    names = [f for f, _ in families]
    allv = list(rd)
    return {t: {f: float(np.mean([supported(rd[v].get(f, []), t) for v in allv])) for f in names}
            for t in targets}


def assert_quality_parity(rd, families, targets, high_floor=90.0, max_spread=0.40):
    """The data-layer gate: raise SystemExit if, at any target >= high_floor, the cross-codec
    support-coverage spread exceeds max_spread (i.e. the oracle there would be biased toward the
    better-covered codec). Call this before training/eval. Mirrors check_quality_coverage.py."""
    rep = coverage_report(rd, families, targets)
    bad = []
    for t in targets:
        if t < high_floor:
            continue
        cov = rep[t]
        spread = max(cov.values()) - min(cov.values())
        if spread > max_spread:
            worst, best = min(cov, key=cov.get), max(cov, key=cov.get)
            bad.append(f"target {t:g}: {worst} {cov[worst]*100:.0f}% vs {best} {cov[best]*100:.0f}% (spread {spread*100:.0f}%)")
    if bad:
        raise SystemExit("QUALITY-COVERAGE GATE FAILED — biased oracle in picker-critical band:\n  "
                         + "\n  ".join(bad)
                         + "\nFix in SAMPLING (quality-targeted encode to a common achieved-quality grid),"
                         + " or restrict `targets` to the supported band + flag the rest.")
