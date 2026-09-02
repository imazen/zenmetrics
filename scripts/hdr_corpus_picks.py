#!/usr/bin/env python3
"""T2 corpus picks — gates G0.4 and G0.5 of the HBD AVIF arm.

Registered by ``benchmarks/avif_hdr_arm_plan_2026-09-02.md`` §4.3 (*Track 2 —
HDR-10: the missing RD baseline*).  Selects the **K = 16** HDR references that
Track 2 encodes, from the 76 16-bit PQ PNGs at
``/mnt/v/output/imazen-26-png-v2/**/*.hdr.png``, and writes the picks table that
**must exist before any encode cell runs** (G0.4).

What the plan registers, and what this implements
-------------------------------------------------

* **Cluster on the paired ``.sdr.png``, not the ``.hdr.png``.**  zenanalyze's
  picker features are 8-bit renditions by design ("RGB16/RGBA16 — converted to
  RGB8 by taking the high byte"; for HDR "the analyzer measures the SDR
  rendition", ``zenanalyze/src/lib.rs:373-382``), so the SDR rendition is the
  aligned, documented proxy: same content, same converter, same capture,
  differing only in transfer/range.  What it cannot see is dynamic-range
  structure — hence the explicit HDR axes below.
* **K = 16, centroid-nearest members.**  Categories are unbalanced (nature 47 of
  76), so k-means beats random, which would return a nature-dominated set.
* **G0.5 — primaries balance.**  Primaries are nearly determined by content in
  this corpus (interiors 19/20 BT.709, nature 39/47 Display-P3), so the picks are
  **constrained to carry >= 5 of each primaries value**, taking the
  centroid-*next*-nearest member within a cluster when the nearest would violate
  the constraint, and **recording every such substitution** in the picks TSV.
* **Peak-luminance stratification.**  The plan asks that the picks "span the
  peak-luminance range **by construction, not by luck**".  Implemented as a
  second constraint: >= 1 pick in each quantile bin of the corpus's measured
  peak luminance, repaired by the same minimum-cost substitution machinery and
  **never at the expense of the primaries constraint**.
* **G0.4 — the HDR axes ride along.**  zenanalyze's ``tier_depth`` scalars
  (``EffectiveBitDepth`` / ``HdrHeadroomStops`` / ``HdrPresent``, plus peak and
  p99 luminance) are the one part of zenanalyze that reads **source samples**
  rather than an RGB8 view, and they are carried per pick.

Owners this script drives rather than re-implements
---------------------------------------------------

Per the NO-DUPLICATE-IMPLEMENTATIONS rule (``zensim/CLAUDE.md``):

============================  ==========================================
task                          owner
============================  ==========================================
PNG header / cICP read        ``scripts/hdr_corpus_precheck.py``
                              (``read_png_header``, imported — not copied)
SDR clustering features       ``zenanalyze::analyze_features_rgb8`` via
                              ``zenanalyze/examples/extract_features_for_picker``
                              (``--sizes 0`` = native, no resize)
HDR ``tier_depth`` scalars    ``zenanalyze::analyze_features`` via
                              ``zenanalyze/examples/extract_hdr_features``
k-means                       ``scipy.cluster.vq.kmeans2`` (``minit='++'``)
============================  ==========================================

The GEOM feature exclusion + z-score + centroid-nearest selection mirror the
established convention in ``scripts/imazen26_recluster_even.py``.  That script
uses ``sklearn.cluster.KMeans``; sklearn is **not installed** on this box (its
site-packages are orphaned from a removed python3.10), so this uses scipy's
kmeans2 instead — a different implementation of the same algorithm, seeded and
verified deterministic.  Cluster labels are therefore **not** comparable to that
script's; the selection rule is.

``display_peak_nits``
---------------------

The plan's G0.4 names ``HdrRef.display_peak_nits``
(``zenmetrics-cli/src/sweep/hdr.rs:91``), whose owner is
``crate::hdr::measured_display_peak_nits`` (zenpixels ``CllMeasure``, MaxRgb,
clamped to ``[SDR_WHITE_NITS, 10000]``).  **No built binary exposes it per-file**
— it is computed inside ``decode_hdr_ref`` and reaches the outside world only as
the ``ref_peak_nits`` column of a ``sweep --hdr`` feature parquet, i.e. only
after an encode.  This script therefore leaves ``display_peak_nits`` **empty**
and carries zenanalyze's independently-measured
``peak_luminance_nits`` / ``p99_luminance_nits`` instead, which is what the
stratification actually uses.  The two are different measurements (MaxRgb over
PQ-EOTF'd channels vs zenanalyze's tier_depth luminance) and this script does
not claim they are interchangeable.  ``--display-peak-tsv`` accepts a
``variant<TAB>display_peak_nits`` table if one is ever produced, and fills the
column from it.

Usage
-----

    python3 scripts/hdr_corpus_picks.py \\
        --precheck-tsv ~/tmp/hbdexec/g0/hdr_refs_precheck.tsv \\
        --sdr-features ~/tmp/hbdexec/g0/t2_sdr_features.tsv \\
        --hdr-features /mnt/v/output/imazen-26-features/imazen26_hdr_features_2026-06-29.tsv \\
        --out benchmarks/avif_hdr_t2_picks_k16_2026-09-02.tsv

Exit status: **0** = picks written and every constraint satisfied; **1** = a
constraint could not be satisfied (no substitution exists); **2** = bad inputs.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

import numpy as np
from scipy.cluster.vq import kmeans2

# The PNG header reader is OWNED by the precheck tool, which sits beside this
# file. Import it rather than writing a second one.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hdr_corpus_precheck import (  # noqa: E402
    PRIMARIES_NAMES,
    TRANSFER_NAMES,
    BadPng,
    read_png_header,
)

# Geometry/size features excluded from the clustering space, matching
# scripts/imazen26_recluster_even.py. This corpus is NOT size-homogeneous
# (2355x3308 .. 5712x4284), so leaving these in would cluster by sensor crop.
GEOM = (
    "pixel_count",
    "log_pixels",
    "bitmap_bytes",
    "min_dim",
    "max_dim",
    "aspect",
    "block_misalign",
    "log_padded",
    "channel_count",
)

# G0.5, registered: at least this many picks per primaries value.
MIN_PER_PRIMARIES = 5

# The tier_depth / HDR-axis columns carried per pick, as named in
# zenanalyze/examples/extract_hdr_features output.
HDR_AXIS_COLS = (
    "feat_peak_luminance_nits",
    "feat_p99_luminance_nits",
    "feat_hdr_headroom_stops",
    "feat_hdr_pixel_fraction",
    "feat_wide_gamut_peak",
    "feat_wide_gamut_fraction",
    "feat_effective_bit_depth",
    "feat_hdr_present",
)


def stem_of(path: str) -> str:
    """``…/1064_general_…_3000x4000.hdr.png`` -> ``1064_general_…_3000x4000``.

    The stem is the join key across all four tables (precheck, SDR features,
    HDR features, picks) because ``.hdr.png`` and ``.sdr.png`` are two
    renditions of one capture and share it.
    """
    name = pathlib.PurePosixPath(path).name
    for suffix in (".hdr.png", ".sdr.png", ".png"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def read_tsv(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        rows = []
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            rows.append(dict(zip(header, line.split("\t"))))
    return header, rows


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_refs(precheck_tsv: pathlib.Path, verify_headers: bool) -> list[dict]:
    """The 76 HDR references, primaries re-verified against the live files.

    The precheck TSV is the registered input (G0.5's owner emits it), but a
    stale TSV is exactly the failure this arm cannot afford, so every row's
    ``primaries``/``transfer`` is re-read from the PNG header through the
    precheck tool's own reader and mismatches are fatal.
    """
    _, rows = read_tsv(precheck_tsv)
    refs = []
    for r in rows:
        hdr_path = pathlib.Path(r["path"])
        sdr_path = pathlib.Path(str(hdr_path).replace(".hdr.png", ".sdr.png"))
        if not sdr_path.exists():
            raise SystemExit(f"G0.4 ERROR: no paired .sdr.png for {hdr_path}")
        prim, trans = int(r["primaries"]), int(r["transfer"])
        if verify_headers:
            try:
                info = read_png_header(hdr_path)
            except (BadPng, OSError) as e:
                raise SystemExit(f"G0.4 ERROR: cannot re-read {hdr_path}: {e}") from e
            cicp = info["cicp"] or (None, None, None, None)
            if (cicp[0], cicp[1]) != (prim, trans):
                raise SystemExit(
                    f"G0.4 ERROR: precheck TSV is stale for {hdr_path.name}: "
                    f"TSV says primaries={prim} transfer={trans}, "
                    f"file says primaries={cicp[0]} transfer={cicp[1]}"
                )
        refs.append(
            {
                "variant": stem_of(str(hdr_path)),
                "hdr_path": str(hdr_path),
                "sdr_path": str(sdr_path),
                "category": r["category"],
                "primaries": prim,
                "transfer": trans,
            }
        )
    refs.sort(key=lambda d: d["variant"])  # determinism: fixed row order
    return refs


def build_feature_matrix(
    refs: list[dict], sdr_features: pathlib.Path
) -> tuple[np.ndarray, list[str]]:
    """z-scored clustering matrix over the SDR renditions.

    Drops GEOM columns (size, not content), then any column that is non-finite
    anywhere or constant across the corpus — a constant column contributes
    nothing to a distance and a zero std would divide by zero.
    """
    header, rows = read_tsv(sdr_features)
    by_stem = {stem_of(r["image_path"]): r for r in rows}
    missing = [r["variant"] for r in refs if r["variant"] not in by_stem]
    if missing:
        raise SystemExit(
            f"G0.4 ERROR: {len(missing)} reference(s) absent from {sdr_features}: "
            + ", ".join(missing[:5])
        )
    cols = [
        c
        for c in header
        if c.startswith("feat_") and not any(g in c for g in GEOM)
    ]
    raw = np.array(
        [[_to_float(by_stem[r["variant"]].get(c, "")) for c in cols] for r in refs],
        dtype=np.float64,
    )
    finite = np.isfinite(raw).all(axis=0)
    std = raw.std(axis=0, ddof=0)
    keep = finite & (std > 0)
    kept_cols = [c for c, k in zip(cols, keep) if k]
    x = raw[:, keep]
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)
    return x, kept_cols


def _to_float(s: str) -> float:
    """Empty cell = the analyzer's NaN sentinel (see feature_value_str)."""
    if s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def cluster(
    x: np.ndarray, k: int, seed: int, n_init: int
) -> tuple[np.ndarray, float, np.ndarray]:
    """k-means++ / Lloyd, best inertia over ``n_init`` seeded restarts.

    Restarts use ``seed, seed+1, …`` so the whole run is a pure function of
    ``--seed``.  Restarts that leave a cluster empty are rejected: K=16 buckets
    must each nominate a member, and an empty one would silently shrink the
    corpus below 16.
    """
    best = None
    for i in range(n_init):
        centroids, labels = kmeans2(
            x, k, iter=300, minit="++", seed=seed + i, missing="warn"
        )
        if len(set(labels.tolist())) != k:
            continue
        inertia = float(((x - centroids[labels]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (centroids, labels, inertia)
    if best is None:
        raise SystemExit(
            f"G0.4 ERROR: no seeded restart of k-means produced {k} non-empty "
            f"clusters (seed={seed}, n_init={n_init}); raise --n-init or lower -k"
        )
    centroids, labels, inertia = best
    dists = np.linalg.norm(x - centroids[labels], axis=1)
    return labels, inertia, dists


def quantile_bins(values: list[float], nbins: int) -> list[int]:
    """Assign each value to one of ``nbins`` equal-count quantile bins.

    Bin index is by sorted rank, not by value edges, so ties and a skewed
    distribution cannot produce an empty bin.
    """
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    bins = [0] * len(values)
    for rank, i in enumerate(order):
        bins[i] = min(nbins - 1, rank * nbins // len(values))
    return bins


def select(
    refs: list[dict],
    labels: np.ndarray,
    dists: np.ndarray,
    k: int,
    peak_bins: list[int] | None,
    nbins: int,
    min_per_peak_bin: int,
) -> tuple[list[int], list[dict]]:
    """Centroid-nearest members, repaired to satisfy the registered constraints.

    Repair is greedy **minimum-cost**: the cost of replacing cluster *c*'s pick
    with its rank-*r* member is that member's extra distance to the centroid.
    Each step takes the cheapest substitution that fixes a deficit without
    breaking an already-satisfied constraint; ties break on (cost, cluster,
    rank) so the result is deterministic.  Primaries (G0.5) is repaired first
    and is never traded away to satisfy the peak-luminance constraint.
    """
    # Members of each cluster, ordered by distance to that cluster's centroid.
    members: dict[int, list[int]] = collections.defaultdict(list)
    for i in range(len(refs)):
        members[int(labels[i])].append(i)
    for c in members:
        members[c].sort(key=lambda i: (float(dists[i]), refs[i]["variant"]))

    picks = {c: members[c][0] for c in range(k)}
    subs: list[dict] = []

    def counts_primaries(sel):
        return collections.Counter(refs[i]["primaries"] for i in sel.values())

    def counts_bins(sel):
        if peak_bins is None:
            return collections.Counter()
        return collections.Counter(peak_bins[i] for i in sel.values())

    def cost(c: int, i: int) -> float:
        return float(dists[i]) - float(dists[picks[c]])

    def repair(deficit_fn, gain_fn, protect_fns, label: str) -> bool:
        """Fix one deficit class. Returns False when no substitution exists."""
        while True:
            need = deficit_fn(picks)
            if not need:
                return True
            best = None
            for want in sorted(need):
                for c in range(k):
                    cur = picks[c]
                    for rank, cand in enumerate(members[c]):
                        if cand == cur or gain_fn(cand) != want:
                            continue
                        trial = dict(picks)
                        trial[c] = cand
                        if any(p(trial) for p in protect_fns):
                            continue  # would break a satisfied constraint
                        key = (cost(c, cand), c, rank)
                        if best is None or key < best[0]:
                            best = (key, c, cand, rank, want)
            if best is None:
                return False
            _key, c, cand, rank, want = best
            displaced = picks[c]
            subs.append(
                {
                    "cluster": c,
                    "chosen": cand,
                    "displaced": displaced,
                    "rank": rank,
                    "delta_dist": float(dists[cand]) - float(dists[displaced]),
                    "reason": f"{label}: needed {want}",
                }
            )
            picks[c] = cand

    def primaries_deficit(sel):
        cnt = counts_primaries(sel)
        present = sorted({r["primaries"] for r in refs})
        return [p for p in present if cnt[p] < MIN_PER_PRIMARIES]

    def primaries_broken(sel):
        return bool(primaries_deficit(sel))

    def bins_deficit(sel):
        if peak_bins is None:
            return []
        cnt = counts_bins(sel)
        return [b for b in range(nbins) if cnt[b] < min_per_peak_bin]

    if not repair(
        primaries_deficit,
        lambda i: refs[i]["primaries"],
        [],
        "G0.5 primaries balance",
    ):
        raise SystemExit(
            "G0.5 FAIL: cannot reach "
            f">={MIN_PER_PRIMARIES} picks per primaries value with K={k} clusters. "
            f"Counts: {dict(counts_primaries(picks))}"
        )

    if peak_bins is not None and not repair(
        bins_deficit,
        lambda i: peak_bins[i],
        [primaries_broken],
        "peak-luminance stratification",
    ):
        raise SystemExit(
            "G0.4 FAIL: cannot cover every peak-luminance bin without breaking "
            f"the G0.5 primaries constraint. Bin counts: {dict(counts_bins(picks))}"
        )

    return [picks[c] for c in range(k)], subs


def crosstab(rows: list[dict], title: str) -> str:
    cross = collections.Counter((r["category"], r["primaries"]) for r in rows)
    prims = sorted({p for _, p in cross})
    cats = sorted({c for c, _ in cross})
    out = [f"== {title} ==", f"{'category':<36}" + "".join(f"{PRIMARIES_NAMES.get(p, p)!s:>14}" for p in prims) + f"{'TOTAL':>10}"]
    for c in cats:
        row = [cross[(c, p)] for p in prims]
        out.append(f"{c:<36}" + "".join(f"{v:>14}" for v in row) + f"{sum(row):>10}")
    tot = [sum(cross[(c, p)] for c in cats) for p in prims]
    out.append(f"{'TOTAL':<36}" + "".join(f"{v:>14}" for v in tot) + f"{sum(tot):>10}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--precheck-tsv", type=pathlib.Path, required=True,
                    help="per-file TSV from scripts/hdr_corpus_precheck.py --tsv")
    ap.add_argument("--sdr-features", type=pathlib.Path, required=True,
                    help="zenanalyze features TSV over the paired .sdr.png "
                         "(extract_features_for_picker --sizes 0)")
    ap.add_argument("--hdr-features", type=pathlib.Path,
                    help="zenanalyze HDR features TSV (extract_hdr_features) — "
                         "supplies the tier_depth scalars and the peak-luminance axis")
    ap.add_argument("--display-peak-tsv", type=pathlib.Path,
                    help="optional 'variant<TAB>display_peak_nits' table; without it "
                         "that column is left EMPTY (see the module docstring)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("-k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--peak-bins", type=int, default=4,
                    help="quantile bins of peak luminance the picks must span (0 = off)")
    ap.add_argument("--min-per-peak-bin", type=int, default=1)
    ap.add_argument("--no-verify-headers", action="store_true",
                    help="skip re-reading every PNG header (not recommended)")
    ap.add_argument("--no-sha", action="store_true",
                    help="skip sha256 of the picked references (faster; loses G0.1 input)")
    args = ap.parse_args()

    refs = load_refs(args.precheck_tsv, not args.no_verify_headers)
    if not refs:
        print("G0.4 ERROR: precheck TSV holds no rows", file=sys.stderr)
        return 2
    print(f"references: {len(refs)}")

    x, kept_cols = build_feature_matrix(refs, args.sdr_features)
    print(f"clustering space: {x.shape[0]} refs x {len(kept_cols)} features "
          f"(SDR renditions; GEOM + constant + non-finite columns dropped)")

    # HDR axes (tier_depth) — joined on the shared stem.
    hdr_axes: dict[str, dict[str, str]] = {}
    if args.hdr_features:
        header, rows = read_tsv(args.hdr_features)
        key = "variant_name" if "variant_name" in header else "image_path"
        hdr_axes = {stem_of(r[key]): r for r in rows}
        absent = [r["variant"] for r in refs if r["variant"] not in hdr_axes]
        if absent:
            print(f"WARNING: {len(absent)} reference(s) missing from "
                  f"{args.hdr_features}; their HDR axes will be empty", file=sys.stderr)

    peak_vals: list[float] | None = None
    peak_bins: list[int] | None = None
    if hdr_axes and args.peak_bins > 0:
        vals = [
            _to_float(hdr_axes.get(r["variant"], {}).get("feat_peak_luminance_nits", ""))
            for r in refs
        ]
        if all(np.isfinite(v) for v in vals):
            peak_vals = vals
            peak_bins = quantile_bins(vals, args.peak_bins)
            print(f"peak-luminance axis: {min(vals):.1f} .. {max(vals):.1f} cd/m^2, "
                  f"{args.peak_bins} quantile bins, >= {args.min_per_peak_bin} pick(s) each")
        else:
            print("WARNING: peak luminance not available for every reference — "
                  "peak-luminance stratification DISABLED", file=sys.stderr)

    labels, inertia, dists = cluster(x, args.k, args.seed, args.n_init)
    sizes = collections.Counter(int(v) for v in labels)
    print(f"k-means: K={args.k} seed={args.seed} n_init={args.n_init} "
          f"inertia={inertia:.4f} sizes={[sizes[c] for c in range(args.k)]}")

    chosen, subs = select(refs, labels, dists, args.k, peak_bins,
                          args.peak_bins, args.min_per_peak_bin)

    # Rank of each chosen member inside its cluster (0 = centroid-nearest).
    members: dict[int, list[int]] = collections.defaultdict(list)
    for i in range(len(refs)):
        members[int(labels[i])].append(i)
    for c in members:
        members[c].sort(key=lambda i: (float(dists[i]), refs[i]["variant"]))
    sub_by_cluster = {s["cluster"]: s for s in subs}

    display_peak: dict[str, str] = {}
    if args.display_peak_tsv:
        _, rows = read_tsv(args.display_peak_tsv)
        for r in rows:
            display_peak[stem_of(next(iter(r.values())))] = list(r.values())[1]

    out_cols = [
        "variant", "hdr_path", "sdr_path", "category",
        "primaries", "primaries_name", "transfer", "transfer_name",
        "cluster_id", "cluster_size", "rank_in_cluster", "dist_to_centroid",
        "substituted", "displaced_variant", "displaced_primaries",
        "substitution_reason", "delta_dist_vs_nearest",
        "display_peak_nits",
    ] + [c.replace("feat_", "") for c in HDR_AXIS_COLS] + ["hdr_sha256"]

    picked_rows = []
    for i in chosen:
        c = int(labels[i])
        rank = members[c].index(i)
        sub = sub_by_cluster.get(c)
        axes = hdr_axes.get(refs[i]["variant"], {})
        row = {
            "variant": refs[i]["variant"],
            "hdr_path": refs[i]["hdr_path"],
            "sdr_path": refs[i]["sdr_path"],
            "category": refs[i]["category"],
            "primaries": refs[i]["primaries"],
            "primaries_name": PRIMARIES_NAMES.get(refs[i]["primaries"], "?"),
            "transfer": refs[i]["transfer"],
            "transfer_name": TRANSFER_NAMES.get(refs[i]["transfer"], "?"),
            "cluster_id": c,
            "cluster_size": sizes[c],
            "rank_in_cluster": rank,
            "dist_to_centroid": f"{float(dists[i]):.6f}",
            "substituted": 1 if sub else 0,
            "displaced_variant": refs[sub["displaced"]]["variant"] if sub else "",
            "displaced_primaries": refs[sub["displaced"]]["primaries"] if sub else "",
            "substitution_reason": sub["reason"] if sub else "",
            "delta_dist_vs_nearest": f"{sub['delta_dist']:.6f}" if sub else "",
            "display_peak_nits": display_peak.get(refs[i]["variant"], ""),
            "hdr_sha256": "" if args.no_sha else sha256_of(pathlib.Path(refs[i]["hdr_path"])),
        }
        for col in HDR_AXIS_COLS:
            row[col.replace("feat_", "")] = axes.get(col, "")
        picked_rows.append(row)
    picked_rows.sort(key=lambda r: (r["cluster_id"], r["variant"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("\t".join(out_cols) + "\n")
        for r in picked_rows:
            fh.write("\t".join(str(r[c]) for c in out_cols) + "\n")

    # ---- report -------------------------------------------------------
    print()
    print(crosstab(refs, f"full corpus (n={len(refs)}): primaries x content class"))
    print()
    print(crosstab(picked_rows, f"G0.5 PICKS (n={len(picked_rows)}): primaries x content class"))

    pc = collections.Counter(r["primaries"] for r in picked_rows)
    print()
    for p in sorted(pc):
        print(f"  primaries {p} ({PRIMARIES_NAMES.get(p, '?')}): {pc[p]} picks "
              f"(G0.5 floor {MIN_PER_PRIMARIES}) "
              f"{'OK' if pc[p] >= MIN_PER_PRIMARIES else 'FAIL'}")

    if subs:
        print(f"\n== substitutions ({len(subs)}) ==")
        for s in subs:
            print(f"  cluster {s['cluster']:>2}: {refs[s['chosen']]['variant']} "
                  f"(rank {s['rank']}) displaced {refs[s['displaced']]['variant']} "
                  f"(rank 0); +{s['delta_dist']:.4f} distance — {s['reason']}")
    else:
        print("\n== substitutions: none (centroid-nearest already satisfied every constraint) ==")

    if peak_vals is not None:
        sel_peaks = [peak_vals[i] for i in chosen]
        bc = collections.Counter(peak_bins[i] for i in chosen)
        print(f"\npeak luminance across picks: {min(sel_peaks):.1f} .. {max(sel_peaks):.1f} "
              f"cd/m^2 (corpus {min(peak_vals):.1f} .. {max(peak_vals):.1f}); "
              f"bin coverage {[bc[b] for b in range(args.peak_bins)]}")

    manifest = {
        "gate": "G0.4 + G0.5",
        "plan": "benchmarks/avif_hdr_arm_plan_2026-09-02.md §4.3",
        "k": args.k,
        "seed": args.seed,
        "n_init": args.n_init,
        "kmeans": "scipy.cluster.vq.kmeans2(minit='++')",
        "scipy_version": __import__("scipy").__version__,
        "numpy_version": np.__version__,
        "clustered_on": "paired .sdr.png (plan §4.3)",
        "n_clustering_features": len(kept_cols),
        "clustering_features": kept_cols,
        "inertia": inertia,
        "min_per_primaries": MIN_PER_PRIMARIES,
        "peak_bins": args.peak_bins,
        "min_per_peak_bin": args.min_per_peak_bin,
        "n_substitutions": len(subs),
        "inputs": {
            "precheck_tsv": str(args.precheck_tsv),
            "sdr_features": str(args.sdr_features),
            "hdr_features": str(args.hdr_features) if args.hdr_features else None,
        },
        "input_sha256": {
            k: sha256_of(p)
            for k, p in (
                ("precheck_tsv", args.precheck_tsv),
                ("sdr_features", args.sdr_features),
                ("hdr_features", args.hdr_features),
            )
            if p is not None
        },
        "display_peak_nits": (
            "EMPTY — no built binary exposes HdrRef.display_peak_nits per file; "
            "see the module docstring. peak_luminance_nits (zenanalyze tier_depth) "
            "is carried instead and is a DIFFERENT measurement."
            if not display_peak else str(args.display_peak_tsv)
        ),
    }
    mpath = args.out.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {len(picked_rows)} picks -> {args.out}")
    print(f"wrote provenance   -> {mpath}")

    if any(pc[p] < MIN_PER_PRIMARIES for p in pc):
        print("G0.5 FAIL", file=sys.stderr)
        return 1
    print("G0.4 + G0.5 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
