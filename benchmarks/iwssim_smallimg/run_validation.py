#!/usr/bin/env python3
"""
Adaptive IW-SSIM small-image validation harness.

Question: when IW-SSIM is forced to evaluate sub-176-px inputs via
reflect-pad (the IwssimConfig::allow_small=true path), does the score
still rank distortion-severity in agreement with ssim2 / cvvdp /
butteraugli on the same pairs?

Strategy. There is no human-rated small-image perceptual corpus
readily available (CID22 = 512², KADID-10k = 512×384). We synthesize
small-image pairs from CID22 by Lanczos downsampling references to
{64, 96, 128, 176} px and re-encoding each at q in {5, 20, 50, 80, 95}
with zenjpeg. The reference metric for "this pair is more distorted
than that pair" is the q value (lower q = more distortion). We score
each pair with:

  - iwssim (allow_small=true, reflect-pad to 176)
  - iwssim-upscale-176 (Lanczos to 176 on the host, then stock iwssim)
  - ssim2 (handles small inputs natively, our anchor)
  - butteraugli (handles small natively)
  - cvvdp (requires fp32 + LP pyramid like iwssim; check whether it
    even runs at small dim, document the answer either way)

We then compute per-(image, native_dim):
  - Spearman rho between each metric's q-vs-score curve and the
    "ground truth" q ordering (-q-of-pair is the better-quality dir).
  - For each (q1, q2) pair where q1 != q2, does the metric agree
    with q-ordering on which of the two pairs is more distorted?
    Count agreement rate per metric per native_dim.

The pass criterion is set at the bottom: if iwssim-reflect-pad's
Spearman rho with ssim2 is >= 0.85 across all native_dims and the
rank-flip rate vs ssim2 is <= 10%, we ship the adaptive default
unmodified. If not, the upscale-to-176 variant gets a fair second
look, and tile-to-176 becomes a candidate strategy worth implementing
in iwssim-gpu proper.

Reference for the rule against synthetic upscale features:
~/.claude/CLAUDE.md "Sweep / Calibration / Source-informing
Benchmark Discipline" -- we are NOT training a model here, we are
*measuring agreement of a quality metric on already-downsampled
inputs*. Downsampling is fine; the model-training upscale-ban does
not apply to the eval harness.
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

# ─────────────────────────── configuration ────────────────────────────

# Where the input CID22 originals live.
CID22_DIR = Path("/mnt/v/dataset/cid22/CID22_validation_set/original")

# Where this harness stages refs / distortions / scores.
HARNESS_DIR = Path("/home/lilith/iwssim-val")

# Native dims under test. 176 is the boundary case (stock iwssim
# threshold); the rest are below it.
NATIVE_DIMS = [64, 96, 128, 176]

# Quality steps spanning aggressive to near-lossless. Picker training
# territory.
Q_STEPS = [5, 20, 50, 80, 95]

# Image sample size. 100 refs * 4 dims * 5 q's = 2000 pairs. ~5 min
# of GPU scoring + a few minutes of encoding.
NUM_SOURCES = 100

# RNG seed for reproducibility.
SEED = 20260517

# Pass criteria — see module docstring.
SPEARMAN_PASS = 0.85
RANK_FLIP_PASS = 0.10  # max disagreement rate vs ssim2

# Binary path.
ZEN_METRICS = HARNESS_DIR / "zenmetrics"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─────────────────────────── data prep ────────────────────────────────


def pick_sources() -> list[Path]:
    """Pick NUM_SOURCES CID22 originals at random (seeded)."""
    all_refs = sorted(CID22_DIR.glob("*.png"))
    if len(all_refs) == 0:
        raise SystemExit(f"no PNGs in {CID22_DIR}")
    rng = random.Random(SEED)
    return rng.sample(all_refs, min(NUM_SOURCES, len(all_refs)))


def downsample_lanczos(src: Path, dst: Path, target_dim: int) -> tuple[int, int]:
    """Downsample `src` so the long axis is `target_dim` (preserving
    aspect ratio). Returns the resulting (w, h)."""
    im = Image.open(src).convert("RGB")
    w, h = im.size
    if w >= h:
        new_w = target_dim
        new_h = max(1, round(h * target_dim / w))
    else:
        new_h = target_dim
        new_w = max(1, round(w * target_dim / h))
    out = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
    out.save(dst)
    return new_w, new_h


def encode_jpeg_via_zenmetrics(ref: Path, q: int, out: Path) -> None:
    """Use zenmetrics 'sweep' to re-encode one image at one q.

    Why not just PIL: zenjpeg's encoding path is what production uses,
    and using `PIL.Image.save(format='JPEG')` would test a different
    codec than what'll ship. Sweep also pairs ref/dist automatically
    via the pairs-tsv output, which our scorer wants downstream.

    We run sweep with a tiny per-image group: one source, one q step.
    This is slower than batched sweep but keeps the harness simple
    and deterministic per ref.
    """
    # Run sweep on a single-image source dir.
    src_dir = out.parent / "_src" / f"{ref.stem}"
    src_dir.mkdir(parents=True, exist_ok=True)
    src_link = src_dir / ref.name
    if src_link.exists() or src_link.is_symlink():
        src_link.unlink()
    src_link.symlink_to(ref.resolve())
    dist_dir = out.parent / "_dist" / f"{ref.stem}_q{q}"
    dist_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(ZEN_METRICS),
        "sweep",
        "--codec", "zenjpeg",
        "--sources", str(src_dir),
        "--q-grid", str(q),
        "--metric", "ssim2",  # cheapest CPU metric, we only want the dist file
        "--output", str(out.parent / f"_sweep_{ref.stem}_q{q}.tsv"),
        "--pairs-tsv", str(out.parent / f"_pairs_{ref.stem}_q{q}.tsv"),
        "--distorted-out-dir", str(dist_dir),
        "--jobs", "1",
    ]
    rc = subprocess.run(cmd, capture_output=True, text=True)
    if rc.returncode != 0:
        log(f"sweep FAIL ref={ref.name} q={q}: {rc.stderr[-300:]}")
        return None
    # zenmetrics sweep writes the round-tripped DECODED PNG of the
    # JPEG encode to --distorted-out-dir (not the .jpg itself). That's
    # what we want for scoring — the distorted decoded image. Find it.
    dist_candidates = list(dist_dir.rglob("*.png")) + list(dist_dir.rglob("*.jpg")) + list(dist_dir.rglob("*.jpeg"))
    if not dist_candidates:
        log(f"sweep produced no dist file for ref={ref.name} q={q}")
        return None
    # Rename / move to the canonical out path (keep .png extension if
    # source was .png so callers downstream don't get confused).
    canon = out.with_suffix(dist_candidates[0].suffix)
    shutil.move(str(dist_candidates[0]), str(canon))
    return canon


# ─────────────────────────── strategy preprocessing ───────────────────


def host_upscale_to_176(src: Path, dst: Path) -> None:
    """Lanczos upscale src to (>=176, >=176) preserving aspect. Used by
    the iwssim-upscale-176 strategy: caller scores against another
    176+ image (the upscaled ref vs the upscaled dist) using stock
    iwssim (no allow_small needed)."""
    im = Image.open(src).convert("RGB")
    w, h = im.size
    if w >= h:
        new_w = max(176, w)
        new_h = max(1, round(h * new_w / w))
        if new_h < 176:
            new_h = 176
            new_w = max(1, round(w * new_h / h))
    else:
        new_h = max(176, h)
        new_w = max(1, round(w * new_h / h))
        if new_w < 176:
            new_w = 176
            new_h = max(1, round(h * new_w / w))
    out = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
    out.save(dst)


def host_tile_to_176(src: Path, dst: Path) -> None:
    """Tile src horizontally + vertically until both dims >= 176.

    Different from reflect-pad: tile *repeats* the content (no
    reflection), giving the iwssim pyramid a periodic signal whose
    boundary statistics are at least as well-behaved as the interior.
    Whether this beats reflect-pad is what the harness will answer.

    Worst case (22-px input -> 176-px tile = 8x replication) the
    pyramid sees a 8-period checkerboard of the source content
    instead of mirrored boundaries.
    """
    im = Image.open(src).convert("RGB")
    w, h = im.size
    rep_w = math.ceil(176 / w)
    rep_h = math.ceil(176 / h)
    tile_w = w * rep_w
    tile_h = h * rep_h
    out = Image.new("RGB", (tile_w, tile_h))
    for y in range(rep_h):
        for x in range(rep_w):
            out.paste(im, (x * w, y * h))
    out.save(dst)


# ─────────────────────────── scoring ──────────────────────────────────


def score_pairs(
    pairs: list[tuple[Path, Path, str]],
    out_parquet: Path,
    metric: str,
    extra_args: list[str] = None,
) -> Path | None:
    """Run `zenmetrics score-pairs --metric <metric>` over the pairs.

    Each entry in `pairs` is `(ref_path, dist_path, ident_str)` where
    ident_str is a short tag we use later to recover the (image,
    native_dim, q, strategy) tuple.
    """
    if extra_args is None:
        extra_args = []
    pairs_tsv = out_parquet.with_suffix(".pairs.tsv")
    with pairs_tsv.open("w") as f:
        f.write("ref_path\tdist_path\timage_path\tcodec\tq\tknob_tuple_json\n")
        for ref, dist, ident in pairs:
            f.write(f"{ref}\t{dist}\t{ident}\ttest\t0\t{{}}\n")
    cmd = [
        str(ZEN_METRICS),
        "score-pairs",
        "--metric", metric,
        "--pairs-tsv", str(pairs_tsv),
        "--out-parquet", str(out_parquet),
        "--gpu-runtime", "cuda",
    ] + extra_args
    env = os.environ.copy()
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = (
        "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    )
    rc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if rc.returncode != 0:
        log(f"score-pairs FAIL metric={metric}: {rc.stderr[-400:]}")
        return None
    return out_parquet


# ─────────────────────────── analysis ─────────────────────────────────


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank-correlation of paired (x, y) lists."""
    n = len(x)
    if n < 2:
        return float("nan")

    def ranks(v):
        # Simple average-rank handling for ties.
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def rank_flip_rate(metric_vals: list[float], ground_truth: list[float]) -> float:
    """Fraction of (i, j) pairs where the metric disagrees with the
    ground truth on the ordering.

    `metric_vals` and `ground_truth` are paired lists. Larger
    metric_val = better quality (for SSIM-family metrics). Smaller
    ground_truth = more distortion (for our q-as-truth scheme).
    Translation: metric agrees with ground truth when both have the
    same sign on (val_i - val_j) for ground_truth_i > ground_truth_j.
    """
    n = len(metric_vals)
    total = 0
    flips = 0
    for i in range(n):
        for j in range(i + 1, n):
            gt_diff = ground_truth[i] - ground_truth[j]
            if gt_diff == 0:
                continue
            m_diff = metric_vals[i] - metric_vals[j]
            if m_diff == 0:
                continue
            total += 1
            # Both should have the same sign (higher q -> higher metric
            # quality score for ssim2/iwssim; for distance metrics
            # like butteraugli/dssim/cvvdp_jod the relationship is
            # inverted — we handle that in the caller by passing
            # negated metric_vals.)
            if (gt_diff > 0) != (m_diff > 0):
                flips += 1
    if total == 0:
        return float("nan")
    return flips / total


# ─────────────────────────── pipeline ─────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prep", "score", "analyze", "all"], default="all")
    ap.add_argument("--num-sources", type=int, default=NUM_SOURCES)
    ap.add_argument("--metrics", nargs="+", default=["ssim2", "iwssim", "butteraugli"])
    ap.add_argument("--include-cvvdp", action="store_true",
                    help="Also score with cvvdp. May fail on small inputs.")
    args = ap.parse_args()

    HARNESS_DIR.mkdir(exist_ok=True, parents=True)
    sources_dir = HARNESS_DIR / "small_refs"
    sources_dir.mkdir(exist_ok=True, parents=True)
    dist_dir = HARNESS_DIR / "dist"
    dist_dir.mkdir(exist_ok=True, parents=True)
    scores_dir = HARNESS_DIR / "scores"
    scores_dir.mkdir(exist_ok=True, parents=True)

    manifest_path = HARNESS_DIR / "manifest.tsv"

    # ── Phase 1: prep — downsample CID22 + encode JPEGs at varied q ──

    if args.phase in ("prep", "all"):
        log("Phase 1: prep")
        srcs = pick_sources()[: args.num_sources]
        log(f"  selected {len(srcs)} CID22 refs")

        manifest_rows = []
        for i, src in enumerate(srcs):
            for d in NATIVE_DIMS:
                ref_out = sources_dir / f"{src.stem}__{d}.png"
                if not ref_out.exists():
                    new_w, new_h = downsample_lanczos(src, ref_out, d)
                else:
                    im = Image.open(ref_out); new_w, new_h = im.size
                for q in Q_STEPS:
                    # dist_path is the decoded distorted PNG (from zenmetrics sweep).
                    dist_stem = dist_dir / f"{src.stem}__{d}__q{q}"
                    existing = list(dist_dir.glob(f"{src.stem}__{d}__q{q}.*"))
                    dist_out = existing[0] if existing else None
                    if dist_out is None:
                        dist_out = encode_jpeg_via_zenmetrics(
                            ref_out, q, dist_stem.with_suffix(".png")
                        )
                    if dist_out is not None and dist_out.exists():
                        manifest_rows.append({
                            "src": src.stem,
                            "native_dim": d,
                            "native_w": new_w,
                            "native_h": new_h,
                            "q": q,
                            "ref_path": str(ref_out),
                            "dist_path": str(dist_out),
                        })
            if (i + 1) % 10 == 0:
                log(f"    {i + 1}/{len(srcs)} refs prepped ({len(manifest_rows)} pairs)")

        log(f"  prep done: {len(manifest_rows)} pairs")
        with manifest_path.open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(manifest_rows)

    # ── Phase 2: score — run metrics over the pair set ──

    if args.phase in ("score", "all"):
        log("Phase 2: score")
        manifest = list(csv.DictReader(manifest_path.open(), delimiter="\t"))
        log(f"  {len(manifest)} pairs in manifest")

        # Strategies to test for iwssim:
        # - "iwssim" with --allow-small-images (reflect-pad, current default)
        # - "iwssim_upscale176" with stock iwssim, after host-side Lanczos upscale
        # - "iwssim_tile176" with stock iwssim, after host-side tile (TODO if reflect-pad falls short)
        #
        # Other reference metrics scored natively (they handle small
        # inputs without preprocessing).

        # Build the upscaled ref/dist set in a sibling dir.
        upscaled_ref_dir = HARNESS_DIR / "upscaled_refs"
        upscaled_dist_dir = HARNESS_DIR / "upscaled_dist"
        upscaled_ref_dir.mkdir(exist_ok=True, parents=True)
        upscaled_dist_dir.mkdir(exist_ok=True, parents=True)

        # Also tile variant.
        tiled_ref_dir = HARNESS_DIR / "tiled_refs"
        tiled_dist_dir = HARNESS_DIR / "tiled_dist"
        tiled_ref_dir.mkdir(exist_ok=True, parents=True)
        tiled_dist_dir.mkdir(exist_ok=True, parents=True)

        for row in manifest:
            ref = Path(row["ref_path"])
            dist = Path(row["dist_path"])
            need_pre = min(int(row["native_w"]), int(row["native_h"])) < 176
            # Always materialize upscale + tile variants for sub-176
            # inputs. For 176+ inputs we skip preprocessing so the
            # stock iwssim path matches what production would do.
            if need_pre:
                u_ref = upscaled_ref_dir / ref.name
                u_dist = upscaled_dist_dir / dist.name
                if not u_ref.exists():
                    host_upscale_to_176(ref, u_ref)
                if not u_dist.exists():
                    host_upscale_to_176(dist, u_dist)
                t_ref = tiled_ref_dir / ref.name
                t_dist = tiled_dist_dir / dist.name
                if not t_ref.exists():
                    host_tile_to_176(ref, t_ref)
                if not t_dist.exists():
                    host_tile_to_176(dist, t_dist)
            else:
                # >= 176: just symlink the native files so the
                # upscale/tile pairs.tsv still has rows for these dims
                # (stock iwssim path is identical to the native run).
                for src_path, dst_dir in [(ref, upscaled_ref_dir), (dist, upscaled_dist_dir),
                                          (ref, tiled_ref_dir), (dist, tiled_dist_dir)]:
                    link = dst_dir / src_path.name
                    if not link.exists():
                        link.symlink_to(src_path.resolve())

        log("  preprocessed upscale + tile variants")

        # Score sets — each metric gets its own pairs list.
        metric_specs = [
            ("ssim2_gpu", "ref_path", "dist_path", []),
            ("butteraugli-gpu", "ref_path", "dist_path", []),
            ("iwssim", "ref_path", "dist_path", ["--allow-small-images"]),
        ]
        if args.include_cvvdp:
            metric_specs.append(("cvvdp", "ref_path", "dist_path", []))

        # Add the upscale + tile iwssim variants — these use stock
        # iwssim (no --allow-small-images), and a different ref/dist
        # column pair.
        # We build separate manifests for those.
        upscaled_pairs_tsv = HARNESS_DIR / "pairs_upscale.tsv"
        tiled_pairs_tsv = HARNESS_DIR / "pairs_tile.tsv"

        with upscaled_pairs_tsv.open("w") as f:
            f.write("ref_path\tdist_path\timage_path\tcodec\tq\tknob_tuple_json\n")
            for row in manifest:
                u_ref = upscaled_ref_dir / Path(row["ref_path"]).name
                u_dist = upscaled_dist_dir / Path(row["dist_path"]).name
                if u_ref.exists() and u_dist.exists():
                    ident = f"{row['src']}__{row['native_dim']}__q{row['q']}"
                    f.write(f"{u_ref}\t{u_dist}\t{ident}\ttest\t{row['q']}\t{{}}\n")
        with tiled_pairs_tsv.open("w") as f:
            f.write("ref_path\tdist_path\timage_path\tcodec\tq\tknob_tuple_json\n")
            for row in manifest:
                t_ref = tiled_ref_dir / Path(row["ref_path"]).name
                t_dist = tiled_dist_dir / Path(row["dist_path"]).name
                if t_ref.exists() and t_dist.exists():
                    ident = f"{row['src']}__{row['native_dim']}__q{row['q']}"
                    f.write(f"{t_ref}\t{t_dist}\t{ident}\ttest\t{row['q']}\t{{}}\n")

        # Main pairs.tsv for native-scoring metrics (ssim2, butter, iwssim-allow_small).
        main_pairs_tsv = HARNESS_DIR / "pairs_main.tsv"
        with main_pairs_tsv.open("w") as f:
            f.write("ref_path\tdist_path\timage_path\tcodec\tq\tknob_tuple_json\n")
            for row in manifest:
                ident = f"{row['src']}__{row['native_dim']}__q{row['q']}"
                f.write(f"{row['ref_path']}\t{row['dist_path']}\t{ident}\ttest\t{row['q']}\t{{}}\n")

        def run_one(name: str, pairs_tsv: Path, metric: str, extra: list[str]):
            out = scores_dir / f"{name}.parquet"
            log(f"  scoring {name} ({metric}) -> {out.name}")
            cmd = [
                str(ZEN_METRICS), "score-pairs",
                "--metric", metric,
                "--pairs-tsv", str(pairs_tsv),
                "--out-parquet", str(out),
                "--gpu-runtime", "cuda",
            ] + extra
            env = os.environ.copy()
            env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
            env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
            t0 = time.time()
            rc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            dt = time.time() - t0
            if rc.returncode != 0:
                log(f"    FAIL ({dt:.1f}s): {rc.stderr[-400:]}")
                return None
            log(f"    OK ({dt:.1f}s)")
            return out

        results = {}
        results["ssim2_gpu"] = run_one("ssim2_gpu", main_pairs_tsv, "ssim2-gpu", [])
        results["butteraugli_gpu"] = run_one("butteraugli_gpu", main_pairs_tsv, "butteraugli-gpu", [])
        results["iwssim_reflect"] = run_one("iwssim_reflect", main_pairs_tsv, "iwssim", ["--allow-small-images"])
        results["iwssim_upscale"] = run_one("iwssim_upscale", upscaled_pairs_tsv, "iwssim", [])
        results["iwssim_tile"] = run_one("iwssim_tile", tiled_pairs_tsv, "iwssim", [])
        if args.include_cvvdp:
            results["cvvdp"] = run_one("cvvdp", main_pairs_tsv, "cvvdp", [])

        log("  scores written")

    # ── Phase 3: analyze — compute Spearman + rank-flip per native_dim ──

    if args.phase in ("analyze", "all"):
        log("Phase 3: analyze")
        manifest = list(csv.DictReader(manifest_path.open(), delimiter="\t"))

        def load_parquet(name: str) -> dict[tuple[str, int, int], float]:
            p = scores_dir / f"{name}.parquet"
            if not p.exists():
                return None
            t = pq.read_table(p).to_pandas()
            return {
                (row["image_path"], int(row["q"]), 0): row.iloc[-1]
                for _, row in t.iterrows()
            }

        # The score-pairs sidecar stores image_path == our ident. Load
        # each strategy, then index by (src, native_dim, q).
        scoresets = {}
        for name in ["ssim2_gpu", "butteraugli_gpu", "iwssim_reflect", "iwssim_upscale", "iwssim_tile"]:
            p = scores_dir / f"{name}.parquet"
            if not p.exists():
                log(f"  missing: {p.name}")
                continue
            t = pq.read_table(p).to_pandas()
            score_col = [c for c in t.columns if c not in ("image_path", "codec", "q", "knob_tuple_json")][0]
            d = {}
            for _, row in t.iterrows():
                ident = row["image_path"]  # form: src__dim__qN
                parts = ident.split("__")
                if len(parts) < 3:
                    continue
                src = parts[0]
                dim = int(parts[1])
                q = int(parts[2].lstrip("q"))
                d[(src, dim, q)] = row[score_col]
            scoresets[name] = d
            log(f"  loaded {name}: {len(d)} scores")

        # Build per-(src, dim) ordered lists across q.
        by_src_dim: dict[tuple[str, int], list[tuple[int, dict[str, float]]]] = defaultdict(list)
        for row in manifest:
            src = row["src"]; dim = int(row["native_dim"]); q = int(row["q"])
            entry = {"q": q}
            for name, d in scoresets.items():
                entry[name] = d.get((src, dim, q), float("nan"))
            by_src_dim[(src, dim)].append((q, entry))

        # Two analyses per native_dim:
        #
        # 1. Per-source Spearman over q (sanity check — strict
        #    monotonicity in q means this is ~1.0 if the metric is
        #    well-behaved at all. A failure here is catastrophic).
        #
        # 2. POOLED across sources at fixed q (the real signal —
        #    "does iwssim_reflect rank these 100 different images at
        #    q=20 the same way ssim2 does?"). This is what tells us
        #    whether iwssim is usable as a quality oracle on small
        #    images.
        #
        # 3. Fully pooled (all sources × all q's per dim) — Spearman
        #    + rank-flip across the entire (src, q) population. The
        #    headline number.
        per_dim = defaultdict(lambda: defaultdict(list))
        # Distance metrics (lower = better) — flip sign so higher
        # = better quality, matching ssim2/iwssim convention.
        distance_metrics = {"butteraugli_gpu"}

        # Per-source Spearman over q (sanity).
        for (src, dim), entries in by_src_dim.items():
            entries.sort(key=lambda x: x[0])
            qs = [e[0] for e in entries]
            vals = {name: [e[1].get(name, float("nan")) for e in entries]
                    for name in scoresets}
            for name in scoresets:
                v = vals[name]
                pairs_q = [(qs[i], v[i]) for i in range(len(qs))
                           if not (isinstance(v[i], float) and math.isnan(v[i]))]
                if len(pairs_q) >= 2:
                    v_signed = [(-x if name in distance_metrics else x) for _, x in pairs_q]
                    q_only = [q for q, _ in pairs_q]
                    rho_q = spearman(q_only, v_signed)
                    flip_q = rank_flip_rate(v_signed, q_only)
                    per_dim[dim][f"{name}_vs_q_rho"].append(rho_q)
                    per_dim[dim][f"{name}_vs_q_flip"].append(flip_q)

        # Pooled across (src, q) per dim — the headline measurement.
        # Gather ALL (src, q) → score pairs into lists per dim per
        # strategy, then compute one Spearman per (dim, strategy).
        pooled = defaultdict(lambda: defaultdict(list))  # dim -> name -> [(src, q, score)]
        for row in manifest:
            src = row["src"]; dim = int(row["native_dim"]); q = int(row["q"])
            for name, d in scoresets.items():
                v = d.get((src, dim, q))
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                pooled[dim][name].append((src, q, v))

        # For each strategy in each dim, compute Spearman against the
        # ssim2 baseline AT MATCHING (src, q). The two scoresets are
        # filtered to their intersection.
        for dim in NATIVE_DIMS:
            ssim2_map = {(src, q): v for (src, q, v) in pooled[dim].get("ssim2_gpu", [])}
            if not ssim2_map:
                continue
            for name in scoresets:
                if name == "ssim2_gpu":
                    # Self-correlation against q (sanity).
                    items = pooled[dim][name]
                    if len(items) < 2:
                        continue
                    q_vals = [q for (_, q, _) in items]
                    v_vals = [v for (_, _, v) in items]
                    per_dim[dim][f"ssim2_gpu_pooled_vs_q_rho"].append(spearman(q_vals, v_vals))
                    per_dim[dim][f"ssim2_gpu_pooled_vs_q_flip"].append(rank_flip_rate(v_vals, q_vals))
                    continue
                # Intersect with ssim2_map.
                aligned = []
                for (src, q, v) in pooled[dim][name]:
                    s = ssim2_map.get((src, q))
                    if s is None: continue
                    v_signed = -v if name in distance_metrics else v
                    aligned.append((s, v_signed, q))
                if len(aligned) < 2: continue
                s_vals = [a[0] for a in aligned]
                v_vals = [a[1] for a in aligned]
                q_vals = [a[2] for a in aligned]
                per_dim[dim][f"{name}_pooled_vs_ssim2_rho"].append(spearman(s_vals, v_vals))
                per_dim[dim][f"{name}_pooled_vs_ssim2_flip"].append(rank_flip_rate(v_vals, s_vals))
                per_dim[dim][f"{name}_pooled_vs_q_rho"].append(spearman(q_vals, v_vals))
                per_dim[dim][f"{name}_pooled_vs_q_flip"].append(rank_flip_rate(v_vals, q_vals))

        # Report.
        report_path = HARNESS_DIR / "report.txt"
        with report_path.open("w") as f:
            def w(msg=""): f.write(msg + "\n"); print(msg)
            w(f"Adaptive IW-SSIM small-image validation — {time.strftime('%Y-%m-%d %H:%M:%S')}")
            w("=" * 80)
            w()
            w(f"Pass criteria: Spearman ρ ≥ {SPEARMAN_PASS} vs ssim2 at every native_dim;")
            w(f"               rank-flip rate ≤ {RANK_FLIP_PASS} vs ssim2.")
            w()
            w(f"Sources: {NUM_SOURCES} CID22 refs (seed={SEED})")
            w(f"Native dims: {NATIVE_DIMS}")
            w(f"Q steps: {Q_STEPS}")
            w()
            w("Per-source Spearman (sanity — monotonic in q means ~1.0):")
            w()
            for dim in NATIVE_DIMS:
                w(f"── native_dim = {dim} px ──")
                stats = per_dim[dim]
                strategies = ["iwssim_reflect", "iwssim_upscale", "iwssim_tile",
                              "butteraugli_gpu"]
                w(f"  {'strategy':<22} {'ρ vs q':>10} {'flip vs q':>10}")
                for s in strategies:
                    rho_q = stats.get(f"{s}_vs_q_rho", [])
                    flip_q = stats.get(f"{s}_vs_q_flip", [])
                    def m(lst): return float("nan") if not lst else sum(lst) / len(lst)
                    w(f"  {s:<22} {m(rho_q):>10.4f} {m(flip_q):>10.4f}")
                if f"ssim2_gpu_vs_q_rho" in stats:
                    rho_q = stats[f"ssim2_gpu_vs_q_rho"]; flip_q = stats[f"ssim2_gpu_vs_q_flip"]
                    def m(lst): return float("nan") if not lst else sum(lst) / len(lst)
                    w(f"  {'ssim2_gpu (anchor)':<22} {m(rho_q):>10.4f} {m(flip_q):>10.4f}")
                w()

            w("Pooled across (src × q) per native_dim — the headline:")
            w()
            for dim in NATIVE_DIMS:
                w(f"── native_dim = {dim} px ──")
                stats = per_dim[dim]
                strategies = ["iwssim_reflect", "iwssim_upscale", "iwssim_tile",
                              "butteraugli_gpu"]
                w(f"  {'strategy':<22} {'ρ vs q':>10} {'flip vs q':>10} {'ρ vs ssim2':>12} {'flip vs ssim2':>14}")
                for s in strategies:
                    rho_q = stats.get(f"{s}_pooled_vs_q_rho", [])
                    flip_q = stats.get(f"{s}_pooled_vs_q_flip", [])
                    rho_s = stats.get(f"{s}_pooled_vs_ssim2_rho", [])
                    flip_s = stats.get(f"{s}_pooled_vs_ssim2_flip", [])
                    def m(lst): return float("nan") if not lst else sum(lst) / len(lst)
                    w(f"  {s:<22} {m(rho_q):>10.4f} {m(flip_q):>10.4f} {m(rho_s):>12.4f} {m(flip_s):>14.4f}")
                if f"ssim2_gpu_pooled_vs_q_rho" in stats:
                    rho_q = stats[f"ssim2_gpu_pooled_vs_q_rho"]; flip_q = stats[f"ssim2_gpu_pooled_vs_q_flip"]
                    def m(lst): return float("nan") if not lst else sum(lst) / len(lst)
                    w(f"  {'ssim2_gpu (anchor)':<22} {m(rho_q):>10.4f} {m(flip_q):>10.4f} {'-':>12} {'-':>14}")
                w()

            w()
            w("Pass/fail per strategy (using POOLED ρ vs ssim2):")
            w(f"  Absolute thresholds (ρ ≥ {SPEARMAN_PASS}, flip ≤ {RANK_FLIP_PASS}):")
            for s in ["iwssim_reflect", "iwssim_upscale", "iwssim_tile"]:
                w(f"    {s}:")
                for dim in NATIVE_DIMS:
                    stats = per_dim[dim]
                    rho_s = stats.get(f"{s}_pooled_vs_ssim2_rho", [])
                    flip_s = stats.get(f"{s}_pooled_vs_ssim2_flip", [])
                    if not rho_s:
                        w(f"      dim={dim}: NO DATA"); continue
                    avg_rho = sum(rho_s) / len(rho_s)
                    avg_flip = sum(flip_s) / len(flip_s)
                    passed = avg_rho >= SPEARMAN_PASS and avg_flip <= RANK_FLIP_PASS
                    tag = "PASS" if passed else "FAIL"
                    w(f"      dim={dim}: ρ={avg_rho:.4f} flip={avg_flip:.4f} {tag}")
            w()

            # Baseline: stock iwssim at dim=176 = the iwssim-ssim2
            # disagreement floor on this corpus. Subtract from each
            # sub-176 dim's result to isolate the small-image-strategy-
            # introduced drift from the iwssim-ssim2 fundamental gap.
            w(f"  Relative to dim=176 stock iwssim baseline (Δρ vs baseline_rho, Δflip vs baseline_flip):")
            for s in ["iwssim_reflect", "iwssim_upscale", "iwssim_tile"]:
                # Use that strategy's own dim=176 number (all three are
                # equal at 176 since no preprocessing happens, but pull
                # the per-strategy value for safety).
                base_rho_lst = per_dim[176].get(f"{s}_pooled_vs_ssim2_rho", [])
                base_flip_lst = per_dim[176].get(f"{s}_pooled_vs_ssim2_flip", [])
                if not base_rho_lst or not base_flip_lst:
                    w(f"    {s}: missing 176-px baseline"); continue
                base_rho = sum(base_rho_lst) / len(base_rho_lst)
                base_flip = sum(base_flip_lst) / len(base_flip_lst)
                w(f"    {s} (baseline: ρ={base_rho:.4f} flip={base_flip:.4f}):")
                for dim in NATIVE_DIMS:
                    if dim == 176: continue
                    stats = per_dim[dim]
                    rho_s = stats.get(f"{s}_pooled_vs_ssim2_rho", [])
                    flip_s = stats.get(f"{s}_pooled_vs_ssim2_flip", [])
                    if not rho_s:
                        w(f"      dim={dim}: NO DATA"); continue
                    avg_rho = sum(rho_s) / len(rho_s)
                    avg_flip = sum(flip_s) / len(flip_s)
                    drho = avg_rho - base_rho
                    dflip = avg_flip - base_flip
                    # PASS if the small-image strategy doesn't make
                    # things meaningfully worse than the 176-px run.
                    # Tolerance: 0.02 Spearman drop, 0.02 flip-rate
                    # increase. Tighter than absolute thresholds and
                    # more meaningful — we're asking whether the
                    # strategy is worse than just running iwssim
                    # natively on a 176-px image.
                    passed = drho >= -0.02 and dflip <= 0.02
                    tag = "PASS" if passed else "FAIL"
                    w(f"      dim={dim}: Δρ={drho:+.4f} Δflip={dflip:+.4f} {tag}")
            w()

            # Tile vs reflect vs upscale: which one is best?
            w("Strategy ranking at each sub-176 native_dim (best ρ vs ssim2 first):")
            for dim in NATIVE_DIMS:
                if dim == 176: continue
                ranks = []
                for s in ["iwssim_reflect", "iwssim_upscale", "iwssim_tile"]:
                    rho_s = per_dim[dim].get(f"{s}_pooled_vs_ssim2_rho", [])
                    if not rho_s: continue
                    ranks.append((sum(rho_s) / len(rho_s), s))
                ranks.sort(reverse=True)
                w(f"  dim={dim}: " + ", ".join(f"{s}({rho:.4f})" for rho, s in ranks))
            w()
            w(f"Full report: {report_path}")
        log(f"  report at {report_path}")


if __name__ == "__main__":
    main()
