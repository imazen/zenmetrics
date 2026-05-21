#!/usr/bin/env python3
"""Extended per-codec training-parquet builder that merges three
R2 prefixes into a single per-codec parquet:

  1. cvvdp-v15rc-2026-05-18      — 2568 omni + zensim_features + source_features
  2. omni-multi-codec-2026-05-19 — 365  omni + zensim_features + source_features
  3. multi-codec-2026-05-18      — 112  omni ONLY (no zensim_features sibling,
                                    no preserved encoded variants, but sources
                                    overlap with mc05-19's 200-image gen-* corpus
                                    so existing source_features are reusable)

This is a left-outer extension of `build_per_codec_training.py`:
the zensim_features (300-D `zsm_feat_*`) join is LEFT OUTER so mc18
rows are kept even though they lack a `zensim_features/` sibling.
Source features join is INNER on `image_basename` since the picker
requires those.

Output per codec at <out_dir>/<codec>_training.parquet, schema:

  image_path, codec, q, knob_tuple_json,
  encoded_bytes, encode_ms, decode_ms,
  score_cvvdp_imazen_v0_0_1, score_ssim2_gpu, score_butteraugli_max_gpu,
  score_butteraugli_pnorm3_gpu, score_dssim_gpu, score_iwssim_gpu,
  score_zensim_gpu, zensim_score, width, height,
  src_feat_<id>...      — zenanalyze per-source (always present)
  zsm_feat_<id>...      — zensim per-pair (present for mc05-19 + cvvdp-v15rc,
                          NULL for mc18 rows; downstream consumers either
                          ignore or filter on these)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

R2_ENDPOINT = os.environ.get("R2_ENDPOINT") or (
    f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    if "R2_ACCOUNT_ID" in os.environ
    else None
)


def s5cmd(*args):
    cmd = ["s5cmd", "--endpoint-url", R2_ENDPOINT, "--profile", "r2", *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def sync_prefix(r2_prefix: str, local_dir: Path, expected_kind: str) -> int:
    """Download every parquet under r2_prefix to local_dir. Returns
    count. Tolerates missing prefixes (returns 0) so callers can
    request optional kinds (e.g. mc18 has no zensim_features)."""
    local_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in local_dir.glob("*.parquet")}
    res = s5cmd("ls", r2_prefix)
    if res.returncode != 0 or not res.stdout.strip():
        print(f"  WARN ({expected_kind}): no objects under {r2_prefix}", file=sys.stderr)
        return len(existing)
    listing = res.stdout.strip().splitlines()
    to_download = []
    for line in listing:
        parts = line.split()
        if not parts:
            continue
        fname = parts[-1]
        if fname.endswith(".parquet") and fname not in existing:
            to_download.append(fname)
    if to_download:
        runfile = local_dir / "_dl.run"
        with open(runfile, "w") as f:
            for fname in to_download:
                f.write(f"cp {r2_prefix}{fname} {local_dir / fname}\n")
        subprocess.run(
            ["s5cmd", "--endpoint-url", R2_ENDPOINT, "--profile", "r2",
             "--numworkers", "32", "run", str(runfile)],
            check=True,
        )
        runfile.unlink()
    return len(existing) + len(to_download)


def rename_feat_columns(t: pa.Table, prefix: str, keep: set) -> pa.Table:
    new_names = []
    for name in t.column_names:
        if name in keep:
            new_names.append(name)
        elif name.startswith("feat_"):
            new_names.append(f"{prefix}{name}")
        else:
            new_names.append(name)
    return t.rename_columns(new_names)


def basename_col(t: pa.Table, src_col: str = "image_path", dst_col: str = "image_basename") -> pa.Table:
    if dst_col in t.column_names:
        return t
    paths = t[src_col].to_pylist()
    basenames = [p.rsplit("/", 1)[-1] for p in paths]
    return t.append_column(dst_col, pa.array(basenames, pa.string()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["cvvdp-v15rc-2026-05-18",
                             "omni-multi-codec-2026-05-19",
                             "multi-codec-2026-05-18"],
                    help="R2 run ids to ingest")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--codecs", nargs="+", default=None)
    args = ap.parse_args()

    if R2_ENDPOINT is None:
        sys.exit("error: R2_ENDPOINT or R2_ACCOUNT_ID env required")

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # STEP A: sync. zensim_features is OPTIONAL per-run.
    print("=== STEP A: sync sidecars ===")
    for run in args.runs:
        for kind in ("omni", "zensim_features", "source_features"):
            r2_prefix = f"s3://zentrain/{run}/{kind}/"
            local = cache_dir / run / kind
            print(f"  syncing {r2_prefix}")
            n = sync_prefix(r2_prefix, local, kind)
            print(f"    -> {n} parquets in cache")

    # STEP B: load with DuckDB unions (handles cross-file dtype promotion).
    print("\n=== STEP B: load + concat across runs ===")
    import duckdb

    def _load_kind(kind: str, runs_subset=None) -> pa.Table:
        runs_use = runs_subset or args.runs
        globs = []
        for r in runs_use:
            d = cache_dir / r / kind
            if any(d.glob("*.parquet")):
                globs.append(str((d / "*.parquet").as_posix()))
        if not globs:
            return pa.table({})
        sql = " UNION ALL BY NAME ".join(
            f"SELECT * FROM read_parquet('{g}', union_by_name=true)" for g in globs)
        r = duckdb.query(sql).arrow()
        return r.read_all() if hasattr(r, "read_all") else r

    omni = _load_kind("omni")
    print(f"  omni  : {omni.num_rows} rows × {omni.num_columns} cols")
    zsm = _load_kind("zensim_features")
    print(f"  zsm   : {zsm.num_rows} rows × {zsm.num_columns} cols")
    src = _load_kind("source_features")
    print(f"  src   : {src.num_rows} rows × {src.num_columns} cols")

    # STEP C: namespace feat_ columns
    print("\n=== STEP C: namespace feat_<N> columns ===")
    cell_keys = {"image_path", "codec", "q", "knob_tuple_json"}
    if zsm.num_rows > 0:
        zsm = rename_feat_columns(zsm, "zsm_", keep=cell_keys | {"zensim_score"})
    src_keys = {"image_basename", "width", "height"}
    src = rename_feat_columns(src, "src_", keep=src_keys | {"chunk_id", "run_id", "image_path"})

    # STEP D: 3-way join
    print("\n=== STEP D: joins ===")
    omni = basename_col(omni, "image_path", "image_basename")
    duckdb.register("omni", omni)
    duckdb.register("src", src)

    def _ddb(sql):
        r = duckdb.query(sql).arrow()
        return r.read_all() if hasattr(r, "read_all") else r

    # Dedupe source features by (run_id, image_basename) → keep one row per source.
    # Then for sources missing in src.run_id (mc18 has no source_features sidecar),
    # we'll fall back to ANY existing source_features row by basename only.
    src_dedupe = _ddb("""
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY image_basename ORDER BY chunk_id) AS rn
            FROM src
        ) WHERE rn = 1
    """)
    print(f"  src_dedupe: {src_dedupe.num_rows} unique image_basename rows")
    duckdb.register("src_dedupe", src_dedupe)

    if zsm.num_rows > 0:
        duckdb.register("zsm", zsm)
        # LEFT join on omni × zsm — mc18 rows will have NULL zsm_feat_*
        joined_cell = _ddb("""
            SELECT
                omni.image_path, omni.codec, omni.q, omni.knob_tuple_json,
                omni.encoded_bytes, omni.encode_ms, omni.decode_ms,
                omni.encoded_filename,
                omni.score_zensim_gpu, omni.score_ssim2_gpu,
                omni.score_butteraugli_max_gpu, omni.score_butteraugli_pnorm3_gpu,
                omni.score_cvvdp_imazen_v0_0_1, omni.score_dssim_gpu, omni.score_iwssim_gpu,
                omni.run_id, omni.chunk_id, omni.image_basename,
                zsm.* EXCLUDE(image_path, codec, q, knob_tuple_json)
            FROM omni
            LEFT JOIN zsm USING (image_path, codec, q, knob_tuple_json)
        """)
    else:
        joined_cell = _ddb("""
            SELECT
                omni.image_path, omni.codec, omni.q, omni.knob_tuple_json,
                omni.encoded_bytes, omni.encode_ms, omni.decode_ms,
                omni.encoded_filename,
                omni.score_zensim_gpu, omni.score_ssim2_gpu,
                omni.score_butteraugli_max_gpu, omni.score_butteraugli_pnorm3_gpu,
                omni.score_cvvdp_imazen_v0_0_1, omni.score_dssim_gpu, omni.score_iwssim_gpu,
                omni.run_id, omni.chunk_id, omni.image_basename
            FROM omni
        """)
    print(f"  joined_cell: {joined_cell.num_rows} rows × {joined_cell.num_columns} cols")
    duckdb.register("joined_cell", joined_cell)

    # Inner join × src_dedupe on image_basename only (mc18 + mc05-19 share the gen-* corpus
    # so the basename match is sufficient).
    joined = _ddb("""
        SELECT jc.*,
               src.* EXCLUDE(run_id, image_basename, chunk_id, image_path)
        FROM joined_cell jc
        INNER JOIN src_dedupe src
            ON jc.image_basename = src.image_basename
    """)
    print(f"  final: {joined.num_rows} rows × {joined.num_columns} cols")
    duckdb.register("joined", joined)

    # STEP E: split per codec + write
    print("\n=== STEP E: split per codec + write ===")
    codecs_in_data = set(joined["codec"].unique().to_pylist())
    codecs_to_write = args.codecs or sorted(codecs_in_data)
    for codec in codecs_to_write:
        if codec not in codecs_in_data:
            print(f"  WARN: requested codec {codec!r} not in data")
            continue
        # Combine like-named codec aliases (v12_zenwebp + zenwebp → zenwebp).
        canonical = codec
        for alias in (codec, f"v12_{codec}", f"v13_{codec}", f"v15rc_{codec}"):
            if alias in codecs_in_data and alias != canonical:
                pass
        sub = _ddb(f"SELECT * FROM joined WHERE codec = '{codec}' "
                   f"OR codec = 'v12_{codec}' OR codec = 'v13_{codec}' "
                   f"OR codec = 'v15rc_{codec}'")
        out_path = out_dir / f"{codec}_training.parquet"
        pq.write_table(sub, out_path, compression="zstd", compression_level=3)
        print(f"  {codec}: {sub.num_rows} rows → {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    print("\n✅ done")


if __name__ == "__main__":
    main()
