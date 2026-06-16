#!/usr/bin/env python3
"""Build per-codec training parquets via 3-way join of:

  ┌──────────────────────────────────────────────────────────────────────┐
  │ DEPRECATED — use the Rust `zenmetrics assemble` subcommand instead.   │
  │                                                                        │
  │   cargo build --release -p zenmetrics-cli --features assemble         │
  │   zenmetrics assemble --runs <run> --cache-dir <dir> --out-dir <dir>  │
  │                                                                        │
  │ The Rust port (crates/zenmetrics-cli/src/assemble/) reproduces this   │
  │ exact 3-way join but with a TYPED full-key (`PairKey`) that makes the  │
  │ 2026-05-25 parquet corruption (ssim2_gpu ref-misjoin + iwssim mock /   │
  │ human-copy leak) structurally impossible — see                         │
  │ zensim/benchmarks/DATA_INTEGRITY_root_cause_2026-05-25.md and the      │
  │ four guards ported from zensim/scripts/canonical_corpus/join_safety.py.│
  │ This Python script is kept only as a fallback / reference and uses     │
  │ DuckDB's ref-only-collapsing merge that caused the original bug.       │
  └──────────────────────────────────────────────────────────────────────┘

  1. omni/<chunk>.parquet        — per-cell metric scores + encode/decode stats
  2. zensim_features/<chunk>.parquet — per-cell 300-D zensim feature vector
  3. source_features/<chunk>.parquet — per-source 62+ zenanalyze features

Inner-joined on:
  - per-cell join (omni × zensim_features): (image_path, codec, q, knob_tuple_json)
  - per-source join (... × source_features): image_basename (extracted from image_path)

Output: one parquet per codec at <out_dir>/<codec>_training.parquet.

Renames colliding `feat_<N>` columns:
  - zensim feature vector: feat_<N> → zsm_feat_<N>
  - zenanalyze source features: feat_<N> → src_feat_<N>

Run from anywhere; downloads sidecars to a local cache directory
the first time, then re-uses.

Usage:
  python3 build_per_codec_training.py \\
      --runs cvvdp-v15rc-2026-05-18 omni-multi-codec-2026-05-19 \\
      --cache-dir /tmp/training-cache \\
      --out-dir /mnt/v/zen/zensim-training/2026-05-19/per_codec
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Cross-repo import of zensim's join_safety. The DuckDB joins below use the
# correct full per-pair key (image_path, codec, q, knob_tuple_json) — but
# they still need a Mode-A leak guard + Mode-B constant-per-ref guard on
# the final per-codec parquet so a future schema/source change can't silently
# re-introduce the kadid/tid corruption shape.
_ZEN_CORPUS_JOIN_DIR = Path("/home/lilith/work/zen/zensim/scripts/canonical_corpus")
if str(_ZEN_CORPUS_JOIN_DIR) not in sys.path:
    sys.path.insert(0, str(_ZEN_CORPUS_JOIN_DIR))
try:
    from join_safety import guard_metric_table  # noqa: E402
except ImportError:
    def guard_metric_table(label, table, *, source_key=None):  # type: ignore
        print(
            f"WARN: join_safety not on path ({_ZEN_CORPUS_JOIN_DIR} missing); "
            f"skipping guard_metric_table({label!r})",
            file=sys.stderr,
        )

R2_ENDPOINT = os.environ.get("R2_ENDPOINT") or (
    f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    if "R2_ACCOUNT_ID" in os.environ
    else None
)


def s5cmd(*args):
    """Run s5cmd with the R2 endpoint + profile."""
    cmd = ["s5cmd", "--endpoint-url", R2_ENDPOINT, "--profile", "r2", *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def sync_prefix(r2_prefix: str, local_dir: Path) -> int:
    """Download every parquet under r2_prefix to local_dir. Returns count."""
    local_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in local_dir.glob("*.parquet")}
    listing = s5cmd("ls", r2_prefix).stdout.strip().splitlines()
    to_download = []
    for line in listing:
        parts = line.split()
        if not parts:
            continue
        fname = parts[-1]
        if fname.endswith(".parquet") and fname not in existing:
            to_download.append(fname)
    if to_download:
        # Use s5cmd run for parallel downloads.
        runfile = local_dir / "_dl.run"
        with open(runfile, "w") as f:
            for fname in to_download:
                f.write(f"cp {r2_prefix}{fname} {local_dir / fname}\n")
        subprocess.run(
            [
                "s5cmd", "--endpoint-url", R2_ENDPOINT, "--profile", "r2",
                "--numworkers", "32",
                "run", str(runfile),
            ],
            check=True,
        )
        runfile.unlink()
    return len(existing) + len(to_download)


def load_concat(local_dir: Path, expected_n: int) -> pa.Table:
    """Read all parquets in local_dir via DuckDB (auto-unifies dtypes
    across files, coercing per-column to the broadest type seen).
    pyarrow's concat_tables is strict (double vs string mismatch is
    fatal); DuckDB widens to the common supertype, which is what we
    want for sidecars where a per-cell score occasionally went null /
    empty string on metric-fail rows.
    """
    files = sorted(local_dir.glob("*.parquet"))
    if len(files) != expected_n:
        print(f"  WARN: {local_dir.name}: {len(files)} files, expected {expected_n}", file=sys.stderr)
    if not files:
        return pa.table({})
    import duckdb
    glob = str(local_dir / "*.parquet")
    # .arrow() returns a RecordBatchReader; materialize into a Table.
    reader = duckdb.query(
        f"SELECT * FROM read_parquet('{glob}', union_by_name=true)"
    ).arrow()
    if hasattr(reader, "read_all"):
        return reader.read_all()
    return reader


def rename_feat_columns(t: pa.Table, prefix: str, keep: set[str]) -> pa.Table:
    """Rename every `feat_<N>` column to `<prefix>feat_<N>`; keep cols in `keep` untouched."""
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
    """Add a column with the basename of `src_col` (last `/`-separated component)."""
    if dst_col in t.column_names:
        return t
    paths = t[src_col].to_pylist()
    basenames = [p.rsplit("/", 1)[-1] for p in paths]
    return t.append_column(dst_col, pa.array(basenames, pa.string()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="R2 run ids to ingest")
    ap.add_argument("--cache-dir", required=True, help="Local cache for downloaded sidecars")
    ap.add_argument("--out-dir", required=True, help="Where to write per-codec parquets")
    ap.add_argument(
        "--codecs",
        nargs="+",
        default=None,
        help="Filter to these codec column values (default: all codecs found)",
    )
    args = ap.parse_args()

    if R2_ENDPOINT is None:
        sys.exit("error: R2_ENDPOINT or R2_ACCOUNT_ID env required")

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step A: download every sidecar from R2 into cache_dir/<run>/<kind>/
    print("=== STEP A: sync sidecars from R2 ===")
    for run in args.runs:
        for kind in ("omni", "zensim_features", "source_features"):
            r2_prefix = f"s3://zentrain/{run}/{kind}/"
            local = cache_dir / run / kind
            print(f"  syncing {r2_prefix} → {local}")
            n = sync_prefix(r2_prefix, local)
            print(f"    {n} parquets in cache")

    # Step B: load all sidecars via DuckDB's union_by_name (handles
    # cross-file + cross-run dtype promotion for partial / failed cells).
    print("\n=== STEP B: load + concat per run ===")
    import duckdb

    def _load_all(kind: str) -> pa.Table:
        globs = [str((cache_dir / run / kind / "*.parquet").as_posix()) for run in args.runs]
        # DuckDB's read_parquet auto-promotes across all matched files.
        union_sql = " UNION ALL BY NAME ".join(
            f"SELECT * FROM read_parquet('{g}', union_by_name=true)" for g in globs
        )
        r = duckdb.query(union_sql).arrow()
        return r.read_all() if hasattr(r, "read_all") else r

    omni = _load_all("omni")
    print(f"  omni: {omni.num_rows} rows × {omni.num_columns} cols")
    zsm = _load_all("zensim_features")
    print(f"  zsm:  {zsm.num_rows} rows × {zsm.num_columns} cols")
    src = _load_all("source_features")
    print(f"  src:  {src.num_rows} rows × {src.num_columns} cols")

    # Step C: rename collisions
    print("\n=== STEP C: namespace feat_<N> columns ===")
    join_keys_cell = {"image_path", "codec", "q", "knob_tuple_json"}
    zsm = rename_feat_columns(zsm, "zsm_", keep=join_keys_cell | {"zensim_score"})
    # Drop zsm's identity tuple columns to avoid duplicates after join
    src_join_keys = {"image_basename", "width", "height"}
    src = rename_feat_columns(src, "src_", keep=src_join_keys | {"chunk_id", "run_id", "image_path"})

    # Step D: 3-way join via DuckDB
    print("\n=== STEP D: 3-way join ===")
    # Add image_basename to omni for join with src + register everything
    omni = basename_col(omni, "image_path", "image_basename")
    duckdb.register("omni", omni)
    duckdb.register("zsm", zsm)
    duckdb.register("src", src)

    # Helper: duckdb.query(...).arrow() returns a RecordBatchReader;
    # materialize to a Table. Also ensure tables referenced by name
    # in SQL are registered first (DuckDB needs explicit
    # `register(name, arrow_table)` — auto-magic name resolution
    # only works inside the relational API).
    def _ddb_table(sql: str) -> pa.Table:
        r = duckdb.query(sql).arrow()
        return r.read_all() if hasattr(r, "read_all") else r

    # Trim src to one row per (run_id, image_basename) — dedupe across chunks
    src_dedupe = _ddb_table("""
        SELECT *
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY run_id, image_basename ORDER BY chunk_id) AS rn
            FROM src
        )
        WHERE rn = 1
    """)
    print(f"  src_dedupe: {src_dedupe.num_rows} unique (run_id, image_basename)")
    duckdb.register("src_dedupe", src_dedupe)

    # Inner-join omni × zsm on (image_path, codec, q, knob_tuple_json)
    joined_cell = _ddb_table("""
        SELECT
            omni.image_path,
            omni.codec,
            omni.q,
            omni.knob_tuple_json,
            omni.encoded_bytes,
            omni.encode_ms,
            omni.decode_ms,
            omni.encoded_filename,
            omni.score_zensim_gpu,
            omni.score_ssim2_gpu,
            omni.score_butteraugli_max_gpu,
            omni.score_butteraugli_pnorm3_gpu,
            omni.score_cvvdp_imazen_v0_0_1,
            omni.score_dssim_gpu,
            omni.score_iwssim_gpu,
            omni.run_id,
            omni.chunk_id,
            omni.image_basename,
            zsm.* EXCLUDE(image_path, codec, q, knob_tuple_json)
        FROM omni
        INNER JOIN zsm USING (image_path, codec, q, knob_tuple_json)
    """)
    print(f"  joined_cell: {joined_cell.num_rows} rows × {joined_cell.num_columns} cols")
    duckdb.register("joined_cell", joined_cell)

    # Drop zsm-side identity dup (handled by EXCLUDE above; ensure zensim_score kept)

    # Now join × src on (run_id, image_basename)
    joined = _ddb_table("""
        SELECT
            jc.*,
            src.* EXCLUDE(run_id, image_basename, chunk_id, image_path)
        FROM joined_cell jc
        INNER JOIN src_dedupe src
            ON jc.run_id = src.run_id AND jc.image_basename = src.image_basename
    """)
    print(f"  final joined: {joined.num_rows} rows × {joined.num_columns} cols")
    duckdb.register("joined", joined)

    # Step E: split per codec + write
    print("\n=== STEP E: split per codec + write ===")
    codecs_in_data = set(joined["codec"].unique().to_pylist())
    codecs_to_write = args.codecs or sorted(codecs_in_data)
    for codec in codecs_to_write:
        if codec not in codecs_in_data:
            print(f"  WARN: requested codec {codec!r} not in data")
            continue
        sub = _ddb_table(f"SELECT * FROM joined WHERE codec = '{codec}'")
        out_path = out_dir / f"{codec}_training.parquet"
        # Pre-write Mode-A + Mode-B guards. Raises on any mock column, any
        # ssim2/cvvdp/butter/dssim column bit-identical to human_score, OR
        # any of those metric columns constant within every image_basename
        # group (the 2026-05-25 corruption signature). Source key is
        # image_basename because that's the ref-only key the per-source
        # join carries downstream; per-pair score columns SHOULD vary
        # within each image_basename group across (codec, q, knob).
        guard_metric_table(
            f"build_per_codec_training[{codec}]",
            sub,
            source_key="image_basename" if "image_basename" in sub.schema.names else None,
        )
        pq.write_table(sub, out_path, compression="zstd", compression_level=3)
        print(f"  {codec}: {sub.num_rows} rows → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")

    print("\n✅ done")


if __name__ == "__main__":
    main()
