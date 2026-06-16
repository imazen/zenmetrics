#!/usr/bin/env python3
"""Generate the v26 zenavif-extended sweep input parquet + chunks.jsonl.

The v26 sweep targets the R²=0.69 butter_p3 weakness identified in
the cvvdp-v15rc / mc18 picker analyses. zenavif's `lrf` (loop
restoration filter) is the key knob — at high q the default preset
turns LRF off, but turning it back on can recover edge quality that
butter_p3 is sensitive to. We sweep 5 knob tuples × 10 q × 1332
sources = 66,600 cells.

Output schema per the unified-worker pipeline (see
crates/zenfleet-vastai/src/worker/inline.rs::ChunkRecord):

  v26_zenavif_input.parquet:
    image_path:string  codec:string  q:int64  knob_tuple_json:string

  chunks.jsonl one record per chunk:
    {chunk_id, input_parquet, input_parquet_r2, row_range:[start,end],
     source_dir_r2, image_basenames:[...], run_id, out_sidecar_omni,
     out_encoded_prefix}

Run:

  python3 scripts/sweep/generate_v26_avif_extended.py \\
      --sources-list /tmp/v26_corpus_full.txt \\
      --source-dir-r2 s3://zentrain/sweep-v15-2026-05-06/sources \\
      --run-id v26-avif-extended-2026-05-21 \\
      --out-dir /tmp/v26-avif-extended
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Targets the R²=0.69 butter_p3 weakness — `lrf=true` at high q is
# the recovery knob (default preset turns LRF off at q ≥ 85).
#
# 5 knob tuples × 10 q × 1332 sources = 66,600 cells. Trimmed from 6:
# the archival `speed=5 + complex=true + lrf=true + fast_deblock=false`
# tuple was dropped (Option A, 2026-05-22) — too slow per-cell at our
# budget and the LRF-only and slow-deblock-only tuples isolate its
# components individually. $5.50 hard ceiling on 24 GB boxes.
KNOB_TUPLES = [
    # baseline default — speed 5, no expert overrides
    {"speed": 5, "complex_prediction_modes": False, "lrf": False, "fast_deblock": True},
    # + LRF on (attacks butter_p3 high-q weakness; the key knob)
    {"speed": 5, "complex_prediction_modes": False, "lrf": True, "fast_deblock": True},
    # + slow deblock search (edge-sensitive)
    {"speed": 5, "complex_prediction_modes": False, "lrf": False, "fast_deblock": False},
    # fast default — speed 7
    {"speed": 7, "complex_prediction_modes": False, "lrf": False, "fast_deblock": True},
    # fast + LRF (the same recovery knob at the fast tier)
    {"speed": 7, "complex_prediction_modes": False, "lrf": True, "fast_deblock": True},
]

# 10-step q grid with denser low-q coverage per CLAUDE.md "web-focused
# = q5-q40 is where structural problems hide". Same shape mc18 used.
Q_GRID = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]


def canonical_knob_json(d: dict) -> str:
    """JSON-encode with sorted keys so the (codec, knob_tuple_json)
    group key is stable across runs."""
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources-list", required=True,
                    help="Path to a newline-delimited list of source basenames")
    ap.add_argument("--source-dir-r2", required=True,
                    help="R2 prefix containing the source PNGs")
    ap.add_argument("--run-id", required=True,
                    help="Sweep run id, used in R2 prefixes")
    ap.add_argument("--out-dir", required=True,
                    help="Local directory to write input parquet + chunks.jsonl")
    ap.add_argument("--cells-per-chunk", type=int, default=300,
                    help="Approximate cells per chunk (default 300)")
    args = ap.parse_args()

    src_list = Path(args.sources_list)
    basenames = [b.strip() for b in src_list.read_text().splitlines() if b.strip()]
    print(f"# read {len(basenames)} source basenames from {src_list}",
          file=sys.stderr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build the input parquet ────────────────────────────────────
    # 1 row per (image, codec, q, knob) cell.
    knob_jsons = [canonical_knob_json(k) for k in KNOB_TUPLES]
    rows = []
    for b in basenames:
        # The worker resolves `image_path` against the R2 source dir
        # via the `image_basenames` field; the path stored here is
        # used as the join key for downstream sidecars. Use the
        # source-dir-relative bare basename so it matches the worker's
        # `chunk_input::read_and_group` basename extraction (rsplit
        # on '/').
        ip = b
        for k in knob_jsons:
            for q in Q_GRID:
                rows.append((ip, "zenavif", q, k))
    print(f"# {len(rows)} total cells across {len(basenames)} images × "
          f"{len(KNOB_TUPLES)} knob_tuples × {len(Q_GRID)} q values",
          file=sys.stderr)

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "image_path": pa.array([r[0] for r in rows], type=pa.string()),
        "codec": pa.array([r[1] for r in rows], type=pa.string()),
        "q": pa.array([r[2] for r in rows], type=pa.int64()),
        "knob_tuple_json": pa.array([r[3] for r in rows], type=pa.string()),
    })

    input_parquet_local = out_dir / "v26_zenavif_input.parquet"
    pq.write_table(table, input_parquet_local, compression="zstd",
                   compression_level=3)
    print(f"# wrote {input_parquet_local} ({input_parquet_local.stat().st_size} B)",
          file=sys.stderr)

    input_parquet_r2 = (f"s3://zentrain/{args.run_id}/input/"
                        f"v26_zenavif_input.parquet")
    print(f"# upload to: {input_parquet_r2}", file=sys.stderr)

    # ── Build chunks.jsonl ─────────────────────────────────────────
    # Sort rows by (codec, knob_tuple_json, image_path, q) so each
    # chunk groups cleanly by `(codec, knob_tuple_json)` (the worker's
    # per-group dispatcher key). Then walk in row-range slices.
    n_rows = len(rows)
    chunks_path = out_dir / "chunks.jsonl"

    # Build a (codec_knob_idx, image_idx, q_idx) packing where each
    # chunk holds ~`cells_per_chunk` contiguous rows of (image_path,
    # codec, q, knob_tuple_json). Since rows in `table` are written
    # in the order we built them — outer loop = image, then knob,
    # then q — a chunk's row range is contiguous in image space.
    #
    # With cells_per_chunk=300 and 5 knobs × 10 q = 50 cells per
    # image, each chunk spans exactly 6 images = 300 cells/chunk.
    # 1332 / 6 = 222 chunks total.
    cells_per_image = len(KNOB_TUPLES) * len(Q_GRID)
    images_per_chunk = max(1, args.cells_per_chunk // cells_per_image)
    chunk_n_cells = images_per_chunk * cells_per_image
    print(f"# {cells_per_image} cells/image; {images_per_chunk} images/chunk "
          f"→ {chunk_n_cells} cells/chunk", file=sys.stderr)

    n_chunks = 0
    with chunks_path.open("w") as f:
        for img_start in range(0, len(basenames), images_per_chunk):
            img_end = min(img_start + images_per_chunk, len(basenames))
            row_start = img_start * cells_per_image
            row_end = img_end * cells_per_image
            chunk_basenames = basenames[img_start:img_end]
            chunk_id = f"zenavif-{img_start:05d}"
            spec = {
                "chunk_id": chunk_id,
                "input_parquet": "v26_zenavif_input.parquet",
                "input_parquet_r2": input_parquet_r2,
                "row_range": [row_start, row_end],
                "source_dir_r2": args.source_dir_r2.rstrip("/"),
                "image_basenames": chunk_basenames,
                "run_id": args.run_id,
                "out_sidecar_omni": (
                    f"s3://zentrain/{args.run_id}/omni/{chunk_id}.parquet"
                ),
                "out_encoded_prefix": (
                    f"s3://zentrain/{args.run_id}/encoded/{chunk_id}/"
                ),
            }
            f.write(json.dumps(spec))
            f.write("\n")
            n_chunks += 1

    print(f"# wrote {chunks_path} with {n_chunks} chunks ({n_rows} total cells)",
          file=sys.stderr)

    # Summary for the operator
    print("", file=sys.stderr)
    print("# === v26 zenavif-extended summary ===", file=sys.stderr)
    print(f"# run id           : {args.run_id}", file=sys.stderr)
    print(f"# sources (R2)     : {args.source_dir_r2}", file=sys.stderr)
    print(f"# input parquet    : {input_parquet_r2}", file=sys.stderr)
    print(f"# chunks.jsonl     : {chunks_path}", file=sys.stderr)
    print(f"# images           : {len(basenames)}", file=sys.stderr)
    print(f"# knob_tuples      : {len(KNOB_TUPLES)}", file=sys.stderr)
    print(f"# q values         : {len(Q_GRID)}", file=sys.stderr)
    print(f"# total cells      : {n_rows}", file=sys.stderr)
    print(f"# chunks           : {n_chunks}", file=sys.stderr)
    print("", file=sys.stderr)
    print("# Next: upload input parquet + chunks to R2:", file=sys.stderr)
    print(f"#   aws s3 cp {input_parquet_local} {input_parquet_r2}", file=sys.stderr)
    chunks_r2 = f"s3://coefficient/jobs/{args.run_id}/chunks.jsonl"
    print(f"#   aws s3 cp {chunks_path} {chunks_r2}", file=sys.stderr)
    print("", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
