#!/usr/bin/env python3
"""Codec-agnostic sweep input-parquet + chunks.jsonl generator.

Generalizes the per-codec copies (`generate_v26_avif_extended.py`,
`generate_jobspecs_v06.py`, ...) into ONE tool so we stop forking a new
script per sweep. It enumerates a (codec x q x knob-tuple) cell list into
the v26 input-parquet format the inline-sweep worker consumes
(`zen-cloud-vastai/src/worker/inline.rs::ChunkRecord` ->
`sweep_runner::run_group_inline` -> `run_sweep`; same path the Hetzner
`Dockerfile.sweep.hetzner.v1` links), plus the matching `chunks.jsonl`.

Knob axes use the SAME `{axis:[values]}` Cartesian-product JSON that
`zen-metrics sweep --knob-grid` (grid.rs::KnobGrid) takes. Pass
`--knob-grid` more than once to UNION several grids — e.g. a YCbCr
all-flags grid and a (separately pruned) XYB grid whose axis sets differ
and so cannot live in a single Cartesian product.

Output schema (one row per cell):
    image_path:string  codec:string  q:int64  knob_tuple_json:string
chunks.jsonl (one record per chunk):
    {chunk_id, input_parquet, input_parquet_r2, row_range:[start,end],
     source_dir_r2, image_basenames:[...], run_id, out_sidecar_omni,
     out_encoded_prefix}

Example (the zenjpeg XYB recovery sweep — YCbCr all-flags U pruned-16 XYB):

  python3 scripts/sweep/generate_sweep_input.py \\
    --codec zenjpeg --run-id zenjpeg-xyb-2026-06-02 \\
    --sources-list /tmp/corpus.txt \\
    --source-dir-r2 s3://zentrain/zenjpeg-xyb-2026-06-02/sources \\
    --out-dir /tmp/zenjpeg-xyb \\
    --q-grid 5,15,25,35,45,55,65,75,85,95 \\
    --knob-grid '{"subsampling":["420","444"],"progressive":[true],"chroma_quality":[0,1],"hybrid":[false,true]}' \\
    --knob-grid '{"xyb":[true],"xyb_subsampling":["bquarter","full"],"progressive":[true],"hybrid":[false,true],"aq_enabled":[false,true],"deringing":[false,true]}'
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path


def canonical_knob_json(d: dict) -> str:
    """Sorted-key compact JSON so the (codec, knob_tuple_json) group key is
    byte-stable across runs and matches the worker's grouping key."""
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def expand_grid(grid: dict) -> list[dict]:
    """Cartesian product of {axis: [values]} -> list of knob dicts.
    Mirrors grid.rs::KnobGrid::iter_tuples (an empty grid -> one empty tuple)."""
    if not grid:
        return [{}]
    axes = list(grid.items())
    for name, vals in axes:
        if not isinstance(vals, list) or not vals:
            raise ValueError(f"knob {name!r} must map to a non-empty list")
    out = []
    for combo in itertools.product(*[v for _, v in axes]):
        out.append({axes[i][0]: combo[i] for i in range(len(axes))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", required=True,
                    help="Codec name as the worker expects (zenjpeg/zenwebp/zenavif/zenjxl/zenpng)")
    ap.add_argument("--run-id", required=True, help="Sweep run id (used in R2 prefixes)")
    ap.add_argument("--sources-list", required=True,
                    help="Newline-delimited list of source basenames (relative to --source-dir-r2)")
    ap.add_argument("--source-dir-r2", required=True, help="R2 prefix holding the source images")
    ap.add_argument("--out-dir", required=True, help="Local dir for input parquet + chunks.jsonl")
    ap.add_argument("--q-grid", required=True, help="Comma-separated q values, e.g. 5,15,...,95")
    ap.add_argument("--knob-grid", action="append", default=[], dest="knob_grids",
                    help="JSON {axis:[values]} Cartesian grid; repeat to UNION several grids")
    ap.add_argument("--cells-per-chunk", type=int, default=300, help="Approx cells per chunk")
    ap.add_argument("--r2-jobs-prefix", default="s3://zentrain",
                    help="R2 prefix for the input parquet (default s3://zentrain)")
    args = ap.parse_args()

    basenames = [b.strip() for b in Path(args.sources_list).read_text().splitlines() if b.strip()]
    q_grid = [int(x) for x in args.q_grid.split(",") if x.strip()]

    # Expand + UNION every --knob-grid, de-duping identical knob tuples.
    knob_jsons: list[str] = []
    seen: set[str] = set()
    grids = args.knob_grids or ["{}"]
    for spec in grids:
        for kt in expand_grid(json.loads(spec)):
            j = canonical_knob_json(kt)
            if j not in seen:
                seen.add(j)
                knob_jsons.append(j)

    cells_per_image = len(knob_jsons) * len(q_grid)
    print(f"# codec={args.codec} | {len(basenames)} images x {len(knob_jsons)} knob_tuples "
          f"x {len(q_grid)} q = {len(basenames) * cells_per_image} cells "
          f"({cells_per_image} cells/image)", file=sys.stderr)
    print(f"# knob_tuples (after union+dedup): {len(knob_jsons)}", file=sys.stderr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── input parquet: outer loop image, then knob, then q (so a chunk's
    #    row-range is contiguous in image space, matching the v26 worker). ──
    rows = []
    for b in basenames:
        for k in knob_jsons:
            for q in q_grid:
                rows.append((b, args.codec, q, k))

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "image_path": pa.array([r[0] for r in rows], type=pa.string()),
        "codec": pa.array([r[1] for r in rows], type=pa.string()),
        "q": pa.array([r[2] for r in rows], type=pa.int64()),
        "knob_tuple_json": pa.array([r[3] for r in rows], type=pa.string()),
    })
    input_name = f"{args.codec}_{args.run_id}_input.parquet"
    input_local = out_dir / input_name
    pq.write_table(table, input_local, compression="zstd", compression_level=3)
    input_r2 = f"{args.r2_jobs_prefix.rstrip('/')}/{args.run_id}/input/{input_name}"
    print(f"# wrote {input_local} ({input_local.stat().st_size} B) -> {input_r2}", file=sys.stderr)

    # ── chunks.jsonl: contiguous image-range slices ──
    images_per_chunk = max(1, args.cells_per_chunk // cells_per_image)
    chunks_path = out_dir / "chunks.jsonl"
    n_chunks = 0
    with chunks_path.open("w") as f:
        for img_start in range(0, len(basenames), images_per_chunk):
            img_end = min(img_start + images_per_chunk, len(basenames))
            chunk_id = f"{args.codec}-{img_start:05d}"
            spec = {
                "chunk_id": chunk_id,
                "input_parquet": input_name,
                "input_parquet_r2": input_r2,
                "row_range": [img_start * cells_per_image, img_end * cells_per_image],
                "source_dir_r2": args.source_dir_r2.rstrip("/"),
                "image_basenames": basenames[img_start:img_end],
                "run_id": args.run_id,
                "out_sidecar_omni": f"{args.r2_jobs_prefix.rstrip('/')}/{args.run_id}/omni/{chunk_id}.parquet",
                "out_encoded_prefix": f"{args.r2_jobs_prefix.rstrip('/')}/{args.run_id}/encoded/{chunk_id}/",
            }
            f.write(json.dumps(spec) + "\n")
            n_chunks += 1

    print(f"# wrote {chunks_path}: {n_chunks} chunks, {len(rows)} cells, "
          f"{images_per_chunk} images/chunk", file=sys.stderr)
    print(f"# upload: aws s3 cp {input_local} {input_r2}", file=sys.stderr)
    print(f"#         aws s3 cp {chunks_path} s3://coefficient/jobs/{args.run_id}/chunks.jsonl",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
