#!/usr/bin/env python3
"""Generate chunks.jsonl for the CVVDP-backfill sweep.

PINNED TASK (CLAUDE.md): backfill cvvdp JOD scores into the existing
unified-V_X parquet store at
``/mnt/v/zen/zensim-training/2026-05-07/unified/`` (or wherever the
caller points the script). Splits each parquet into fixed-row chunks
so a vast.ai fleet can fan out — each worker claims one chunk,
re-encodes the distortions, scores both cvvdp_imazen and
cvvdp_pycvvdp_v054, uploads two parquet sidecars back to R2.

Unlike ``generate_jobspecs_v06.py`` (which drives a fresh sweep from
sources + codec × q × knob_tuple grid), this generator works against
*already-swept* identity tuples — no codec or knob_grid in each
chunk; the worker reads the tuple directly from the parquet slice.

OUTPUT
======

One JSON object per line. Schema (one chunk = ``CHUNK_SIZE`` rows of
the source parquet, indexed by ``row_range``):

.. code-block:: json

    {
        "chunk_id": "v12-zenwebp-0000",
        "input_parquet": "unified_v12_zenwebp.parquet",
        "input_parquet_r2": "s3://zentrain/unified-v12/unified_v12_zenwebp.parquet",
        "row_range": [0, 100],
        "source_dir_r2": "s3://zentrain/sweep-v15-2026-05-06/sources",
        "out_sidecar_imazen": "s3://zentrain/cvvdp-backfill-2026-05-15/cvvdp_imazen/v12-zenwebp-0000.parquet",
        "out_sidecar_pycvvdp": "s3://zentrain/cvvdp-backfill-2026-05-15/cvvdp_pycvvdp_v054/v12-zenwebp-0000.parquet",
        "image_basenames": ["foo.png", "bar.png", ...],
        "row_count": 100,
        "run_id": "cvvdp-backfill-2026-05-15"
    }

``image_basenames`` is the deduplicated set of basenames referenced
by the chunk's rows — the worker uses it to sync only the needed
sources, not the whole sweep-run's source dir.

USAGE
=====

.. code-block:: bash

    python3 scripts/sweep/generate_cvvdp_backfill_chunks.py \\
        --unified-dir /mnt/v/zen/zensim-training/2026-05-07/unified \\
        --run-id cvvdp-backfill-2026-05-15 \\
        --source-r2-prefix s3://zentrain/sweep-v15-2026-05-06/sources \\
        --input-r2-prefix s3://zentrain/unified-2026-05-07 \\
        --output-r2-prefix s3://zentrain/cvvdp-backfill-2026-05-15 \\
        --chunk-size 100 \\
        --out chunks.jsonl

The script does NOT upload anything; it just emits the manifest.
Upload via ``s5cmd`` separately:

.. code-block:: bash

    s5cmd cp chunks.jsonl s3://coefficient/jobs/cvvdp-backfill-2026-05-15/

For local smoke runs, pass ``--filter-codec zenwebp`` to restrict to
one parquet, and ``--max-chunks 1`` to emit just one chunk.

NOTES
=====

- Chunks are deterministic given ``--chunk-size`` + sort order
  (alphabetical on parquet filename, then row index). Re-running
  the generator on the same inputs reproduces the same chunks.jsonl
  byte-for-byte, modulo filesystem ordering of glob results — sorted
  here to be explicit.

- The 7-parquet unified store at 2026-05-07 has 2,374,666 rows total.
  At ``--chunk-size 100`` that's 23,747 chunks; at 1000 it's 2,374.
  Default 100 keeps per-chunk wall-time bounded (~5-10 min) for vast.ai
  rebalancing and graceful failure recovery.

- The ``image_basenames`` field deduplicates within a chunk. zensim
  sweep rows are clustered per-image (each row = one quality), so a
  100-row chunk typically references 1-20 unique source images. The
  worker syncs those basenames only, not the whole sweep-run's source
  dir (which is ~60 GB for v15).

- Don't confuse this with ``generate_jobspecs_v06.py``: v06 emits
  full sweep specs (codec × image × knob grid). THIS script emits
  score-only chunks that take an existing parquet slice as input.
  No codec / knob_tuple expansion happens here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _basename(image_path: str) -> str:
    """Strip the ``/workspace/sweep/stage-*/`` prefix and return the
    basename. Defensive against absolute paths, relative paths, and
    paths with no stage prefix (e.g. the unified-v12 sweep records
    paths like ``/workspace/sweep/stage-zenwebp-000/foo.png``)."""
    return os.path.basename(image_path)


def emit_chunks(
    parquet_path: Path,
    chunk_size: int,
    run_id: str,
    source_r2_prefix: str,
    input_r2_prefix: str,
    output_r2_prefix: str,
    parquet_codec: str,
) -> list[dict]:
    """Read one parquet's identity-tuple columns and emit chunks."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "ERROR: pyarrow not installed. `pip install pyarrow` in the venv.",
            file=sys.stderr,
        )
        sys.exit(2)

    name = parquet_path.name
    # Derive the per-parquet chunk-id prefix from the filename. Strip
    # the leading ``unified_`` and the trailing ``.parquet``.
    stem = name.removeprefix("unified_").removesuffix(".parquet")
    chunk_prefix = stem  # e.g. v12_zenwebp -> v12_zenwebp

    table = pq.read_table(
        parquet_path,
        columns=["image_path", "codec", "q", "knob_tuple_json"],
    )
    n_rows = table.num_rows
    if n_rows == 0:
        return []

    rows = table.to_pylist()
    # Sanity: all rows in one parquet should share a codec. If not,
    # surface it — the chunk's codec field is per-chunk in our schema.
    codecs = {r["codec"] for r in rows}
    if len(codecs) != 1:
        print(
            f"WARN: {name} has {len(codecs)} codecs ({sorted(codecs)}); "
            "using first row's codec as chunk-level codec",
            file=sys.stderr,
        )

    chunks: list[dict] = []
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        slice_rows = rows[start:end]
        basenames = sorted({_basename(r["image_path"]) for r in slice_rows})
        chunk_id = f"{chunk_prefix}-{start // chunk_size:04d}"

        chunks.append(
            {
                "chunk_id": chunk_id,
                "input_parquet": name,
                "input_parquet_r2": f"{input_r2_prefix.rstrip('/')}/{name}",
                "row_range": [start, end],
                "source_dir_r2": source_r2_prefix.rstrip("/"),
                "out_sidecar_imazen": (
                    f"{output_r2_prefix.rstrip('/')}/cvvdp_imazen/"
                    f"{chunk_id}.parquet"
                ),
                "out_sidecar_pycvvdp": (
                    f"{output_r2_prefix.rstrip('/')}/cvvdp_pycvvdp_v054/"
                    f"{chunk_id}.parquet"
                ),
                "image_basenames": basenames,
                "row_count": end - start,
                "run_id": run_id,
            }
        )

    return chunks


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--unified-dir",
        type=Path,
        required=True,
        help="Local path to the unified-V_X parquet store",
    )
    p.add_argument(
        "--run-id",
        required=True,
        help='Sweep run id (e.g. "cvvdp-backfill-2026-05-15")',
    )
    p.add_argument(
        "--source-r2-prefix",
        required=True,
        help="R2 prefix for the source images (workers sync from here)",
    )
    p.add_argument(
        "--input-r2-prefix",
        required=True,
        help="R2 prefix where the unified parquets live (workers download)",
    )
    p.add_argument(
        "--output-r2-prefix",
        required=True,
        help="R2 prefix where workers upload sidecars",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Rows per chunk (default 100; 1000+ for large fleet tests)",
    )
    p.add_argument(
        "--filter-codec",
        default=None,
        help='Optional substring filter on parquet filename (e.g. "zenwebp")',
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Stop after emitting this many chunks (smoke runs)",
    )
    p.add_argument("--out", type=Path, default=Path("-"), help='Output file ("-" = stdout)')

    args = p.parse_args()

    if not args.unified_dir.is_dir():
        print(f"ERROR: --unified-dir {args.unified_dir} does not exist", file=sys.stderr)
        return 1

    parquets = sorted(args.unified_dir.glob("unified_*.parquet"))
    if args.filter_codec:
        parquets = [p for p in parquets if args.filter_codec in p.name]
    if not parquets:
        print(
            f"ERROR: no parquets found in {args.unified_dir} "
            f"(filter={args.filter_codec!r})",
            file=sys.stderr,
        )
        return 1

    all_chunks: list[dict] = []
    for pq_path in parquets:
        codec_guess = pq_path.stem.split("_")[-1]
        these = emit_chunks(
            pq_path,
            args.chunk_size,
            args.run_id,
            args.source_r2_prefix,
            args.input_r2_prefix,
            args.output_r2_prefix,
            parquet_codec=codec_guess,
        )
        all_chunks.extend(these)
        print(f"  {pq_path.name}: {len(these)} chunks", file=sys.stderr)
        if args.max_chunks and len(all_chunks) >= args.max_chunks:
            all_chunks = all_chunks[: args.max_chunks]
            break

    out = sys.stdout if str(args.out) == "-" else args.out.open("w")
    try:
        for chunk in all_chunks:
            print(json.dumps(chunk, separators=(",", ":")), file=out)
    finally:
        if out is not sys.stdout:
            out.close()

    print(
        f"emitted {len(all_chunks)} chunks "
        f"(from {len(parquets)} parquets, chunk-size {args.chunk_size})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
