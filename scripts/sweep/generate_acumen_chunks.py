#!/usr/bin/env python3
"""Generate chunks.jsonl for the Gate A castleCSF Mode A
zensim-gpu feature-extraction sweep.

Reads canonical training parquets (`safesyn`, `kadid`, `tid`,
`konjnd-dense`, `cvvdp_iwssim_LARGE`) and slices each into
fixed-row chunks. Each chunk references rows by `row_range`, lists
the unique `image_basenames` that the worker needs to sync from
R2, and points at an `out_sidecar_zensim_acumen` parquet output
path.

The worker (onstart_acumen_modea.sh + metric_backfill_chunk_worker.sh
with ACUMEN_MODE_A=1) consumes one chunk per claim, re-encodes the
dist images via `zen-metrics sweep`, then runs `zen-metrics
score-pairs --metric zensim-gpu --acumen-mode-a` to produce the
sidecar.

Tracking: imazen/zensim#40 Gate A.

USAGE:

    python3 scripts/sweep/generate_acumen_chunks.py \\
        --canonical-dir /mnt/v/zen/zensim-training/canonical-2026-05-21 \\
        --run-id acumen-modea-gate-a-2026-05-21 \\
        --source-r2-prefix s3://zentrain/sweep-v15-2026-05-06/sources \\
        --input-r2-prefix s3://zentrain/canonical-training-2026-05-21 \\
        --output-r2-prefix s3://zentrain/acumen-modea-gate-a-2026-05-21 \\
        --chunk-size 100 \\
        --out chunks.jsonl

Then upload via s5cmd:

    s5cmd cp chunks.jsonl s3://coefficient/jobs/<run-id>/chunks.jsonl

NOTES:

- Reuses identity tuples (image_path, codec, q, knob_tuple_json)
  from each canonical parquet — same shape as
  generate_cvvdp_backfill_chunks.py. Caller can filter by
  `--filter-parquet safesyn` to limit to one parquet for smoke
  tests.
- Default `--chunk-size 100` produces ~2400 chunks across the
  full 240k canonical training rows at 5-15 min per chunk.
- Deterministic: same inputs → same chunks.jsonl byte-for-byte
  (sorted by parquet filename, then row index).
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    print(
        "pyarrow required; install via:\n"
        "  pip install --user pyarrow>=15",
        file=sys.stderr,
    )
    sys.exit(2)


def chunks_from_parquet(
    parquet_path: Path,
    parquet_name: str,
    chunk_size: int,
    source_r2_prefix: str,
    input_r2_prefix: str,
    output_r2_prefix: str,
    run_id: str,
):
    """Yield one chunk dict per chunk_size-row slice of the parquet."""
    table = pq.read_table(parquet_path, columns=["image_path"])
    # image_path here is the *ref* image; the worker derives dist
    # images by re-encoding per the identity tuple. For Gate A we
    # only need basenames to sync.
    image_paths = table.column("image_path").to_pylist()
    n_rows = len(image_paths)
    n_chunks = (n_rows + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, n_rows)
        chunk_rows = image_paths[start:end]
        # Unique basenames within the chunk — the worker syncs only
        # these from R2 rather than the whole source dir.
        basenames = sorted({Path(p).name for p in chunk_rows})
        chunk_id = f"{parquet_name}-{ci:04d}"
        yield {
            "chunk_id": chunk_id,
            "input_parquet": parquet_path.name,
            "input_parquet_r2": f"{input_r2_prefix}/{parquet_path.name}",
            "row_range": [start, end],
            "source_dir_r2": source_r2_prefix,
            "out_sidecar_zensim_acumen": (
                f"{output_r2_prefix}/zensim_acumen/{chunk_id}.parquet"
            ),
            "image_basenames": basenames,
            "row_count": end - start,
            "run_id": run_id,
        }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--canonical-dir",
        type=Path,
        required=True,
        help="Local canonical training dir (train/*.parquet files).",
    )
    p.add_argument("--run-id", required=True)
    p.add_argument("--source-r2-prefix", required=True)
    p.add_argument("--input-r2-prefix", required=True)
    p.add_argument("--output-r2-prefix", required=True)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument(
        "--filter-parquet",
        default=None,
        help="Restrict to a single parquet name (e.g. 'safesyn').",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Emit at most this many chunks (smoke testing).",
    )
    p.add_argument("--out", type=Path, default=Path("chunks.jsonl"))
    args = p.parse_args()

    train_dir = args.canonical_dir / "train"
    if not train_dir.is_dir():
        print(f"ERROR: train/ not found at {train_dir}", file=sys.stderr)
        return 2

    parquets = sorted(train_dir.glob("*.parquet"))
    if args.filter_parquet:
        parquets = [
            p for p in parquets if args.filter_parquet in p.stem
        ]
    if not parquets:
        print(f"ERROR: no matching parquets in {train_dir}", file=sys.stderr)
        return 2

    with args.out.open("w") as f:
        total = 0
        for parquet_path in parquets:
            parquet_name = parquet_path.stem
            for chunk in chunks_from_parquet(
                parquet_path,
                parquet_name,
                args.chunk_size,
                args.source_r2_prefix,
                args.input_r2_prefix,
                args.output_r2_prefix,
                args.run_id,
            ):
                f.write(json.dumps(chunk) + "\n")
                total += 1
                if args.max_chunks is not None and total >= args.max_chunks:
                    break
            if args.max_chunks is not None and total >= args.max_chunks:
                break
    print(
        f"wrote {total} chunks to {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
