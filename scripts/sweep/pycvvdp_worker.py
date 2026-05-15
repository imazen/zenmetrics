#!/usr/bin/env python3
"""pycvvdp v0.5.4 scoring worker.

Consumes a TSV of (ref, dist) image pairs and writes a parquet
sidecar with the `cvvdp_pycvvdp_v054` column per
`crates/cvvdp-gpu/docs/CVVDP_SIDECAR_SCHEMA.md`.

The TSV input format avoids depending on parquet for the upstream
(producer of the encoded distorted images writes a simple TSV;
this worker reads it). Identity-tuple columns are passed through
verbatim so the sidecar joins 1:1 against the source unified
parquet.

Input TSV schema (tab-separated, header row required):

    image_path  codec  q  knob_tuple_json  ref_path  dist_path

- `image_path`, `codec`, `q`, `knob_tuple_json` are the identity
  tuple from the source parquet, copied through unchanged.
- `ref_path` is the absolute path to the reference RGB image
  (PNG/JPEG/anything Pillow can decode).
- `dist_path` is the absolute path to the distorted variant
  produced by re-encoding `image_path` at `(codec, q, knobs)`.

Both must be the same dimensions; mismatched dims are an error.

Output parquet schema:

    image_path:string  codec:string  q:int64  knob_tuple_json:string
    cvvdp_pycvvdp_v054:float64

Column-name override: pass `--score-col-name`. Defaults to
`cvvdp_pycvvdp_v054`.

Usage:

    pycvvdp-worker score-pairs \\
        --pairs-tsv pairs.tsv \\
        --out-parquet out.parquet \\
        [--display-name standard_4k] \\
        [--batch-row-group 4096] \\
        [--score-col-name cvvdp_pycvvdp_v054]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def load_rgb_uint8(path: str) -> np.ndarray:
    """Load an image as HxWx3 uint8 RGB. Raises on failure."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def score_pair(metric, ref: np.ndarray, dist: np.ndarray) -> float:
    """Run pycvvdp on a (ref, dist) RGB uint8 pair, return JOD."""
    if ref.shape != dist.shape:
        raise ValueError(
            f"shape mismatch: ref={ref.shape} dist={dist.shape}"
        )
    # pycvvdp's still-image predict expects HxWx3 uint8 in [0, 255].
    jod, _ = metric.predict(ref, dist, dim_order="HWC")
    return float(jod)


def write_parquet(rows: list[dict], out_path: str, score_col_name: str,
                  row_group_size: int) -> None:
    """Write rows to a single-file parquet with the schema spec's shape."""
    schema = pa.schema([
        ("image_path", pa.string()),
        ("codec", pa.string()),
        ("q", pa.int64()),
        ("knob_tuple_json", pa.string()),
        (score_col_name, pa.float64()),
    ])
    arrays = [
        pa.array([r["image_path"] for r in rows], type=pa.string()),
        pa.array([r["codec"] for r in rows], type=pa.string()),
        pa.array([r["q"] for r in rows], type=pa.int64()),
        pa.array([r["knob_tuple_json"] for r in rows], type=pa.string()),
        pa.array([r["score"] for r in rows], type=pa.float64()),
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_table(
        table, out_path,
        compression="zstd",
        compression_level=3,
        row_group_size=row_group_size,
    )


def cmd_score_pairs(args: argparse.Namespace) -> int:
    # Import pycvvdp lazily so `--help` doesn't pay the import cost.
    import pycvvdp

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # One metric instance — pycvvdp.cvvdp() caches the per-display
    # CSF state. Reuse across all pairs.
    metric = pycvvdp.cvvdp(
        display_name=args.display_name,
        heatmap="none",
    )

    rows: list[dict] = []
    failed = 0
    started = time.perf_counter()

    with open(args.pairs_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"image_path", "codec", "q", "knob_tuple_json",
                    "ref_path", "dist_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            print(
                f"[pycvvdp-worker] missing TSV columns: {sorted(missing)}",
                file=sys.stderr,
            )
            return 2

        for i, rec in enumerate(reader):
            try:
                ref = load_rgb_uint8(rec["ref_path"])
                dist = load_rgb_uint8(rec["dist_path"])
                jod = score_pair(metric, ref, dist)
            except Exception as e:
                failed += 1
                print(
                    f"[pycvvdp-worker] row {i} failed: {e}",
                    file=sys.stderr,
                )
                jod = float("nan")

            rows.append({
                "image_path": rec["image_path"],
                "codec": rec["codec"],
                "q": int(rec["q"]),
                "knob_tuple_json": rec["knob_tuple_json"],
                "score": jod,
            })

            if (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - started
                rate = (i + 1) / elapsed
                print(
                    f"[pycvvdp-worker] {i + 1} pairs scored, "
                    f"{rate:.2f} pairs/s, {failed} failed",
                    file=sys.stderr,
                )

    if not rows:
        print("[pycvvdp-worker] no rows produced", file=sys.stderr)
        return 3

    write_parquet(rows, str(out_path), args.score_col_name,
                  args.batch_row_group)

    total = len(rows)
    elapsed = time.perf_counter() - started
    print(
        f"[pycvvdp-worker] wrote {total} rows ({failed} failed) "
        f"to {out_path} in {elapsed:.1f}s",
        file=sys.stderr,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pycvvdp-worker",
        description="pycvvdp v0.5.4 scoring worker for parquet sidecars.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser(
        "score-pairs",
        help="Score (ref, dist) pairs from a TSV; emit parquet sidecar.",
    )
    sp.add_argument("--pairs-tsv", required=True,
                    help="Input TSV with identity tuple + ref/dist paths.")
    sp.add_argument("--out-parquet", required=True,
                    help="Output parquet sidecar path.")
    sp.add_argument("--display-name", default="standard_4k",
                    help="pycvvdp display preset (default: standard_4k).")
    sp.add_argument("--score-col-name", default="cvvdp_pycvvdp_v054",
                    help="Column name for the JOD score. Default tracks "
                         "the canonical pycvvdp v0.5.4 reference.")
    sp.add_argument("--batch-row-group", type=int, default=65536,
                    help="Row group size for the output parquet.")
    sp.set_defaults(func=cmd_score_pairs)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
