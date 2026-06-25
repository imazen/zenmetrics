# CVVDP parquet sidecar schema

Spec for the **PINNED TASK** in repo-root `CLAUDE.md`: compute CVVDP
JOD scores across every zensim training data set, write them as
parquet sidecars, and let multiple CVVDP implementations land
side-by-side via implementation-distinguished column names.

This doc fixes the schema so that:

- multiple parallel sweep workers writing different implementations
  produce mergeable sidecars,
- offline join scripts know which columns to expect,
- downstream consumers (zentrain, anchor calibration) can find
  cvvdp values by predictable column name regardless of which
  implementation produced them.

## Identity tuple

Every row keys on the same identity tuple already used by the unified
sweep parquets at
`/mnt/v/zen/zensim-training/2026-05-07/unified/`:

| column            | type   | nullability | notes |
|-------------------|--------|-------------|-------|
| `image_path`      | string | non-null    | reference image, absolute or sweep-relative |
| `codec`           | string | non-null    | `zenjpeg`, `zenwebp`, `zenavif`, … |
| `q`               | int64  | non-null    | quality dial. zenpng (lossless) uses sentinel `-1`. |
| `knob_tuple_json` | string | non-null    | JSON-serialised expert-knob bundle. Use the empty object `"{}"` for "default knobs only" — never `NULL` |

This tuple must join 1:1 with the corresponding unified parquet row.
**Do not include rows with NULL identity keys**; a missing knob string
is `"{}"`, never SQL `NULL`.

## Score columns

One column per `(implementation, version)` pair. Naming rule:

```
cvvdp_<vendor>_<version_or_tag>
```

When written to a sweep TSV/parquet, the sweep harness prefixes with
`score_`, producing `score_cvvdp_<vendor>_<version_or_tag>` — same
convention as `score_zensim` / `score_butteraugli_*` etc.

### Reserved implementation tags

| column                          | producer                                | binary                  |
|---------------------------------|-----------------------------------------|-------------------------|
| `cvvdp_pycvvdp_v054`            | upstream pycvvdp v0.5.4 (CUDA PyTorch)  | `pycvvdp` from R2 image |
| `cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>` | this crate (`cvvdp-gpu`) host scalar / GPU path | `zenmetrics` binary |
| `cvvdp_burn_v<MAJOR>_<MINOR>_<PATCH>`   | (abandoned tick 324; the Burn port was investigated and ruled out — see the "Status: ABANDONED" banner in `BURN_PORT_PLAN.md`. The tag stays reserved in case a future re-attempt wants to reuse it.) | n/a |

The `cvvdp-gpu` crate's column comes from
`cvvdp_gpu::CVVDP_COLUMN_NAME`. Default form derives from
`CARGO_PKG_VERSION`. The build-time env var `CVVDP_IMPL_TAG`
overrides the entire string — set it to bake in a git short hash
when iterating without bumping the crate version:

```
CVVDP_IMPL_TAG=cvvdp_imazen_v0_0_1_b2b7f13 \
  cargo build --release -p zenmetrics-cli --features png,gpu,gpu-cvvdp,gpu-cuda
```

### Type contract

| column suffix | parquet type | constraints |
|---------------|--------------|-------------|
| any `cvvdp_*` | `float64`    | `[0.0, 10.0]` per pycvvdp's JOD spec, 10 = imperceptible. NaN allowed only when the underlying compute failed. |

Pyramid level count, ppd, display model, etc. are **not** in the
sidecar — they're fixed per-codec by the sweep config and inferable
from `image_path` + `codec`. If a future sweep varies them per row,
add `ppd` / `display_name` columns explicitly and bump the schema
version.

## Sidecar file layout

```
/mnt/v/zen/zensim-training/<DATE>/unified/cvvdp_sidecars/
  cvvdp_<vendor>_<tag>_unified_v<X>_<codec>.parquet
```

One sidecar parquet per `(implementation, source unified parquet)`
pair. Examples:

```
cvvdp_pycvvdp_v054_unified_v15rc_zenjpeg.parquet
cvvdp_imazen_v0_0_1_unified_v15rc_zenjpeg.parquet
cvvdp_pycvvdp_v054_unified_v12_zenavif.parquet
cvvdp_imazen_v0_0_1_unified_v12_zenavif.parquet
```

Each file contains *exactly* the four identity columns plus *one*
score column. Keeping sidecars one-column-per-file makes parallel
producer workflows trivial: each worker writes its own file with no
contention, and join-time `polars`/`duckdb`/`pyarrow.dataset` reads
just discover them.

Compression: `zstd` level 3 (matches the zenmetrics sweep default).
Row group size: 65,536 (the harness default — overrides not
required).

## Manifest

A single `_MANIFEST.json` per sidecar directory mirrors the unified
parquets' manifest convention:

```json
{
  "schema_version": 1,
  "generated_at": "<ISO-8601-UTC>",
  "implementations": {
    "cvvdp_pycvvdp_v054": {
      "binary": "pycvvdp 0.5.4",
      "git_commit": "<pycvvdp repo HEAD>",
      "docker_image": "imazen/pycvvdp:0.5.4-cuda121",
      "machine": "<vast.ai instance id>"
    },
    "cvvdp_imazen_v0_0_1": {
      "binary": "zenmetrics 0.6.0",
      "git_commit": "<zenmetrics HEAD>",
      "cvvdp_gpu_version": "0.0.1",
      "docker_image": "imazen/zenmetrics:0.6.0-cvvdp-cuda121",
      "machine": "<vast.ai instance id>"
    }
  },
  "sources": [
    {
      "name": "unified_v15rc_zenjpeg.parquet",
      "rows": 513570,
      "completed_rows": 513570,
      "sidecar": "cvvdp_imazen_v0_0_1_unified_v15rc_zenjpeg.parquet"
    }
  ]
}
```

`completed_rows < rows` means a partial run — the consumer should
filter the sidecar to those rows or treat the rest as missing.

## Producer protocol (writer side)

Producers MUST:

1. Read the source unified parquet's identity tuples in row order.
2. Re-encode the distorted image deterministically from
   `(image_path, codec, q, knob_tuple_json)`. If re-encoding is not
   deterministic for some codec/knob combo, that codec is excluded
   from this task until determinism is established.
3. Score `(ref, dist)` pairs in batches, writing parquet row groups
   incrementally so worker death never leaves more than one row
   group of unrecorded work.
4. Maintain identity-tuple alignment with the source parquet — the
   row order in the sidecar matches the source parquet row order.
   This lets joiners use positional matching when an explicit join
   would be too slow.
5. Update `_MANIFEST.json` atomically (write-temp-then-rename) at
   shutdown.

Producers MUST NOT:

- emit `NULL` knob strings (use `"{}"`).
- emit cvvdp values outside `[0, 10]` (NaN allowed for failure;
  out-of-range is a producer bug).
- collide column names across implementations — the `vendor_tag`
  segment is the disambiguator.

## Consumer protocol (reader side)

```python
import pyarrow.dataset as ds

unified = ds.dataset(
    "/mnt/v/zen/zensim-training/2026-05-07/unified/unified_v15rc_zenjpeg.parquet"
)
side_pycvvdp = ds.dataset(
    "/mnt/v/zen/zensim-training/2026-05-07/unified/cvvdp_sidecars/"
    "cvvdp_pycvvdp_v054_unified_v15rc_zenjpeg.parquet"
)
side_imazen = ds.dataset(
    "/mnt/v/zen/zensim-training/2026-05-07/unified/cvvdp_sidecars/"
    "cvvdp_imazen_v0_0_1_unified_v15rc_zenjpeg.parquet"
)

# Positional join is fine because producer protocol guarantees same row order.
# For mixed-completeness sidecars, use identity-tuple join instead.
```

## Schema version & evolution

- `schema_version: 1` is the initial spec.
- Adding a new implementation column does not bump the version.
- Changing the identity tuple, score type, or compression default
  bumps to version 2 and requires a `MIGRATION.md` note in this
  crate's docs.

## Compute infra

Sidecar production runs on vast.ai docker images per the repo-root
`CLAUDE.md` "Sweep runner discipline" section. Each implementation
ships its own image (pycvvdp + PyTorch is ~3 GB; the cvvdp-gpu
image is ~50 MB), but both consume the same chunk-file format
already used by `scripts/sweep/onstart_unified.sh`.
