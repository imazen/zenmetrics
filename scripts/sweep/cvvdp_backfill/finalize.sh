#!/usr/bin/env bash
#
# finalize.sh — post-fleet finalize for the cvvdp-backfill (PINNED
# TASK). Pulls all per-chunk sidecars from R2, concatenates per
# (impl, source-parquet), writes consolidated parquets, generates a
# parity manifest, and (optionally) writes the consolidated sidecars
# back into the local unified parquet store.
#
# Companion to:
#   scripts/sweep/cvvdp_backfill/launch.sh            (host launcher)
#   scripts/sweep/onstart_cvvdp_backfill.sh           (vast.ai onstart)
#   scripts/sweep/cvvdp_backfill_chunk_worker.sh      (per-chunk)
#   scripts/sweep/generate_cvvdp_backfill_chunks.py   (chunk manifest)
#
# Sibling to scripts/sweep/finalize.sh (which finalizes Pareto TSVs
# for the v15 sweep); this one specifically handles cvvdp sidecar
# parquets, which have a different shape (identity tuple + one score
# column per implementation).
#
# Required:
#   ~/.config/cloudflare/r2-credentials (sources R2_ACCOUNT_ID + keys)
#   python3 with pyarrow
#   s5cmd
#
# Usage:
#
#   bash scripts/sweep/cvvdp_backfill/finalize.sh
#
# Environment:
#   SWEEP_RUN_ID                 (default: cvvdp-backfill-<YYYY-MM-DD>)
#   WORK                         (default: /tmp/cvvdp-backfill-finalize)
#   LOCAL_UNIFIED_DIR            optional — if set, copies the
#                                consolidated sidecars there
#                                (e.g. /mnt/v/zen/zensim-training/<date>/unified/)
#   UPLOAD_CONSOLIDATED          1 to push the consolidated parquets
#                                back to R2 (default 1)
#
# Outputs:
#   $WORK/cvvdp_imazen_<input_stem>.parquet      per source parquet
#   $WORK/cvvdp_pycvvdp_v054_<input_stem>.parquet per source parquet
#   $WORK/parity_<input_stem>.tsv                joined side-by-side
#                                                with diff column
#   $WORK/manifest.json                          summary: chunk count,
#                                                row count, parity stats
#                                                per source parquet

set -euo pipefail

SWEEP_RUN_ID="${SWEEP_RUN_ID:-cvvdp-backfill-$(date -u +%Y-%m-%d)}"
WORK="${WORK:-/tmp/cvvdp-backfill-finalize}"
LOCAL_UNIFIED_DIR="${LOCAL_UNIFIED_DIR:-}"
UPLOAD_CONSOLIDATED="${UPLOAD_CONSOLIDATED:-1}"

mkdir -p "$WORK"

set -a
# shellcheck disable=SC1091
source "$HOME/.config/cloudflare/r2-credentials"
set +a
R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

R2() {
    s5cmd \
        --endpoint-url "$R2_ENDPOINT" \
        --profile r2 \
        "$@"
}

mkdir -p ~/.aws
cat > ~/.aws/credentials <<EOF
[r2]
aws_access_key_id = ${R2_ACCESS_KEY_ID}
aws_secret_access_key = ${R2_SECRET_ACCESS_KEY}
EOF

echo "[finalize] SWEEP_RUN_ID=$SWEEP_RUN_ID  WORK=$WORK"

# Per-impl R2 prefixes (matches the chunk-worker upload targets in
# generate_cvvdp_backfill_chunks.py).
IMAZEN_PREFIX="s3://zentrain/${SWEEP_RUN_ID}/cvvdp_imazen"
PYCVVDP_PREFIX="s3://zentrain/${SWEEP_RUN_ID}/cvvdp_pycvvdp_v054"

echo "[finalize] step 1/4: sync per-chunk sidecars"
mkdir -p "$WORK/imazen" "$WORK/pycvvdp"
R2 sync "${IMAZEN_PREFIX}/*" "$WORK/imazen/" >&2 || true
R2 sync "${PYCVVDP_PREFIX}/*" "$WORK/pycvvdp/" >&2 || true

N_IMAZEN=$(ls "$WORK/imazen"/*.parquet 2>/dev/null | wc -l || echo 0)
N_PYCVVDP=$(ls "$WORK/pycvvdp"/*.parquet 2>/dev/null | wc -l || echo 0)
echo "  imazen sidecars: $N_IMAZEN"
echo "  pycvvdp sidecars: $N_PYCVVDP"
if [[ "$N_IMAZEN" == 0 && "$N_PYCVVDP" == 0 ]]; then
    echo "ERROR: no chunk sidecars found under $IMAZEN_PREFIX or $PYCVVDP_PREFIX" >&2
    echo "Has the fleet finished? Check heartbeats:" >&2
    echo "  s5cmd ls s3://coefficient/heartbeats/${SWEEP_RUN_ID}/" >&2
    exit 1
fi

echo "[finalize] step 2/4: concatenate per (impl, source-parquet)"
WORK="$WORK" SWEEP_RUN_ID="$SWEEP_RUN_ID" python3 - <<'PYEOF' >&2
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

work = Path(os.environ["WORK"])
run = os.environ["SWEEP_RUN_ID"]

# Chunk_id format from the generator: <input_stem>-NNNN where
# input_stem is e.g. "v12_zenwebp". Group sidecar files by stem.
def stem_of(chunk_filename: str) -> str:
    # foo-NNNN.parquet -> foo
    stem = Path(chunk_filename).stem
    if "-" in stem:
        return stem.rsplit("-", 1)[0]
    return stem

manifest = {"run_id": run, "sources": {}}

for impl in ("imazen", "pycvvdp"):
    impl_dir = work / impl
    if not impl_dir.is_dir():
        continue
    by_stem = defaultdict(list)
    for f in sorted(impl_dir.glob("*.parquet")):
        by_stem[stem_of(f.name)].append(f)

    for stem, files in by_stem.items():
        if not files:
            continue
        tables = []
        for f in files:
            try:
                tables.append(pq.read_table(f))
            except Exception as e:
                print(f"  skip {impl}/{f.name}: {e}", file=sys.stderr)
        if not tables:
            continue
        merged = pa.concat_tables(tables, promote_options="default")
        impl_label = "imazen" if impl == "imazen" else "pycvvdp_v054"
        out_name = f"cvvdp_{impl_label}_{stem}.parquet"
        out_path = work / out_name
        pq.write_table(merged, out_path, compression="zstd")
        info = manifest["sources"].setdefault(stem, {"impls": {}})
        info["impls"][impl_label] = {
            "chunks": len(files),
            "rows": merged.num_rows,
            "consolidated_parquet": out_name,
        }
        print(f"  {impl}/{stem}: {len(files)} chunks -> {merged.num_rows} rows -> {out_name}", file=sys.stderr)

# Parity joiner per source: load both impls, join on identity tuple,
# emit diff stats + the parity TSV.
for stem, info in manifest["sources"].items():
    impls = info["impls"]
    if "imazen" not in impls or "pycvvdp_v054" not in impls:
        info["parity"] = None
        continue
    imazen_path = work / impls["imazen"]["consolidated_parquet"]
    pycvvdp_path = work / impls["pycvvdp_v054"]["consolidated_parquet"]
    imazen = pq.read_table(imazen_path).to_pylist()
    pycvvdp = pq.read_table(pycvvdp_path).to_pylist()

    # Identity tuple key. The score column on each side carries the
    # impl tag; we scan for it (cvvdp_imazen_v* or cvvdp_pycvvdp_v054).
    imazen_col = next((k for k in imazen[0].keys() if k.startswith("cvvdp_imazen")), None)
    pycvvdp_col = "cvvdp_pycvvdp_v054"
    if imazen_col is None or pycvvdp_col not in pycvvdp[0]:
        print(f"  parity skip {stem}: missing score col(s)", file=sys.stderr)
        info["parity"] = None
        continue

    key = lambda r: (r["image_path"], r["codec"], r["q"], r["knob_tuple_json"])
    imap = {key(r): r[imazen_col] for r in imazen}
    pmap = {key(r): r[pycvvdp_col] for r in pycvvdp}
    joined = []
    for k, im in imap.items():
        pv = pmap.get(k)
        if pv is None or im is None:
            continue
        diff = im - pv
        joined.append((k, im, pv, diff))
    if not joined:
        info["parity"] = {"joined": 0}
        continue

    parity_tsv = work / f"parity_{stem}.tsv"
    with open(parity_tsv, "w") as f:
        f.write("image_path\tcodec\tq\tknob_tuple_json\t")
        f.write(f"{imazen_col}\t{pycvvdp_col}\tdiff\n")
        for (ip, cd, q, kj), im, pv, df in joined:
            f.write(f"{ip}\t{cd}\t{q}\t{kj}\t{im:.6f}\t{pv:.6f}\t{df:+.6f}\n")

    diffs = sorted(abs(d) for _, _, _, d in joined)
    n = len(diffs)
    info["parity"] = {
        "joined": n,
        "mean_abs_diff": sum(diffs) / n,
        "median_abs_diff": diffs[n // 2],
        "max_abs_diff": diffs[-1],
        "parity_tsv": parity_tsv.name,
        "imazen_col": imazen_col,
    }
    print(
        f"  parity {stem}: n={n}  mean={diffs[0:1] and sum(diffs)/n:.4f}  "
        f"median={diffs[n//2]:.4f}  max={diffs[-1]:.4f}",
        file=sys.stderr,
    )

with open(work / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
PYEOF

echo "[finalize] step 3/4: copy consolidated sidecars to local unified dir"
if [[ -n "$LOCAL_UNIFIED_DIR" ]]; then
    mkdir -p "$LOCAL_UNIFIED_DIR/cvvdp_sidecars"
    cp -v "$WORK"/cvvdp_*.parquet "$LOCAL_UNIFIED_DIR/cvvdp_sidecars/" >&2 || true
    cp -v "$WORK"/parity_*.tsv "$LOCAL_UNIFIED_DIR/cvvdp_sidecars/" >&2 || true
    cp -v "$WORK/manifest.json" "$LOCAL_UNIFIED_DIR/cvvdp_sidecars/" >&2 || true
    echo "  copied to $LOCAL_UNIFIED_DIR/cvvdp_sidecars/"
else
    echo "  LOCAL_UNIFIED_DIR not set; skipping local copy"
fi

echo "[finalize] step 4/4: upload consolidated parquets back to R2"
if [[ "$UPLOAD_CONSOLIDATED" == "1" ]]; then
    R2 sync "$WORK/" "s3://zentrain/${SWEEP_RUN_ID}/consolidated/" \
        --exclude '*' --include 'cvvdp_*.parquet' --include 'parity_*.tsv' \
        --include 'manifest.json' >&2 || true
    echo "  consolidated/* -> s3://zentrain/${SWEEP_RUN_ID}/consolidated/"
else
    echo "  UPLOAD_CONSOLIDATED=0; skipping R2 upload"
fi

echo
echo "[finalize] manifest summary:"
python3 -c "
import json
m = json.load(open('$WORK/manifest.json'))
print(f'  run_id: {m[\"run_id\"]}')
for stem, info in m['sources'].items():
    line = f'  {stem}:'
    for impl, d in info['impls'].items():
        line += f' [{impl} {d[\"rows\"]}r/{d[\"chunks\"]}c]'
    p = info.get('parity')
    if p and p.get('joined'):
        line += f' parity n={p[\"joined\"]} mean={p[\"mean_abs_diff\"]:.4f} max={p[\"max_abs_diff\"]:.4f}'
    print(line)
"
