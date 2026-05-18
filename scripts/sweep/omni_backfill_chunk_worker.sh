#!/usr/bin/env bash
#
# omni_backfill_chunk_worker.sh — one-time multi-metric + encoded-
# variants backfill chunk worker.
#
# Differs from metric_backfill_chunk_worker.sh:
#   - Runs ALL metrics in one `zen-metrics sweep` pass (no separate
#     score-pairs step). The sweep TSV directly carries the score
#     columns.
#   - Saves encoded codec bytes (.jpg / .webp / .avif / .jxl / .png)
#     via the new `--encoded-out-dir` flag and uploads them to R2 at
#     `s3://<bucket>/<run>/encoded/<chunk_id>/<filename>`. Future
#     backfills can skip the encode step entirely by downloading the
#     same R2 path.
#
# Chunk-manifest fields read (same shape as cvvdp backfill chunks):
#   chunk_id, input_parquet, input_parquet_r2, row_range,
#   source_dir_r2, image_basenames, run_id
# Plus optional `out_sidecar_omni` (defaults to
#   s3://zentrain/<run_id>/omni/<chunk_id>.parquet) and
#   `out_encoded_prefix` (defaults to
#   s3://zentrain/<run_id>/encoded/<chunk_id>/).
#
# Required env vars:
#   R2_ACCOUNT_ID  R2_ACCESS_KEY_ID  R2_SECRET_ACCESS_KEY
#
# CLI flags (most also accept env-var alternatives in shouty case):
#   --metrics m1,m2,...    comma-list (default: zensim,ssim2-gpu,
#                          butteraugli-gpu,cvvdp,dssim-gpu,iwssim-gpu)
#   --gpu-runtime cuda     (default; CPU fallback blocked at the trap
#                          level — see run_with_error_trap.sh)
#   --parallel N           cells/group concurrency (default 0 = rayon
#                          auto)
#   --work-dir DIR         scratch root
#   --skip-upload          do not push to R2 (dev only)
#
# Usage:
#   echo '<chunk_json>' | omni_backfill_chunk_worker.sh

set -euo pipefail

METRICS="${METRICS:-zensim,ssim2-gpu,butteraugli-gpu,cvvdp,dssim-gpu,iwssim-gpu}"
GPU_RUNTIME="${GPU_RUNTIME:-cuda}"
PARALLEL="${PARALLEL:-0}"
WORK_DIR="${WORK_DIR:-}"
CHUNK_JSON="${CHUNK_JSON:-}"
KEEP_WORK="${KEEP_WORK:-0}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --metrics) METRICS="$2"; shift 2;;
        --gpu-runtime) GPU_RUNTIME="$2"; shift 2;;
        --parallel) PARALLEL="$2"; shift 2;;
        --work-dir) WORK_DIR="$2"; shift 2;;
        --chunk-json) CHUNK_JSON="$2"; shift 2;;
        --skip-upload) SKIP_UPLOAD=1; shift;;
        --keep-work) KEEP_WORK=1; shift;;
        -h|--help) sed -n '2,32p' "$0" >&2; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done

if [[ -z "$CHUNK_JSON" ]]; then
    CHUNK_JSON=$(cat)
fi
if [[ -z "$CHUNK_JSON" ]]; then
    echo "ERROR: --chunk-json or stdin required" >&2; exit 2
fi

: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID missing}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY missing}"

CHUNK_ID=$(jq -r '.chunk_id' <<< "$CHUNK_JSON")
INPUT_PARQUET=$(jq -r '.input_parquet' <<< "$CHUNK_JSON")
INPUT_PARQUET_R2=$(jq -r '.input_parquet_r2' <<< "$CHUNK_JSON")
ROW_START=$(jq -r '.row_range[0]' <<< "$CHUNK_JSON")
ROW_END=$(jq -r '.row_range[1]' <<< "$CHUNK_JSON")
SOURCE_DIR_R2=$(jq -r '.source_dir_r2' <<< "$CHUNK_JSON")
RUN_ID=$(jq -r '.run_id // empty' <<< "$CHUNK_JSON")
[[ -z "$RUN_ID" || "$RUN_ID" == "null" ]] && {
    echo "ERROR: chunk JSON missing .run_id" >&2; exit 2
}

OUT_SIDECAR_R2=$(jq -r --arg r "$RUN_ID" --arg c "$CHUNK_ID" \
    '.out_sidecar_omni // ("s3://zentrain/" + $r + "/omni/" + $c + ".parquet")' <<< "$CHUNK_JSON")
OUT_ENCODED_PREFIX=$(jq -r --arg r "$RUN_ID" --arg c "$CHUNK_ID" \
    '.out_encoded_prefix // ("s3://zentrain/" + $r + "/encoded/" + $c + "/")' <<< "$CHUNK_JSON")

WORK_DIR="${WORK_DIR:-/workspace/omni-backfill/$CHUNK_ID}"
mkdir -p "$WORK_DIR"/{sources,encoded,sweeps}
cd "$WORK_DIR"

cleanup() {
    if [[ "$KEEP_WORK" != "1" ]]; then
        cd /; rm -rf "$WORK_DIR"
    fi
}
trap cleanup EXIT

R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
R2() { s5cmd --endpoint-url "$R2_ENDPOINT" --profile r2 "$@"; }

mkdir -p ~/.aws
cat > ~/.aws/credentials <<CREDS
[r2]
aws_access_key_id = $R2_ACCESS_KEY_ID
aws_secret_access_key = $R2_SECRET_ACCESS_KEY
CREDS

LOG="[omni-chunk-worker $CHUNK_ID]"

echo "$LOG step 1/5: download input parquet" >&2
R2 cp "$INPUT_PARQUET_R2" "$WORK_DIR/$INPUT_PARQUET" >&2

echo "$LOG step 2/5: sync sources" >&2
N_BASENAMES=$(jq -r '.image_basenames | length' <<< "$CHUNK_JSON")
echo "  $N_BASENAMES basenames" >&2
jq -r --arg src "$SOURCE_DIR_R2" '.image_basenames[] | "cp \($src)/\(.) \(.)"' <<< "$CHUNK_JSON" \
    > "$WORK_DIR/sources/_dl.run"
( cd "$WORK_DIR/sources" && s5cmd --endpoint-url "$R2_ENDPOINT" --profile r2 \
        run "$WORK_DIR/sources/_dl.run" >&2 ) || {
    echo "$LOG FAIL: source sync"; exit 3
}

echo "$LOG step 3/5: slice + group" >&2
python3 - "$WORK_DIR/$INPUT_PARQUET" "$ROW_START" "$ROW_END" \
        "$WORK_DIR/_groups.tsv" "$WORK_DIR/_keys.tsv" <<'PY'
import sys, json
import pyarrow.parquet as pq
p, rs, re_, out_groups, out_keys = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
t = pq.read_table(p, columns=['image_path','codec','q','knob_tuple_json']).slice(rs, re_ - rs)
# Group by (codec, q, knob_tuple_json) — same encoding for many images.
from collections import defaultdict
groups = defaultdict(list)
key_rows = []
for i in range(t.num_rows):
    ip = t['image_path'][i].as_py()
    cd = t['codec'][i].as_py()
    qv = int(t['q'][i].as_py())
    kt = t['knob_tuple_json'][i].as_py()
    basename = ip.rsplit('/', 1)[-1]
    groups[(cd, qv, kt)].append(basename)
    key_rows.append((ip, cd, qv, kt, basename))
with open(out_groups, 'w') as f:
    f.write('gid\tcodec\tq\tknob_tuple_json\tn\n')
    for gid, ((cd, qv, kt), bs) in enumerate(groups.items()):
        f.write(f'{gid}\t{cd}\t{qv}\t{kt}\t{len(bs)}\n')
with open(out_keys, 'w') as f:
    f.write('image_path\tcodec\tq\tknob_tuple_json\tbasename\n')
    for r in key_rows:
        f.write('\t'.join(str(x) for x in r) + '\n')
print(f'{t.num_rows} rows, {len(groups)} groups', file=sys.stderr)
PY

echo "$LOG step 4/5: sweep per group (multi-metric + encoded-out)" >&2
MERGED_TSV="$WORK_DIR/sweeps/merged.tsv"
HEADER_WRITTEN=0
G_TOTAL=$(awk 'NR>1' "$WORK_DIR/_groups.tsv" | wc -l)
G_IDX=0
while IFS=$'\t' read -r gid codec q kj n; do
    [[ "$gid" == "gid" ]] && continue
    G_IDX=$((G_IDX + 1))
    GD="$WORK_DIR/g$gid"
    mkdir -p "$GD/sources"
    awk -F'\t' -v g="$gid" 'NR>1 && $1==g {print $5}' "$WORK_DIR/_groups.tsv" >/dev/null  # placeholder
    awk -F'\t' -v c="$codec" -v qv="$q" -v k="$kj" 'NR>1 && $2==c && $3==qv && $4==k {print $5}' \
        "$WORK_DIR/_keys.tsv" | sort -u | while read -r b; do
        ln -sf "$WORK_DIR/sources/$b" "$GD/sources/$b" 2>/dev/null || true
    done
    SWEEP_ARGS=( sweep
        --codec "$codec"
        --sources "$GD/sources"
        --q-grid "$q"
        --output "$GD/sweep.tsv"
        --encoded-out-dir "$WORK_DIR/encoded"
        --gpu-runtime "$GPU_RUNTIME"
        --jobs "$PARALLEL" )
    if [[ "$kj" != "{}" && -n "$kj" ]]; then
        KGRID=$(echo "$kj" | jq -c 'with_entries(.value |= [.])')
        SWEEP_ARGS+=( --knob-grid "$KGRID" )
    fi
    IFS=',' read -ra MARR <<< "$METRICS"
    for m in "${MARR[@]}"; do SWEEP_ARGS+=( --metric "$m" ); done

    echo "$LOG   group $G_IDX/$G_TOTAL  codec=$codec q=$q n=$n" >&2
    if ! /usr/local/bin/zen-metrics "${SWEEP_ARGS[@]}" 2>&1 \
            | sed "s/^/  [g$gid] /" >&2; then
        echo "$LOG FAIL: sweep group g$gid"; exit 4
    fi

    if [[ "$HEADER_WRITTEN" == "0" ]]; then
        head -1 "$GD/sweep.tsv" > "$MERGED_TSV"
        HEADER_WRITTEN=1
    fi
    tail -n +2 "$GD/sweep.tsv" >> "$MERGED_TSV"
    rm -rf "$GD"  # free group scratch immediately (sources are symlinks)
done < "$WORK_DIR/_groups.tsv"

ENC_COUNT=$(ls "$WORK_DIR/encoded/" 2>/dev/null | wc -l)
echo "$LOG step 5/5: $(wc -l < "$MERGED_TSV") rows scored; $ENC_COUNT encoded variants" >&2

# Convert merged TSV → parquet sidecar (single file, all metric columns
# + encoded_filename + identity tuple).
python3 - "$MERGED_TSV" "$WORK_DIR/$CHUNK_ID.omni.parquet" "$CHUNK_ID" "$RUN_ID" "$OUT_ENCODED_PREFIX" <<'PY'
import sys
import pyarrow as pa, pyarrow.csv as pa_csv, pyarrow.parquet as pq, pyarrow.compute as pc
tsv_p, out_p, chunk_id, run_id, enc_prefix = sys.argv[1:6]
t = pa_csv.read_csv(tsv_p, parse_options=pa_csv.ParseOptions(delimiter='\t'))
# Add chunk_id, run_id, encoded_r2_uri columns.
n = t.num_rows
t = t.append_column('chunk_id', pa.array([chunk_id]*n))
t = t.append_column('run_id',   pa.array([run_id]*n))
# encoded_r2_uri = enc_prefix + encoded_filename (where present).
fn = t['encoded_filename']
def to_uri(s):
    return enc_prefix + s if s else ''
uri = [to_uri(v) for v in fn.to_pylist()]
t = t.append_column('encoded_r2_uri', pa.array(uri))
pq.write_table(t, out_p, compression='zstd')
print(f'wrote {out_p} ({t.num_rows} rows)', file=sys.stderr)
PY

if [[ "$SKIP_UPLOAD" == "1" ]]; then
    echo "$LOG SKIP_UPLOAD=1; sidecar at $WORK_DIR/$CHUNK_ID.omni.parquet" >&2
    KEEP_WORK=1
    exit 0
fi

echo "$LOG uploading encoded variants → $OUT_ENCODED_PREFIX" >&2
( cd "$WORK_DIR/encoded" && R2 cp --concurrency 8 '*' "$OUT_ENCODED_PREFIX" >&2 ) || {
    echo "$LOG FAIL: encoded upload"; exit 5
}

echo "$LOG uploading sidecar → $OUT_SIDECAR_R2" >&2
R2 cp "$WORK_DIR/$CHUNK_ID.omni.parquet" "$OUT_SIDECAR_R2" >&2

echo "$LOG done." >&2
