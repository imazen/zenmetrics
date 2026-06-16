#!/usr/bin/env bash
#
# omni_backfill_chunk_worker.sh — one-time multi-metric + encoded-
# variants backfill chunk worker.
#
# Differs from metric_backfill_chunk_worker.sh:
#   - Runs ALL metrics in one `zenmetrics sweep` pass (no separate
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
#   --metrics m1,m2,...    comma-list (default: zensim-gpu,ssim2-gpu,
#                          butteraugli-gpu,cvvdp,dssim-gpu,iwssim-gpu)
#                          NOTE: zensim is the CPU variant which is
#                          disabled in the production sweep binary
#                          (built without `cpu-metrics`). zensim-gpu
#                          gives the same score column; the 300-feat
#                          extended vector requires CPU zensim and
#                          would need a binary rebuild.
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

METRICS="${METRICS:-zensim-gpu,ssim2-gpu,butteraugli-gpu,cvvdp,dssim-gpu,iwssim-gpu}"
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

echo "$LOG step 3/5: slice + group by (codec, knob_tuple_json)" >&2
# Group by (codec, knob_tuple_json) — same encoder config, multiple q
# values + multiple source images per group. The previous (codec, q,
# knob_tuple_json) grouping spawned one zenmetrics-sweep invocation
# per (q, knob), eating ~3-5s of cubecl-cuda init per call. For a
# 200-row chunk with mostly-unique (q, knob) tuples that was ~200
# inits × 4s ≈ 13 min of overhead per chunk before any real work.
# Wider grouping lets each sweep call score many cells against one
# warm cubecl device.
python3 - "$WORK_DIR/$INPUT_PARQUET" "$ROW_START" "$ROW_END" \
        "$WORK_DIR/_groups.tsv" "$WORK_DIR/_keys.tsv" <<'PY'
import sys
import pyarrow.parquet as pq
p, rs, re_, out_groups, out_keys = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
t = pq.read_table(p, columns=['image_path','codec','q','knob_tuple_json']).slice(rs, re_ - rs)
from collections import defaultdict
# Each group: {(codec, knob_tuple_json) -> {'qs': set, 'basenames': set}}
groups = defaultdict(lambda: {'qs': set(), 'basenames': set()})
key_rows = []
for i in range(t.num_rows):
    ip = t['image_path'][i].as_py()
    cd = t['codec'][i].as_py()
    qv = int(t['q'][i].as_py())
    kt = t['knob_tuple_json'][i].as_py()
    basename = ip.rsplit('/', 1)[-1]
    key = (cd, kt)
    groups[key]['qs'].add(qv)
    groups[key]['basenames'].add(basename)
    key_rows.append((ip, cd, qv, kt, basename))
with open(out_groups, 'w') as f:
    # q_grid is now a comma-list (e.g. '5,10,15,...'); each group covers
    # the full q range it actually needs.
    f.write('gid\tcodec\tknob_tuple_json\tq_grid\tn_q\tn_basenames\n')
    for gid, ((cd, kt), v) in enumerate(groups.items()):
        qs = sorted(v['qs'])
        f.write(f'{gid}\t{cd}\t{kt}\t{",".join(str(q) for q in qs)}\t{len(qs)}\t{len(v["basenames"])}\n')
with open(out_keys, 'w') as f:
    f.write('image_path\tcodec\tq\tknob_tuple_json\tbasename\n')
    for r in key_rows:
        f.write('\t'.join(str(x) for x in r) + '\n')
print(f'{t.num_rows} rows, {len(groups)} groups (avg q/grp={sum(len(v["qs"]) for v in groups.values())/max(1,len(groups)):.1f}, avg images/grp={sum(len(v["basenames"]) for v in groups.values())/max(1,len(groups)):.1f})', file=sys.stderr)
PY

echo "$LOG step 4/5: sweep per group (multi-metric + encoded-out)" >&2
SWEEP_DIR="$WORK_DIR/sweeps"
mkdir -p "$SWEEP_DIR"
G_TOTAL=$(awk 'NR>1' "$WORK_DIR/_groups.tsv" | wc -l)
G_IDX=0
GROUPS_OK=0
GROUPS_FAIL=0
while IFS=$'\t' read -r gid codec kj q_grid n_q n_bn; do
    [[ "$gid" == "gid" ]] && continue
    G_IDX=$((G_IDX + 1))
    GD="$WORK_DIR/g$gid"
    mkdir -p "$GD/sources"
    awk -F'\t' -v c="$codec" -v k="$kj" 'NR>1 && $2==c && $4==k {print $5}' \
        "$WORK_DIR/_keys.tsv" | sort -u | while read -r b; do
        ln -sf "$WORK_DIR/sources/$b" "$GD/sources/$b" 2>/dev/null || true
    done
    SWEEP_ARGS=( sweep
        --codec "$codec"
        --sources "$GD/sources"
        --q-grid "$q_grid"
        --output "$SWEEP_DIR/g${gid}.tsv"
        --distorted-out-dir "$WORK_DIR/encoded"
        --gpu-runtime "$GPU_RUNTIME"
        --jobs "$PARALLEL" )
    if [[ "$kj" != "{}" && -n "$kj" ]]; then
        KGRID=$(echo "$kj" | jq -c 'with_entries(.value |= [.])')
        SWEEP_ARGS+=( --knob-grid "$KGRID" )
    fi
    IFS=',' read -ra MARR <<< "$METRICS"
    for m in "${MARR[@]}"; do SWEEP_ARGS+=( --metric "$m" ); done

    echo "$LOG   group $G_IDX/$G_TOTAL  codec=$codec knobs=${kj:0:40} qs=$n_q imgs=$n_bn" >&2
    # Tolerate per-group failures. The omni design is "produce whatever
    # we can; the next backfill pass picks up the rest." A panic in
    # group g123 must not poison groups g0..g122's already-written
    # sweep.tsv files. Track success/fail counts; final parquet
    # conversion reads only the surviving group TSVs.
    if /usr/local/bin/zenmetrics "${SWEEP_ARGS[@]}" 2>&1 \
            | sed "s/^/  [g$gid] /" >&2; then
        GROUPS_OK=$((GROUPS_OK + 1))
    else
        GROUPS_FAIL=$((GROUPS_FAIL + 1))
        # Salvage: if zenmetrics wrote ANY rows before dying, keep
        # the file. Otherwise drop the stub to avoid header-only
        # files cluttering the merge.
        if [[ -f "$SWEEP_DIR/g${gid}.tsv" ]] && \
           (( $(wc -l < "$SWEEP_DIR/g${gid}.tsv") <= 1 )); then
            rm -f "$SWEEP_DIR/g${gid}.tsv"
        fi
    fi
    rm -rf "$GD"  # free group scratch immediately (sources are symlinks)
done < "$WORK_DIR/_groups.tsv"

SWEEP_TSV_COUNT=$(ls "$SWEEP_DIR"/g*.tsv 2>/dev/null | wc -l)
ENC_COUNT=$(ls "$WORK_DIR/encoded/" 2>/dev/null | wc -l)
echo "$LOG step 5/5: $GROUPS_OK ok, $GROUPS_FAIL fail; $SWEEP_TSV_COUNT TSVs; $ENC_COUNT encoded variants" >&2

if (( SWEEP_TSV_COUNT == 0 )); then
    echo "$LOG ERROR: no group produced output — exiting non-zero to trigger trap" >&2
    exit 6
fi

# Convert per-group TSVs → one parquet sidecar. Reading each TSV
# separately + concat (instead of bash-merging then reading once) means
# one corrupt group can't take down the whole chunk: pa_csv's
# invalid_row_handler skips malformed rows; a fully-broken group's
# file gets caught + logged + skipped.
python3 - "$SWEEP_DIR" "$WORK_DIR/$CHUNK_ID.omni.parquet" "$CHUNK_ID" "$RUN_ID" "$OUT_ENCODED_PREFIX" <<'PY'
import os, sys, glob
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

sweep_dir, out_p, chunk_id, run_id, enc_prefix = sys.argv[1:6]

n_bad_rows = [0]
def on_bad_row(row):
    n_bad_rows[0] += 1
    if n_bad_rows[0] <= 5:
        sys.stderr.write(f'WARN: skipping malformed row (#{n_bad_rows[0]}): {row}\n')
    return 'skip'

parse_opts = pa_csv.ParseOptions(delimiter='\t', invalid_row_handler=on_bad_row)
conv_opts = pa_csv.ConvertOptions(strings_can_be_null=True)

tables = []
group_files = sorted(glob.glob(os.path.join(sweep_dir, 'g*.tsv')))
sys.stderr.write(f'reading {len(group_files)} group TSVs\n')
for gf in group_files:
    try:
        t = pa_csv.read_csv(gf, parse_options=parse_opts, convert_options=conv_opts)
        if t.num_rows > 0:
            tables.append(t)
    except Exception as e:
        sys.stderr.write(f'WARN: skipping {os.path.basename(gf)} (parse failed: {e})\n')

if not tables:
    sys.stderr.write('ERROR: zero usable rows after parse\n')
    sys.exit(7)

# Concat. Schema mismatches across groups (e.g. one group dropped a
# metric column) get the promote=True treatment so missing columns
# fill with NULL.
t = pa.concat_tables(tables, promote_options='default')
n = t.num_rows
t = t.append_column('chunk_id', pa.array([chunk_id]*n))
t = t.append_column('run_id',   pa.array([run_id]*n))
fn = t['encoded_filename']
uri = [(enc_prefix + s) if s else '' for s in fn.to_pylist()]
t = t.append_column('encoded_r2_uri', pa.array(uri))
pq.write_table(t, out_p, compression='zstd')
sys.stderr.write(f'wrote {out_p} ({t.num_rows} rows, {n_bad_rows[0]} bad rows skipped)\n')
PY

if [[ "$SKIP_UPLOAD" == "1" ]]; then
    echo "$LOG SKIP_UPLOAD=1; sidecar at $WORK_DIR/$CHUNK_ID.omni.parquet" >&2
    KEEP_WORK=1
    exit 0
fi

# Upload the sidecar FIRST — it's the primary artifact. Encoded
# variants are bonus output; failing their upload must not strand
# the sidecar.
echo "$LOG uploading sidecar → $OUT_SIDECAR_R2" >&2
R2 cp "$WORK_DIR/$CHUNK_ID.omni.parquet" "$OUT_SIDECAR_R2" >&2

# Encoded variants: tolerate empty directory (--distorted-out-dir
# only writes when paired with --pairs-tsv, which the worker doesn't
# pass — so the dir is usually empty in the current pipeline). Use
# `find` to gate the s5cmd call; never let an empty upload fail the
# whole chunk.
ENC_FILES=$(find "$WORK_DIR/encoded" -mindepth 1 -maxdepth 1 -type f 2>/dev/null | wc -l)
if (( ENC_FILES > 0 )); then
    echo "$LOG uploading $ENC_FILES encoded variants → $OUT_ENCODED_PREFIX" >&2
    ( cd "$WORK_DIR/encoded" && R2 cp --concurrency 8 '*' "$OUT_ENCODED_PREFIX" >&2 ) || \
        echo "$LOG WARN: encoded upload failed (continuing)" >&2
else
    echo "$LOG no encoded variants to upload (skipping)" >&2
fi

echo "$LOG done." >&2
