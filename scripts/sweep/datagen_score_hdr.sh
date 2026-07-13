#!/usr/bin/env bash
# datagen_score_hdr.sh — LOCAL HDR score-half of the datagen run, the HDR
# sibling of split_score_worker.sh. Runs on the local 5070 (the vast SPLIT
# image's binary lacks --hdr, so HDR scoring stays local) over the zenjxl HDR
# variants persisted by datagen_encode_hdr.sh.
#
# For each of the 6 GPU metrics it runs `zenmetrics score-pairs --hdr` over the
# (ref .hdr.png, dist .jxl) pairs. The dist is the ENCODED JXL — score-pairs
# --hdr decodes it back to absolute nits (hdr::decode_pq_jxl, requires the jxl
# build feature) and applies the validated per-metric HDR feeding (cvvdp +
# butteraugli-gpu native linear planes; ssim2/dssim/iwssim/zensim PU21 u8
# shell). The zensim pass also emits the 372-feature (with-iw) sidecar from the
# PU21 u8 feeding. Sidecars land at <prefix>/sidecars/<metric>.parquet (+
# zensim_features.parquet), mirroring the SPLIT layout.
#
# Binary MUST be built with the hdr + jxl features:
#   cargo build --release -p zenmetrics-cli --no-default-features \
#     --features sweep,png,jpeg,webp,avif,jxl,cpu-metrics,gpu,gpu-cuda,hdr
set -uo pipefail

B="${ZB:-target/release/zenmetrics}"
R="${REND:-/mnt/v/output/imazen-26-hdr-grid-2026-06-14}"
OUT="${OUT:-/mnt/v/output/zenmetrics/datagen-2026-06-23-hdr}"
# CODEC is overridable for non-encode HDR families (e.g. kadis-hdr synthetic
# distortions, 2026-07-12) whose driver writes a ready pairs TSV — set
# PAIRS=<path> to skip the omni->pairs build below and score that TSV as-is.
CODEC="${CODEC:-zenjxl}"
BUCKET="${BUCKET:-codec-corpus}"
PREFIX="${PREFIX:-picker-sweep-2026-06-22/datagen-2026-06-23-hdr}"
METRICS="${METRICS:-cvvdp ssim2-gpu dssim-gpu butteraugli-gpu zensim-gpu iwssim}"; METRICS="${METRICS//,/ }"
GPU_RUNTIME="${GPU_RUNTIME:-cuda}"
HDR_TRANSFER="${HDR_TRANSFER:-pu-rescale}"
MEM="${MEM:-24G}"
JOBS="${JOBS:-8}"
# Upload sidecars to R2 (set UPLOAD=0 to keep local only).
UPLOAD="${UPLOAD:-1}"

enc="$OUT/enc/$CODEC"
omni="$OUT/omni/$CODEC.tsv"
sdir="$OUT/sidecars/$CODEC"; mkdir -p "$sdir"
log="$OUT/log/$CODEC.score.log"
ts(){ date -u +%H:%M:%S; }

# ── build a LOCAL pairs.tsv from the omni: ref -> $R/<base>, dist -> $enc/<ef> ──
lpairs="${PAIRS:-$OUT/pairs/$CODEC.local.pairs.tsv}"; mkdir -p "$(dirname "$lpairs")"
[ -n "${PAIRS:-}" ] && [ -s "$PAIRS" ] || \
python3 - "$omni" "$R" "$enc" "$lpairs" <<'PY'
import csv, os, sys
omni, refdir, encdir, out = sys.argv[1:5]
n = 0
with open(omni) as f, open(out, "w") as o:
    r = csv.DictReader(f, delimiter="\t")
    o.write("image_path\tcodec\tq\tknob_tuple_json\tref_path\tdist_path\n")
    for row in r:
        ef = row.get("encoded_filename") or ""
        if not ef:
            continue
        base = os.path.basename(row["image_path"])
        refp = os.path.join(refdir, base)
        distp = os.path.join(encdir, ef)
        if not (os.path.exists(refp) and os.path.exists(distp)):
            continue
        o.write("\t".join([row["image_path"], row["codec"], row["q"],
                           row.get("knob_tuple_json", ""), refp, distp]) + "\n")
        n += 1
print(f"local_pairs={n}")
PY
echo "[hdr-score $(ts)] $(($(wc -l < "$lpairs")-1)) local pairs from omni" | tee -a "$log"

if [ "$UPLOAD" = 1 ]; then
  set -a; . ~/.config/cloudflare/r2-credentials; set +a
  EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
  r2(){ AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto s5cmd --endpoint-url "$EP" "$@"; }
fi

rc=0
for m in $METRICS; do
  echo "[hdr-score $(ts)] score-pairs --hdr --metric $m" | tee -a "$log"
  feat=()
  case "$m" in zensim-gpu|zensim) feat=(--feature-output "$sdir/zensim_features.parquet" --zensim-features-regime with-iw);; esac
  # FIRST-PAIR GATE (2026-07-03): a feature-stripped binary "succeeds" while
  # writing 100% NaN rows (the zensim-gpu incident). Score ONE pair first and
  # refuse the run if it errors or yields NaN.
  probe=$(mktemp -d); head -2 "$lpairs" > "$probe/one.tsv"
  if ! "$B" score-pairs --metric "$m" --hdr --hdr-transfer "$HDR_TRANSFER" \
      --pairs-tsv "$probe/one.tsv" --out-parquet "$probe/one.parquet" \
      --gpu-runtime "$GPU_RUNTIME" >"$probe/log" 2>&1 \
      || grep -qiE "failed:|requires a .* build" "$probe/log" \
      || ! python3 -c "import pyarrow.parquet as pq,math,sys; t=pq.read_table('$probe/one.parquet'); c=[x for x in t.schema.names if x not in ('image_path','codec','q','knob_tuple_json')][0]; v=t[c][0].as_py(); sys.exit(0 if v is not None and not math.isnan(v) else 1)"; then
    echo "[hdr-score $(ts)] $m FIRST-PAIR GATE FAILED — binary lacks this metric's HDR path; fix the build (see header). log:" | tee -a "$log"
    tail -3 "$probe/log" | tee -a "$log"; rm -rf "$probe"; continue
  fi
  rm -rf "$probe"
  if ~/work/zen/scripts/run-heavy --mem "$MEM" --jobs "$JOBS" -- \
      "$B" score-pairs --metric "$m" --hdr --hdr-transfer "$HDR_TRANSFER" \
      --pairs-tsv "$lpairs" --out-parquet "$sdir/$m.parquet" \
      --gpu-runtime "$GPU_RUNTIME" "${feat[@]}" >>"$log" 2>&1; then
    rows=$(python3 -c "import pyarrow.parquet as pq;print(pq.read_metadata('$sdir/$m.parquet').num_rows)" 2>/dev/null || echo '?')
    echo "[hdr-score $(ts)] $m OK rows=$rows" | tee -a "$log"
    if [ "$UPLOAD" = 1 ]; then
      r2 cp "$sdir/$m.parquet" "s3://$BUCKET/$PREFIX/$CODEC/sidecars/$m.parquet" >/dev/null 2>&1
      if [ "${#feat[@]}" -gt 0 ] && [ -f "$sdir/zensim_features.parquet" ]; then
        r2 cp "$sdir/zensim_features.parquet" "s3://$BUCKET/$PREFIX/$CODEC/sidecars/zensim_features.parquet" >/dev/null 2>&1
        echo "[hdr-score $(ts)] zensim_features sidecar uploaded" | tee -a "$log"
      fi
    fi
  else
    rc=1; echo "[hdr-score $(ts)] $m FAILED (see $log)" | tee -a "$log"
  fi
done
echo "[hdr-score $(ts)] done rc=$rc metrics='$METRICS' sidecars in $sdir" | tee -a "$log"
exit "$rc"
