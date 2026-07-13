#!/usr/bin/env bash
# datagen_encode_hdr.sh — HDR encode-half of the encode -> GPU-metric -> parquet
# data run, the HDR sibling of datagen_encode.sh.
#
# Only ONE codec is HDR-capable in this pipeline: zenjxl (16-bit PQ + CICP
# round-trip, validated by zenmetrics sweep::hdr::validate_hdr_sweep). zenavif /
# zenjpeg / zenwebp / zenpng have NO wired HDR encode+decode path and the sweep
# refuses them in --hdr mode (it would fake the scores by crushing nits to 8-bit
# SDR), so they are NOT encoded here. Do not "add HDR coverage" for them without
# first wiring a true HDR path in sweep::hdr.
#
# HDR-mode sweep differences vs the SDR datagen:
#   * --hdr is set; the encode path takes the 16-bit PQ .hdr.png reference,
#     encodes JXL with CICP, decodes the variant back to nits, and (because a
#     metric is required) scores one cheap inline metric per cell. The encoded
#     .jxl is what we persist; the real 6-metric + 372-feature scoring is a
#     separate score-pairs --hdr pass (datagen_score_hdr.sh).
#   * --plan is REFUSED in HDR mode (plan cells take the RGB8-typed path), so we
#     sweep a --q-grid. The grid is dense at low q per CLAUDE.md (web-focused
#     compression: every byte matters at q5-q40).
#   * --distorted-out-dir / --feature-output are REFUSED in HDR mode (they write
#     8-bit / take u8 sRGB). The score fleet uses the encoded .jxl directly as
#     dist_path; score-pairs --hdr decodes it back to nits (hdr::decode_pq_jxl).
#   * the corpus is encoded in IMAGE-COUNT CHUNKS so each sweep process resets
#     the jxl-encoder butteraugli BufferPool between chunks (the modes_full
#     fleet OOM — imazen/jxl-encoder#93 — is config-dependent; a --q-grid encode
#     of the largest 7 MP renditions peaks ~8 GB and does NOT grow with cell
#     count, but chunking keeps headroom and bounds any slow creep).
#
# Persists per chunk, then merges: variants + omni (with encoded_filename) +
# pairs.tsv (dist = encoded .jxl, /data paths the SPLIT scorer expects) +
# sha256 content-address index. tar + upload to R2.
#
#   R2 layout (mirrors the SDR prefix, ref shared, zenjxl only):
#     s3://$BUCKET/$PREFIX/ref/<rendition>.hdr.png          (uploaded once)
#     s3://$BUCKET/$PREFIX/zenjxl/variants.tar
#     s3://$BUCKET/$PREFIX/zenjxl/pairs.tsv
#     s3://$BUCKET/$PREFIX/zenjxl/omni.tsv
#     s3://$BUCKET/$PREFIX/zenjxl/encode_index.tsv
set -uo pipefail

B="${ZB:-target/release/zenmetrics}"
# HDR renditions (16-bit PQ PNG-3.0, swept across sizes).
R="${REND:-/mnt/v/output/imazen-26-hdr-grid-2026-06-14}"
OUT="${OUT:-/mnt/v/output/zenmetrics/datagen-2026-06-23-hdr}"
# Dense low-q grid (CLAUDE.md: q5-q60 same density as q60-q100, web focus).
QG="${QG:-5,15,30,50,70,85,95}"
CODEC="${CODEC:-zenjxl}"   # HDR-capable: zenjxl, zenavif (10-bit PQ, 2026-07-12)
BUCKET="${BUCKET:-codec-corpus}"
PREFIX="${PREFIX:-picker-sweep-2026-06-22/datagen-2026-06-23-hdr}"
# Inline metric the HDR sweep needs (cheap; the real scoring is score-pairs).
INLINE_METRIC="${INLINE_METRIC:-ssim2-gpu}"
GPU_RUNTIME="${GPU_RUNTIME:-cuda}"
# Renditions per sweep process. ~8 GB peak on all-7MP chunks; 80 keeps headroom.
CHUNK="${CHUNK:-80}"
MEM="${MEM:-36G}"
JOBS="${JOBS:-8}"

mkdir -p "$OUT"/{enc,omni,pairs,index,log,chunks}

set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
r2(){ AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto s5cmd --endpoint-url "$EP" "$@"; }
ts(){ date -u +%H:%M:%S; }

log="$OUT/log/$CODEC.log"
enc="$OUT/enc/$CODEC"; omni="$OUT/omni/$CODEC.tsv"; pairs="$OUT/pairs/$CODEC.pairs.tsv"
idx="$OUT/index/$CODEC.encidx.tsv"
rm -rf "$enc"; mkdir -p "$enc"

# ── ref renditions uploaded once (shared; the scorer's ref images) ──
if ! r2 ls "s3://$BUCKET/$PREFIX/ref/" 2>/dev/null | grep -q '\.png'; then
  echo "[hdr-datagen $(ts)] uploading ref renditions once ($(ls "$R"/*.hdr.png | wc -l) files)..." | tee -a "$log"
  r2 cp "$R/*.hdr.png" "s3://$BUCKET/$PREFIX/ref/" >/dev/null 2>&1 || true
fi

# ── chunk the corpus by image count, encode each chunk in its own process ──
mapfile -t ALL < <(ls "$R"/*.hdr.png | sort)
N="${#ALL[@]}"
echo "[hdr-datagen $(ts)] === encode $CODEC --hdr (q-grid $QG): $N renditions, chunk=$CHUNK ===" | tee -a "$log"

: > "$omni.parts"   # collect chunk omni paths
chunk_no=0
for ((i=0; i<N; i+=CHUNK)); do
  chunk_no=$((chunk_no+1))
  cdir="$OUT/chunks/c$(printf '%03d' "$chunk_no")"
  rm -rf "$cdir"; mkdir -p "$cdir/src"
  for ((j=i; j<i+CHUNK && j<N; j++)); do ln -sf "${ALL[$j]}" "$cdir/src/$(basename "${ALL[$j]}")"; done
  comni="$cdir/omni.tsv"
  nch=$(ls "$cdir/src"/ | wc -l)
  echo "[hdr-datagen $(ts)] chunk $chunk_no: $nch renditions" | tee -a "$log"
  ~/work/zen/scripts/run-heavy --mem "$MEM" --jobs "$JOBS" -- \
    "$B" sweep --codec "$CODEC" --sources "$cdir/src" --q-grid "$QG" --hdr \
    --metric "$INLINE_METRIC" --gpu-runtime "$GPU_RUNTIME" \
    --encoded-out-dir "$enc" --output "$comni" >>"$log" 2>&1
  rc=$?
  echo "[hdr-datagen $(ts)] chunk $chunk_no rc=$rc variants_total=$(ls "$enc" 2>/dev/null | wc -l)" | tee -a "$log"
  [ "$rc" = 0 ] || echo "[hdr-datagen] chunk $chunk_no rc=$rc — salvaging on-disk variants, continuing" | tee -a "$log"
  echo "$comni" >> "$omni.parts"
  rm -rf "$cdir/src"     # symlinks only; the encoded variants live in $enc
done

# ── merge chunk omnis (header once) ──
first=1
: > "$omni"
while read -r p; do
  [ -f "$p" ] || continue
  if [ "$first" = 1 ]; then cat "$p" > "$omni"; first=0; else tail -n +2 "$p" >> "$omni"; fi
done < "$omni.parts"
echo "[hdr-datagen $(ts)] merged omni rows=$(($(wc -l < "$omni")-1)) variants=$(ls "$enc" | wc -l)" | tee -a "$log"

# ── pairs.tsv (dist = encoded .jxl variant; vast-box /data paths) ──
python3 - "$omni" "$pairs" <<'PY'
import csv, os, sys
omni, out = sys.argv[1], sys.argv[2]
n = 0
with open(omni) as f, open(out, "w") as o:
    r = csv.DictReader(f, delimiter="\t")
    o.write("image_path\tcodec\tq\tknob_tuple_json\tref_path\tdist_path\n")
    for row in r:
        ef = row.get("encoded_filename") or ""
        if not ef:
            continue
        base = os.path.basename(row["image_path"])
        o.write("\t".join([row["image_path"], row["codec"], row["q"],
                           row.get("knob_tuple_json", ""),
                           f"/data/ref/{base}", f"/data/variants/{ef}"]) + "\n")
        n += 1
print(f"pairs={n}")
PY

# ── content-address index (sha256 + dims + size_class) ──
python3 scripts/provenance/index_corpus.py --dir "$enc" --out "$idx" --name-dims --glob '*' --jobs 8 >>"$log" 2>&1 || true

# ── tar variants + upload everything for the score pass ──
tar -cf "$OUT/enc/$CODEC.tar" -C "$enc" .
r2 cp "$OUT/enc/$CODEC.tar" "s3://$BUCKET/$PREFIX/$CODEC/variants.tar" >/dev/null 2>&1
r2 cp "$pairs" "s3://$BUCKET/$PREFIX/$CODEC/pairs.tsv" >/dev/null 2>&1
r2 cp "$omni"  "s3://$BUCKET/$PREFIX/$CODEC/omni.tsv"  >/dev/null 2>&1
r2 cp "$idx"   "s3://$BUCKET/$PREFIX/$CODEC/encode_index.tsv" >/dev/null 2>&1
echo "[hdr-datagen $(ts)] $CODEC uploaded: $(($(wc -l < "$pairs")-1)) pairs, $(($(wc -l < "$idx")-1)) indexed variants, tar=$(du -h "$OUT/enc/$CODEC.tar" | cut -f1)" | tee -a "$log"
rm -f "$OUT/enc/$CODEC.tar"
echo "[hdr-datagen $(date -u +%H:%M:%S)] DONE ($CODEC only — the lone HDR-capable codec)" | tee -a "$log"
