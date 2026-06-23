#!/usr/bin/env bash
# datagen_encode.sh — encode-half of the encode→GPU-metric→parquet data run.
# For each codec: sweep (encode-only) the full rendition corpus (all sizes) →
# persist variants + omni (with encoded_filename) → emit pairs.tsv (dist =
# encoded variant, with the vast-box /data paths the SPLIT scorer expects) →
# content-address index the variants (sha256) → tar + upload to R2. The GPU
# score fleet (split_score_worker.sh) consumes <prefix>/<codec>/{variants.tar,
# pairs.tsv} + <prefix>/ref/. MLP/picker training is deferred (zenanalyze WIP).
# jxl is excluded (OOM bug: imazen/jxl-encoder#93).
set -uo pipefail
B="${ZB:-target/release/zenmetrics}"
R="${REND:-/mnt/v/output/imazen-26-features/train_renditions_2026-06-14}"
OUT="${OUT:-/mnt/v/output/zenmetrics/datagen-2026-06-23}"
QG="${QG:-5,15,30,50,70,85,95}"
PLAN="${PLAN:-rd_core}"
CODECS="${CODECS:-zenjpeg zenwebp zenavif zenpng}"
BUCKET="${BUCKET:-codec-corpus}"
PREFIX="${PREFIX:-picker-sweep-2026-06-22/datagen-2026-06-23}"
mkdir -p "$OUT"/{enc,omni,pairs,index,log}

set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
r2(){ AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto s5cmd --endpoint-url "$EP" "$@"; }

# ref renditions uploaded once (shared across codecs; the scorer's ref images)
if ! r2 ls "s3://$BUCKET/$PREFIX/ref/" 2>/dev/null | grep -q '\.png'; then
  echo "[datagen] uploading ref renditions once..."
  r2 cp "$R/*" "s3://$BUCKET/$PREFIX/ref/" >/dev/null 2>&1 || true
fi

for codec in $CODECS; do
  ts(){ date -u +%H:%M:%S; }
  enc="$OUT/enc/$codec"; omni="$OUT/omni/$codec.tsv"; pairs="$OUT/pairs/$codec.pairs.tsv"
  idx="$OUT/index/$codec.encidx.tsv"; log="$OUT/log/$codec.log"
  rm -rf "$enc"; mkdir -p "$enc"
  echo "[datagen $(ts)] === encode $codec ($PLAN) ===" | tee -a "$log"
  "$B" sweep --codec "$codec" --sources "$R" --q-grid "$QG" --plan "$PLAN" \
    --encoded-out-dir "$enc" --output "$omni" >>"$log" 2>&1
  rc=$?; echo "[datagen $(ts)] $codec sweep rc=$rc variants=$(ls "$enc" 2>/dev/null | wc -l)" | tee -a "$log"
  [ "$rc" = 0 ] || { echo "[datagen] $codec encode FAILED — skipping"; continue; }

  # pairs.tsv (dist = encoded variant; vast-box paths)
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
  # content-address index the variants (sha256 + dims + size_class)
  python3 scripts/provenance/index_corpus.py --dir "$enc" --out "$idx" --name-dims --glob '*' --jobs 8 >>"$log" 2>&1 || true

  # tar variants + upload everything for the score fleet
  tar -cf "$OUT/enc/$codec.tar" -C "$enc" .
  r2 cp "$OUT/enc/$codec.tar" "s3://$BUCKET/$PREFIX/$codec/variants.tar" >/dev/null 2>&1
  r2 cp "$pairs" "s3://$BUCKET/$PREFIX/$codec/pairs.tsv" >/dev/null 2>&1
  r2 cp "$omni"  "s3://$BUCKET/$PREFIX/$codec/omni.tsv"  >/dev/null 2>&1
  r2 cp "$idx"   "s3://$BUCKET/$PREFIX/$codec/encode_index.tsv" >/dev/null 2>&1
  echo "[datagen $(ts)] $codec uploaded: $(wc -l < "$pairs") pairs, $(wc -l < "$idx") indexed variants" | tee -a "$log"
  rm -f "$OUT/enc/$codec.tar"
done
echo "[datagen $(date -u +%H:%M:%S)] ALL CODECS DONE"
