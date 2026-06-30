#!/usr/bin/env bash
# HQ-fill A re-sweep (2026-07-01) — zenjxl LOSSY e7, native butteraugli distance
# grid {0.05..1.3}, 4497 clean-picker renditions. Stores EVERYTHING so it is
# forever re-scorable and A-training-joinable:
#   - 372 zensim features (WithIw regime) per cell  (--feature-output)
#   - content-addressed encoded .jxl variants        (--encoded-out-dir)
#   - 6 metrics: zensim-A (gpu==cpu, 372 feat), ssim2 (CPU, canonical),
#     butteraugli max+pnorm3 (gpu; CPU pnorm3 broken), cvvdp (gpu),
#     dssim (CPU==gpu), iwssim (gpu; CPU fails <176px)
# Chunk-mode: ONE fresh sweep process per chunk => bounded GPU pool + resumable.
# Runs under run-heavy (nice/ionice/mem-cap) so it coexists with other agents.
set -euo pipefail

BIN=/home/lilith/work/zen/zenmetrics/target/release/zenmetrics
OUT=/mnt/v/output/jxl-hqfill-A-2026-07-01
DIST='0.05,0.08,0.11,0.14,0.17,0.2,0.25,0.3,0.35,0.45,0.6,0.8,1.0,1.3'
PROGRESS=$OUT/progress.log
AGENT_ID=claude-hqfill-redo
REPO=/home/lilith/work/zen/zenmetrics

mkdir -p "$OUT"/{chunks,encoded,features,tsv,srcdirs}

log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$PROGRESS"; }

CHUNK_LISTS=("$OUT"/chunks/chunk_*.list)
TOTAL=${#CHUNK_LISTS[@]}
log "START hqfill-A: $TOTAL chunks, 14 distances, 6 metrics + 372 feat"

done_cells=0
for LIST in "${CHUNK_LISTS[@]}"; do
  CID=$(basename "$LIST" .list)              # chunk_NN
  DONE_MARK=$OUT/tsv/$CID.done
  TSV=$OUT/tsv/$CID.tsv
  FEAT=$OUT/features/$CID.features.parquet
  SRCDIR=$OUT/srcdirs/$CID

  # refresh workongoing marker
  printf '%s %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$AGENT_ID" "hqfill-A sweep $CID" > "$REPO/.workongoing"

  if [[ -f "$DONE_MARK" ]]; then
    n=$(($(wc -l < "$TSV") - 1))
    log "SKIP $CID (already done, $n cells)"
    done_cells=$((done_cells + n))
    continue
  fi

  # build symlink source dir (no copy)
  rm -rf "$SRCDIR"; mkdir -p "$SRCDIR"
  while IFS= read -r p; do ln -s "$p" "$SRCDIR/$(basename "$p")"; done < "$LIST"
  nrend=$(ls "$SRCDIR" | wc -l)

  log "RUN $CID: $nrend renditions x 14 distances = $((nrend*14)) cells"
  t0=$(date +%s)
  # ZENSIM_FEATURES_REGIME belt-and-suspenders with --zensim-features-regime.
  ZENSIM_FEATURES_REGIME=with-iw "$BIN" sweep \
    --codec zenjxl \
    --sources "$SRCDIR" \
    --q-grid 90 \
    --knob-grid "{\"distance\":[$DIST],\"effort\":[7]}" \
    --metric zensim-gpu \
    --metric ssim2 \
    --metric butteraugli-gpu \
    --metric cvvdp-gpu \
    --metric dssim \
    --metric iwssim-gpu \
    --zensim-features-regime with-iw \
    --feature-output "$FEAT" \
    --encoded-out-dir "$OUT/encoded" \
    --output "$TSV" \
    --gpu-runtime cuda \
    --jobs 1 \
    >"$OUT/tsv/$CID.stdout" 2>&1 || { log "FAIL $CID (see $OUT/tsv/$CID.stdout)"; tail -5 "$OUT/tsv/$CID.stdout" | tee -a "$PROGRESS"; exit 1; }

  t1=$(date +%s)
  # verify chunk: cell count + feature non-null count
  ncells=$(($(wc -l < "$TSV") - 1))
  nfeat=$(python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('$FEAT').num_rows)" 2>/dev/null || echo 0)
  rate=$(python3 -c "print(f'{$ncells/max(1,$t1-$t0):.1f}')")
  log "OK $CID: $ncells cells, $nfeat feature rows, $((t1-t0))s (${rate} cells/s)"
  if [[ "$ncells" -ne "$nfeat" ]]; then
    log "WARN $CID: TSV cells ($ncells) != feature rows ($nfeat)"
  fi
  rm -rf "$SRCDIR"
  touch "$DONE_MARK"
  done_cells=$((done_cells + ncells))
  log "PROGRESS: $done_cells cells total so far"
done

log "ALL CHUNKS DONE: $done_cells cells across $TOTAL chunks"
