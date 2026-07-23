#!/usr/bin/env bash
# OOM-PROOF resume of the hqfill-A sweep for the remaining renditions (chunks 11-29).
#
# ROOT CAUSE of the 2026-07-01 death: the monolithic `zenmetrics sweep` accumulates
# allocator high-water ACROSS the cells in ONE process (documented jxl bug:
# sweep-oom-is-allocator-highwater-not-metric, OOM ∝ cells×size). chunk_11 held
# ~2086 cells of LARGE renditions (1024², 896², 768²) in one process → 25 GB anon-rss
# → cgroup OOM at the 40 G cap. Measured: 8×1024² renditions × 14 dist = 112 cells
# peaks at only 0.89 GB RSS — the high-water is bounded by CELLS-PER-PROCESS.
#
# FIX (OOM-proof by construction, LOCAL, $0, byte-identical to the salvaged chunks
# 0-10 because it uses the SAME binary + SAME metric backends → clean merge, zero
# metric-rev drift): batch the remaining renditions into SMALL per-process units
# (RENDS_PER_BATCH renditions ≈ RENDS_PER_BATCH×14 cells). One FRESH `zenmetrics
# sweep` process per batch → RSS freed on exit → never approaches the cap. This is
# the same "one encode per fresh process" bound the job system provides, achieved
# locally without a fleet/image/cred/cost.
set -euo pipefail

BIN=/home/lilith/work/zen/zenmetrics/target/release/zenmetrics
OUT=/mnt/v/output/jxl-hqfill-A-2026-07-01
DIST='0.05,0.08,0.11,0.14,0.17,0.2,0.25,0.3,0.35,0.45,0.6,0.8,1.0,1.3'
RENDS_PER_BATCH=20                       # 20 renditions × 14 dist = 280 cells/process (<1 GB even for 1024²)
PROGRESS=$OUT/resume_progress.log
AGENT_ID=claude-hqfill-redo
REPO=/home/lilith/work/zen/zenmetrics
REMAIN=/tmp/remaining_rends.txt          # 2847 renditions (chunks 11-29)

mkdir -p "$OUT"/{batches,encoded,bfeatures,btsv,bsrc}

log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$PROGRESS"; }

# Split remaining renditions into batches of RENDS_PER_BATCH (deterministic)
split -l "$RENDS_PER_BATCH" -d -a 3 --additional-suffix=.list "$REMAIN" "$OUT/batches/b_"
BATCHES=("$OUT"/batches/b_*.list)
TOTAL=${#BATCHES[@]}
log "RESUME START: $TOTAL batches × ≤$RENDS_PER_BATCH renditions (remaining 2847 rends, 39858 cells)"

done_cells=0
for LIST in "${BATCHES[@]}"; do
  BID=$(basename "$LIST" .list)          # b_NNN
  DONE_MARK=$OUT/btsv/$BID.done
  TSV=$OUT/btsv/$BID.tsv
  FEAT=$OUT/bfeatures/$BID.features.parquet
  SRCDIR=$OUT/bsrc/$BID

  printf '%s %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$AGENT_ID" "hqfill-A resume $BID / $TOTAL" > "$REPO/.workongoing"

  if [[ -f "$DONE_MARK" ]]; then
    n=$(($(wc -l < "$TSV") - 1))
    done_cells=$((done_cells + n))
    continue
  fi

  rm -rf "$SRCDIR"; mkdir -p "$SRCDIR"
  while IFS= read -r p; do [[ -n "$p" ]] && ln -s "$p" "$SRCDIR/$(basename "$p")"; done < "$LIST"
  nrend=$(ls "$SRCDIR" | wc -l)

  t0=$(date +%s)
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
    >"$OUT/btsv/$BID.stdout" 2>&1 || { log "FAIL $BID (see $BID.stdout)"; tail -5 "$OUT/btsv/$BID.stdout" | tee -a "$PROGRESS"; exit 1; }

  t1=$(date +%s)
  ncells=$(($(wc -l < "$TSV") - 1))
  nfeat=$(python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('$FEAT').num_rows)" 2>/dev/null || echo 0)
  if [[ "$ncells" -ne "$nfeat" ]]; then
    log "GATE-FAIL $BID: cells=$ncells feat_rows=$nfeat (mismatch) — stopping"
    exit 1
  fi
  rm -rf "$SRCDIR"
  touch "$DONE_MARK"
  done_cells=$((done_cells + ncells))
  # heartbeat every 10 batches
  bn=${BID#b_}; bn=$((10#$bn))
  if (( bn % 10 == 0 )); then
    log "PROGRESS $BID: +$ncells cells (${nfeat} feat), $((t1-t0))s; $done_cells remaining-cells done"
  fi
done

log "RESUME ALL BATCHES DONE: $done_cells remaining-cells across $TOTAL batches"
