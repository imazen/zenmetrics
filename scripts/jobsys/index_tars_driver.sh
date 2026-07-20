#!/usr/bin/env bash
# Runs ON the Hetzner index box. Streams every per-box variants tar of the 6 tar-only codecs ONCE and
# declares a byte-range ScoreFile run per tar (bf-<short>-t<N>). ~3 concurrent streams. Idempotent:
# skips a run whose manifest.json already exists in R2. Progress -> /root/index_driver.log.
set -uo pipefail
set -a; . /root/idxenv.sh; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
LOG=/root/index_driver.log; : > "$LOG"
r2(){ s5cmd --endpoint-url "$EP" "$@"; }
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
# codec_label short_run_tag sweep_prefix
CODECS=(
  "zenavif   zavif  mandfix4-zenavif-1782593621"
  "zenjxl    zjxll  jxl-lossy-vardct-1782609551"
  "zenwebp   zwebp  mandfix2-zenwebp-1782584881"
  "zenjxl    zjxlm  jxl-modular-1782596759"
  "zenpng    zpng   mandfix2-zenpng-1782584881"
)
PAR="${PAR:-3}"; running=0
index_one(){ # codec run tar_uri
  local codec="$1" run="$2" tar="$3"
  if r2 ls "s3://zentrain/jobs/$run/manifest.json" >/dev/null 2>&1; then log "SKIP $run (exists)"; return; fi
  log "INDEX $run <- $(basename "$tar")"
  if python3 /root/index_and_declare.py "$tar" "$codec" "$run" zentrain >>"$LOG" 2>&1; then
    log "DONE  $run"
  else
    log "FAIL  $run (see log)"
  fi
}
for row in "${CODECS[@]}"; do
  read -r codec tag sweep <<<"$row"
  mapfile -t tars < <(r2 ls "s3://zentrain/jxl-lossy/runs/$sweep/variants/" 2>/dev/null | awk '/\.tar/{print $NF}')
  for t in "${tars[@]}"; do
    tar="s3://zentrain/jxl-lossy/runs/$sweep/variants/$t"
    # CRITICAL: name the run by the tar's BOX NUMBER, not the loop index. s5cmd ls returns tars in
    # LEXICAL order (box-0,box-1,box-10,box-11,...,box-2,...), so a loop-index suffix put box-10's index
    # under t2 while the consumer (backfill_overnight_manager tar_of) fetches box-2.tar for t2 — a silent
    # tar/index mismatch that only bit codecs with >=10 tars (zjxll). t<boxnum> keeps both sides numeric.
    boxnum=$(printf '%s' "$t" | grep -oE 'box-[0-9]+' | grep -oE '[0-9]+' | head -1)
    [ -n "$boxnum" ] || { log "SKIP $t (no box number)"; continue; }
    run="bf-${tag}-t${boxnum}"
    index_one "$codec" "$run" "$tar" &
    running=$((running+1))
    if [ "$running" -ge "$PAR" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
  done
done
wait
log "ALL INDEXING DONE"
# emit the run list for the launcher
r2 ls "s3://zentrain/jobs/" 2>/dev/null | awk '/DIR  bf-z(avif|jxll|webp|jxlm|png)-t/{print $NF}' | tr -d '/' > /root/runs.txt
log "runs: $(wc -l </root/runs.txt)"
