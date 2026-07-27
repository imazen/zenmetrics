#!/usr/bin/env bash
# run.sh -- the Zen worker pool loop for a macOS node (the mac port of fleet-entrypoint.sh pool_mode).
#
# Works the SAME R2 job pool the cloud/tower boxes use: shuffled round-robin over the runlist, one
# `zenfleet-worker` pass per run, coordinating via the R2 claim ledger. Runs the NATIVE darwin
# binaries (no Docker) at low priority (`nice`), looping until the pool drains OR launchd stops it.
#
# Layout expected (all in the same folder as this script, i.e. the payload dir):
#   zenmetrics  zenfleet-worker  s5cmd  aws   run.sh  worker.env
# `zenmetrics jobexec` is the executor; `zenfleet-worker` is the claim/ledger loop; s5cmd+aws do R2 IO.
#
# macOS note: BSD `timeout` does not exist -- we use `gtimeout` from coreutils (`brew install coreutils`)
# and fall back to no-timeout if absent (with a warning). Everything else is POSIX.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/worker.log"
log(){ printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

# --- config from worker.env (KEY=VALUE) ---
[ -f "$HERE/worker.env" ] && set -a && . "$HERE/worker.env" && set +a
# Bundled binaries first; then brew (arm64 + Intel) — launchd's PATH is /usr/bin:/bin:/usr/sbin:/sbin
# and the claim CAS path (`claim_or_steal_r2_key`) SPAWNS the `aws` CLI, which lives in brew. Without
# this, every claim spawn fails instantly and the worker skips all cells (found 2026-07-27:
# lilith-mac cycled the whole bf924 pool at done=0 while login-shell runs worked).
export PATH="$HERE:/opt/homebrew/bin:/usr/local/bin:$PATH"
BUCKET="${ZEN_BUCKET:-zentrain}"
EP="${ZEN_R2_ENDPOINT:?worker.env must set ZEN_R2_ENDPOINT}"
RUNLIST="${ZEN_POOL_RUNLIST:?worker.env must set ZEN_POOL_RUNLIST}"
WORKER="${ZEN_WORKER:-macpool-$(hostname -s)}"
PROVIDER="${ZEN_PROVIDER:-mac}"
export AWS_REGION="${AWS_REGION:-auto}" AWS_DEFAULT_REGION="${AWS_REGION:-auto}"

# --- required tools baked in / installed; fail loud if missing (no boot-time installs) ---
for t in zenfleet-worker zenmetrics s5cmd aws; do
  command -v "$t" >/dev/null 2>&1 || { log "FATAL: missing $t on PATH ($HERE)"; exit 3; }
done
TIMEOUT="gtimeout"; command -v gtimeout >/dev/null 2>&1 || { TIMEOUT=""; log "WARN: gtimeout absent (brew install coreutils) -- passes run without a wall-clock cap"; }
EXEC="$HERE/zen-exec.sh"                          # tiny shim -> zenmetrics jobexec (written below)
cat > "$EXEC" <<EOF
#!/usr/bin/env bash
exec "$HERE/zenmetrics" jobexec "\$@"
EOF
chmod +x "$EXEC"

# --- CPU niceness: cap threads so the desktop stays responsive ---
: "${ZEN_THREADS:=$(( $(sysctl -n hw.logicalcpu 2>/dev/null || echo 4) / 2 ))}"
[ "$ZEN_THREADS" -ge 1 ] 2>/dev/null || ZEN_THREADS=1
export RAYON_NUM_THREADS="$ZEN_THREADS" OMP_NUM_THREADS="$ZEN_THREADS"
export ZEN_PERSISTENT_EXEC="${ZEN_PERSISTENT_EXEC:-1}"

# --- fetch + shuffle the runlist (spread across runs, not all on run #1) ---
RL="$(mktemp -t zen_runlist)"
s5cmd --endpoint-url "$EP" cp "$RUNLIST" "$RL" >/dev/null 2>&1 || { log "FATAL: cannot fetch runlist $RUNLIST"; exit 4; }
# shuffle
RUNS="$(grep -c . "$RL")"
log "POOL start: $RUNS runs, worker=$WORKER, threads=$ZEN_THREADS, provider=$PROVIDER"

cyc=0
while :; do
  cyc=$((cyc+1)); didany=0
  # shuffle order each cycle
  while IFS=$'\t' read -r run src mode; do
    [ -n "${run:-}" ] || continue
    mode="${mode:-tar}"
    mf="$(mktemp -t "m_$run")"
    if s5cmd --endpoint-url "$EP" cp "s3://$BUCKET/jobs/$run/manifest.json.gz" "$mf.gz" >/dev/null 2>&1; then
      gunzip -f "$mf.gz" 2>/dev/null && mv "$mf" "$mf" 2>/dev/null || { [ -f "$mf.gz" ] && gzcat "$mf.gz" > "$mf"; }
    fi
    [ -s "$mf" ] || { rm -f "$mf" "$mf.gz"; continue; }
    # per-mode source env (mirrors fleet-entrypoint.sh / run.ps1)
    if [ "$mode" = "enc" ]; then
      export ZEN_ENCODES_PREFIX="$src" ZEN_ENCODES_BUCKET="$BUCKET"
      unset ZEN_VARIANTS_TAR_URI ZEN_VARIANT_INDEX_URI
    else
      export ZEN_VARIANTS_TAR_URI="$src" ZEN_VARIANT_INDEX_URI="s3://$BUCKET/jobs/$run/variant_index.tsv"
      unset ZEN_ENCODES_PREFIX ZEN_ENCODES_BUCKET
    fi
    out="$( ${TIMEOUT:+$TIMEOUT ${ZEN_PASS_TIMEOUT:-1800}} zenfleet-worker --manifest "$mf" \
        --ledger-out "s3://$BUCKET/jobs/$run/ledger/mac-$WORKER-$cyc.parquet" \
        --blobs-r2-bucket "$BUCKET" --blobs-r2-prefix "jobs/$run/blobs" \
        --claims-r2-bucket "$BUCKET" --claims-prefix "jobs/$run/claims" \
        --r2-endpoint "$EP" --exec "$EXEC" --worker "$WORKER" --provider "$PROVIDER" 2>&1)"
    done_n="$(printf '%s' "$out" | sed -n 's/.*done=\([0-9]\{1,\}\).*/\1/p' | tail -1)"
    [ "${done_n:-0}" -gt 0 ] 2>/dev/null && didany=1
    log "cyc=$cyc run=$run mode=$mode done=${done_n:-0}"
    rm -f "$mf" "$mf.gz"
  done < <(sort -R "$RL")
  if [ "$didany" -eq 0 ]; then
    log "POOL: whole pool drained (no work in a full cycle) -- exiting"
    printf 'drained %s\n' "$(date -u +%FT%TZ)" > /tmp/zen_drainmark
    s5cmd --endpoint-url "$EP" cp /tmp/zen_drainmark "s3://$BUCKET/jobs/_pool/drained/${WORKER}-$(date +%s)" >/dev/null 2>&1 || true
    break
  fi
  sleep 0.3
done
log "POOL exit"
