#!/usr/bin/env bash
# Keep-alive fleet-worker entrypoint (goal H). Claims work off the one R2 conditional-write-lease queue
# and runs it until the gap drains (K consecutive passes win nothing) or the box is reclaimed
# (SIGTERM → zenfleet-worker releases its in-flight claim → fast requeue, goal F). No boot-time installs:
# every tool is baked into the image; if one is missing we fail loud (the image is broken, rebuild it).
#
# Env (the launcher / cloud-init injects these):
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN  — SCOPED temp R2 creds (never root)
#   ZEN_R2_ENDPOINT   — https://<acct>.r2.cloudflarestorage.com
#   ZEN_BUCKET        — R2 bucket
#   ZEN_RUN           — run prefix under the bucket (queue namespace)
#   ZEN_MANIFEST_URI  — s3:// URI of the DesiredJob manifest to work
#   ZEN_PROVIDER      — tier label for ledger rows (hetzner/vast/oracle/basement/local/…)
#   ZEN_WORKER        — worker id (default: hostname)
#   ZEN_EXEC          — executor program (default /bin/cat; a real box bakes its encoder/scorer)
#   ZEN_SPEC_THRESHOLD_SECS — optional speculative-execution threshold (goal E)
#   ZEN_CONTROL_KEY   — optional s3 key of a RunControl object for pause/drain (goal C)
#   ZEN_IDLE_PASSES (5) / ZEN_PASS_SLEEP (0.2) — drain detection + pacing
#   ZEN_CLAIM_MODE    — optional: `lease` (default) or `epoch-sharded` (wall-clock epochs +
#                       rendezvous-hash ownership; zero claim traffic in steady state — see
#                       docs/RUNNING_JOBS.md "Epoch-sharded claiming"). Single-run mode only;
#                       ALL of a run's workers must use one mode (or set it once in the run's
#                       RunControl object via ZEN_CONTROL_KEY, which overrides this fleet-wide).
#   ZEN_EPOCH_LEN_SECS (600) / ZEN_EPOCH_HB_SECS (120) — epoch-sharded tuning knobs
set -uo pipefail
: "${ZEN_R2_ENDPOINT:?}" "${ZEN_BUCKET:?}"
# ZEN_RUN/ZEN_MANIFEST_URI are required only for single-run mode; POOL mode (ZEN_POOL_RUNLIST) is self-describing.
[ -n "${ZEN_POOL_RUNLIST:-}" ] || : "${ZEN_RUN:?}" "${ZEN_MANIFEST_URI:?}"
PROVIDER="${ZEN_PROVIDER:-fleet}"
WORKER="${ZEN_WORKER:-$(hostname)}"
EXEC="${ZEN_EXEC:-/bin/cat}"
export AWS_REGION="${AWS_REGION:-auto}" AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-auto}"

# ── Observability: force-surface progress, heartbeats, and problems so a human
# OR an LLM tailing `docker logs` CANNOT miss a stall or a failure. Distinctive,
# greppable, line-anchored markers — grep '❌' for every problem, '⚠' for stalls,
# '♥' for liveness, '▸' for progress. Nothing important is ever hidden behind a
# `tail -1` again (the bug this file had: only the last worker line reached the
# logs, so the mode line + panics + warnings were invisible).
hb(){    echo "♥ [hb       $(date -u +%H:%M:%SZ)] $*"; }
prog(){  echo "▸ [progress $(date -u +%H:%M:%SZ)] $*"; }
ferr(){  echo "❌ [FLEET-ERROR $(date -u +%H:%M:%SZ)] $*"; }
stall(){ echo "⚠ [FLEET-STALL $(date -u +%H:%M:%SZ)] $*"; }
# Anything matching this in a worker/tool output blob is force-surfaced as an error.
FLEET_PROBLEM_RE='panicked|thread .*panicked|FATAL|fatal error|error\[E|(^|[^A-Za-z])[Ee]rror(:| )|OOM|out of memory|Killed|SIGKILL|SIGABRT|core dumped|NaN|failed=[1-9]|poisoned=[1-9]|Traceback|Segmentation'
surface_problems(){ # $1: text blob — re-emit each distinct problem line LOUDLY
  local h; h=$(printf '%s\n' "$1" | grep -aiE "$FLEET_PROBLEM_RE" | sort -u || true)
  [ -z "$h" ] && return 1
  printf '%s\n' "$h" | while IFS= read -r l; do ferr "worker output: $l"; done
  return 0
}

# Every baked tool must exist (bake-everything rule) — including `timeout`, which
# the loop uses to turn a hung pass into a LOUD failure instead of a silent stall.
for tool in aws s5cmd zenfleet-worker gunzip timeout; do
  command -v "$tool" >/dev/null || { ferr "baked tool '$tool' missing — image is broken, rebuild"; exit 3; }
done

# ── GPU-SCORING GUARANTEE ─────────────────────────────────────────────────────────────────────────
# Set ZEN_REQUIRE_GPU=1 on a box whose scores MUST come from a GPU. It exports
# ZENMETRICS_REQUIRE_GPU=1 (inherited by every zenmetrics child), which drops the CPU rung from the
# metric backend's `auto` ladder so a GPU-less or GPU-broken box FAILS instead of quietly computing
# on the CPU and writing the result under a GPU column name — a row that is indistinguishable from a
# real GPU measurement afterwards. The fleet executor cannot pass `--gpu-runtime`, so this env var is
# the only way to make fleet scoring GPU-guaranteed.
#
# We also verify a GPU is actually visible HERE, at boot, rather than letting the box claim cells and
# fail them one by one: a boot failure is one loud line, whereas per-cell failures burn lease time and
# look like flaky data. Same spirit as the baked-tool check above — fail loud, fail early.
# ── CUDA 12-vs-13 selection ────────────────────────────────────────────────────────────────────────
# Runs on EVERY boot, before any scoring, and is a no-op on CPU boxes. CUDA Toolkit 13.0 dropped
# offline compilation for Maxwell/Pascal/Volta (floor is sm_75/Turing), and this fleet straddles that
# line: GTX 1050 / GTX 1060 are sm_61 and can only be served by CUDA 12, newer parts want 13. The
# image bakes BOTH; this picks from the GPU actually present and exports CUDA_PATH/LD_LIBRARY_PATH so
# cudarc's dlopen resolves the right NVRTC. Force with ZEN_CUDA_MAJOR=12|13.
if [ -r /usr/local/bin/cuda-select.sh ]; then
  # shellcheck source=/dev/null
  . /usr/local/bin/cuda-select.sh || ferr "cuda-select failed; continuing with the image default"
  [ -n "${ZEN_CUDA_SELECTED:-}" ] && [ "${ZEN_CUDA_SELECTED}" != none ] \
    && prog "CUDA ${ZEN_CUDA_SELECTED} selected for this box (${CUDA_PATH:-?})"
fi

if [ "${ZEN_REQUIRE_GPU:-0}" = "1" ]; then
  export ZENMETRICS_REQUIRE_GPU=1
  command -v nvidia-smi >/dev/null \
    || { ferr "ZEN_REQUIRE_GPU=1 but nvidia-smi is missing — this box cannot honour a GPU-only scoring job"; exit 3; }
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q . \
    || { ferr "ZEN_REQUIRE_GPU=1 but no GPU is visible to nvidia-smi — refusing to score on CPU under GPU column names"; exit 3; }
  prog "GPU-only scoring enforced (ZENMETRICS_REQUIRE_GPU=1); visible GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
fi

# The manifest can be large (92MB+ for big sweeps) and some fleet boxes choke on large R2 downloads
# even though small ops are fine (observed 2026-06-24: a 92MB cp hung at 0 bytes while ls/control.json
# took <0.4s). Prefer a gzipped manifest (~30x smaller — 92MB -> 3MB) at <uri>.gz when present, else
# fall back to the plain object. Backward-compatible: runs without a .gz still work via the fallback.
# (MGZ is computed AFTER the pool-mode branch below — it dereferences ZEN_MANIFEST_URI, which pool
# mode doesn't set.)
# ── POOL MODE — the hourly-efficient lifecycle ─────────────────────────────────────────────────────
# When ZEN_POOL_RUNLIST is set, this box works the WHOLE undone-tar POOL rather than one run: it
# round-robins every run in the runlist (TSV `run<TAB>tar_uri`), scoring one chunk of each per cycle
# and coordinating with peers via the R2 claim/ledger — so it NEVER drains-then-idles on a single small
# tar. It runs until ZEN_MAX_MIN minutes elapse (default 55 = one full paid hour, never a second) then
# EXITS, which drops through to the cloud-init self-destruct. One box == one useful paid hour, no churn.
pool_mode(){
  local END; END=$(( $(date +%s) + ${ZEN_MAX_MIN:-55} * 60 ))
  s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp "$ZEN_POOL_RUNLIST" /tmp/runlist.raw \
    || { ferr "cannot fetch pool runlist $ZEN_POOL_RUNLIST"; exit 4; }
  # Shuffle per box so a large fleet spreads across runs instead of all piling onto run #1 (which would
  # serialize on that one run's claim-lease). Each box still covers every run over successive cycles.
  shuf /tmp/runlist.raw -o /tmp/runlist.tsv 2>/dev/null || cp /tmp/runlist.raw /tmp/runlist.tsv
  hb "POOL mode: $(grep -c . /tmp/runlist.tsv) runs (shuffled), budget ${ZEN_MAX_MIN:-55}min (worker=$WORKER)"
  local cyc=0
  while [ "$(date +%s)" -lt "$END" ]; do
    cyc=$((cyc + 1)); local did=0 seen=0
    while IFS=$'\t' read -r run src mode _rest <&3; do
      [ -z "$run" ] && continue
      [ "$(date +%s)" -ge "$END" ] && break
      seen=$((seen + 1))
      local mf="/tmp/m_${run}.json"
      if [ ! -s "$mf" ]; then
        s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp "s3://$ZEN_BUCKET/jobs/$run/manifest.json.gz" "$mf.gz" 2>/dev/null \
          && gunzip -f "$mf.gz" 2>/dev/null
      fi
      [ -s "$mf" ] || { stall "pool: no manifest for $run"; continue; }
      # Source the variants per mode: 'enc' = direct-object (individual encodes/, e.g. zenjpeg);
      # anything else = byte-range from the tar via the per-run index.
      local venv
      if [ "${mode:-tar}" = "enc" ]; then
        venv=(ZEN_ENCODES_PREFIX="$src" ZEN_ENCODES_BUCKET="$ZEN_BUCKET")
      else
        venv=(ZEN_VARIANTS_TAR_URI="$src" ZEN_VARIANT_INDEX_URI="s3://$ZEN_BUCKET/jobs/$run/variant_index.tsv")
      fi
      # --ledger-in: feed the worker the done-set snapshot so its gap = only-undone, instead of
      # re-scoring cells whose 10-min claim lease already expired (the ~2x tax on resumed runs —
      # POOL mode otherwise passes no --ledger-in, so the reconcile view is empty and the gap is
      # every cell each pass). Re-fetched per pass so a cron-refreshed snapshot stays current; guarded
      # — a run with no snapshot yet just runs the old way (zero downside). Seed/refresh the snapshots
      # with scripts/jobsys/refresh_snapshots.sh (on the cred-holding box, never baked onto a worker).
      local snap="/tmp/snap_${run}.parquet"
      s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp "s3://$ZEN_BUCKET/jobs/$run/ledger_snapshot.parquet" "$snap" >/dev/null 2>&1 || rm -f "$snap"
      local li=(); [ -s "$snap" ] && li=(--ledger-in "$snap")
      out=$(env "${venv[@]}" \
        timeout "${ZEN_PASS_TIMEOUT:-1800}" zenfleet-worker --manifest "$mf" "${li[@]}" \
        --ledger-out "s3://$ZEN_BUCKET/jobs/$run/ledger/pool-$WORKER-$cyc.parquet" \
        --blobs-r2-bucket "$ZEN_BUCKET" --blobs-r2-prefix "jobs/$run/blobs" \
        --claims-r2-bucket "$ZEN_BUCKET" --claims-prefix "jobs/$run/claims" \
        --r2-endpoint "$ZEN_R2_ENDPOINT" --exec "$EXEC" --worker "$WORKER" --provider "$PROVIDER" 2>&1); local rc=$?
      surface_problems "$out" || true
      local s; s=$(printf '%s\n' "$out" | grep -oE 'done=[0-9]+' | head -1 | cut -d= -f2)
      # "made progress" = scored (done>0) OR the pass timed out (rc=124: still had claimable work, just slow).
      # Only a full cycle of genuinely EMPTY passes (done=0, clean exit) means the pool is drained.
      { [ "${s:-0}" -gt 0 ] || [ "$rc" -eq 124 ]; } && did=1
      prog "pool cyc=$cyc @$run done=${s:-0} ($(( (END - $(date +%s)) / 60 ))min left)"
    done 3< /tmp/runlist.tsv
    if [ "$did" -eq 0 ]; then
      hb "POOL: nothing left across all $seen runs — backfill drained, exiting"
      # Completion beacon: this box did a FULL cycle and found EVERY run empty. pool_cron watches
      # s3://.../jobs/_pool/drained/ (a cheap small-prefix list) to auto-stop, instead of reading every
      # ledger. Filename carries the epoch so the cron can window it to "recent".
      printf '%s drained cyc=%s\n' "$WORKER" "$cyc" > /tmp/drainmark 2>/dev/null || true
      s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp /tmp/drainmark "s3://$ZEN_BUCKET/jobs/_pool/drained/${WORKER}-$(date +%s)" 2>/dev/null || true
      break
    fi
    sleep "${ZEN_PASS_SLEEP:-0.2}"
  done
  prog "POOL exit (budget reached or drained) -> container exits -> self-destruct"
}
if [ -n "${ZEN_POOL_RUNLIST:-}" ]; then pool_mode; exit 0; fi

# Single-run mode below — safe to dereference ZEN_MANIFEST_URI now (pool mode has exited).
MGZ="${ZEN_MANIFEST_URI%.gz}.gz"
hb "fetching manifest ($MGZ, else plain) — this is the op that historically hung at 0 bytes on big sweeps"
mfetch_err=$(s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp "$MGZ" /tmp/manifest.json.gz 2>&1)
if [ $? -eq 0 ]; then
  gunzip -f /tmp/manifest.json.gz || { ferr "gunzip $MGZ failed"; exit 4; }
elif s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp "${ZEN_MANIFEST_URI%.gz}" /tmp/manifest.json 2>/dev/null; then
  :
else
  ferr "cannot fetch manifest ($MGZ or plain ${ZEN_MANIFEST_URI%.gz}); s5cmd said: ${mfetch_err:-<silent>}"; exit 4
fi
hb "manifest ready ($(wc -c </tmp/manifest.json 2>/dev/null || echo '?') bytes); $WORKER ($PROVIDER) claiming from s3://$ZEN_BUCKET/$ZEN_RUN/"

idle=0 i=0 fails=0
while [ "$idle" -lt "${ZEN_IDLE_PASSES:-5}" ]; do
  i=$((i + 1))
  # Scratch hygiene BETWEEN passes (2026-08-06 ENOSPC incident, hdrgrid encode
  # wave): the executor's warm-process source cache (`jobexec_src_<pid>_*`) is
  # keyed by pid and never evicted, so every executor generation re-downloads
  # the sources it touches and the container writable layer grows without
  # bound — measured ~620 GB across two operator-box containers (~15 MB
  # retained/cell), which filled the root disk and ENOSPC-killed the wave at
  # 83%. Between passes no executor is alive (they are children of the pass's
  # worker process), so deleting the whole scratch set is race-free; a warm
  # process that later misses its cache simply re-downloads. Variant/dist
  # temps are same-class leaks on the scoring path. Long single passes still
  # need an operator-side TTL cleaner on a bind-mounted /tmp (the incident's
  # 620 GB grew INSIDE passes); this sweep bounds the between-pass baseline.
  rm -f /tmp/jobexec_src_* /tmp/jobexec_var_* /tmp/jobexec_dist_* 2>/dev/null || true
  # --ledger-in done-set snapshot (same contract as POOL mode, which has always fetched it):
  # without it the reconcile view is EMPTY, the gap is ALL cells every pass, and dedup falls to
  # the claim lease — the documented 2.0-3.4x re-work tax (compact_ledgers.py docstring), which
  # single-run workers paid until 2026-08-06 (avifgen encode run, measured rate collapse in the
  # tail). Re-fetched EVERY pass so long-lived workers pick up operator-side refreshes. Absent
  # snapshot (fresh run, nobody compacted yet) = old behaviour.
  SNAP=/tmp/ledger_snapshot.parquet
  s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp "s3://$ZEN_BUCKET/$ZEN_RUN/ledger_snapshot.parquet" "$SNAP" >/dev/null 2>&1 || rm -f "$SNAP"
  LIN=(); [ -s "$SNAP" ] && LIN=(--ledger-in "$SNAP")
  # HEARTBEAT emitted BEFORE the blocking call: if the worker hangs, this line has
  # no matching '▸ progress pass N' — a visible stall, not silence. `timeout` turns
  # a genuine hang into a LOUD rc=124 failure instead of an infinite silent block.
  hb "pass $i start (worker=$WORKER provider=$PROVIDER idle=$idle/${ZEN_IDLE_PASSES:-5} consec_fails=$fails snap=$([ -s "$SNAP" ] && wc -c <"$SNAP" || echo none))"
  out=$(timeout "${ZEN_PASS_TIMEOUT:-1800}" zenfleet-worker --manifest /tmp/manifest.json "${LIN[@]}" \
    --ledger-out "s3://$ZEN_BUCKET/$ZEN_RUN/ledger/pass-$WORKER-$i.parquet" \
    --blobs-r2-bucket "$ZEN_BUCKET" --blobs-r2-prefix "$ZEN_RUN/blobs" \
    --claims-r2-bucket "$ZEN_BUCKET" --claims-prefix "$ZEN_RUN/claims" \
    ${ZEN_SPEC_THRESHOLD_SECS:+--spec-threshold-secs "$ZEN_SPEC_THRESHOLD_SECS"} \
    ${ZEN_CONTROL_KEY:+--control-r2-key "$ZEN_CONTROL_KEY"} \
    ${ZEN_CAPABILITY:+$(for c in ${ZEN_CAPABILITY//,/ }; do printf -- '--capability %s ' "$c"; done)} \
    ${ZEN_CLAIM_MODE:+--claim-mode "$ZEN_CLAIM_MODE"} \
    ${ZEN_EPOCH_LEN_SECS:+--epoch-len-secs "$ZEN_EPOCH_LEN_SECS"} \
    ${ZEN_EPOCH_HB_SECS:+--epoch-heartbeat-secs "$ZEN_EPOCH_HB_SECS"} \
    --r2-endpoint "$ZEN_R2_ENDPOINT" --exec "$EXEC" --worker "$WORKER" --provider "$PROVIDER" 2>&1)
  rc=$?

  # Pass 1: surface the FULL worker output ONCE (startup warnings, run-control state,
  # etc.) so the box's real first pass is in the logs — not tail -1'd away.
  [ "$i" -eq 1 ] && printf '%s\n' "$out" | sed 's/^/  worker| /'
  # Surface the worker's mode/budget line the FIRST time it appears. Pass 1 is often
  # PAUSED (run control) and never prints it — the mode line only shows on the first
  # UNPAUSED pass — so don't gate this on pass number, or it's lost like it used to be.
  if [ -z "${mode_shown:-}" ] && printf '%s\n' "$out" | grep -qaE 'resource-aware concurrent mode|serial per-cell|epoch-sharded epoch'; then
    prog "worker mode :: $(printf '%s\n' "$out" | grep -aE 'resource-aware concurrent mode|serial per-cell|epoch-sharded epoch' | head -1)"
    mode_shown=1
  fi

  # FAILURE force-surface: a hang (timeout=124) or any non-zero exit is LOUD, with a tail.
  if [ "$rc" -eq 124 ]; then
    ferr "pass $i TIMED OUT after ${ZEN_PASS_TIMEOUT:-1800}s — worker hung. tail:"
    printf '%s\n' "$out" | tail -15 | sed 's/^/  worker| /'
    fails=$((fails + 1))
  elif [ "$rc" -ne 0 ]; then
    ferr "pass $i worker exited rc=$rc. tail:"
    printf '%s\n' "$out" | tail -15 | sed 's/^/  worker| /'
    fails=$((fails + 1))
  else
    fails=0
  fi

  # PROBLEM force-surface even on rc=0 (panics a wrapper swallowed, failed/poisoned cells, NaN scores).
  surface_problems "$out" || true

  # PROGRESS every pass: the worker's own summary line (done/failed/poisoned/skipped/rows).
  summary=$(printf '%s\n' "$out" | grep -E 'zenfleet-worker: done=' | tail -1)
  prog "pass $i rc=$rc :: ${summary:-<no summary line — see worker| output above>}"

  # FAIL-FAST: N consecutive worker-level bad passes (crash/hang — NOT individual failed cells, which
  # are counted in the rc=0 summary) ⇒ exit non-zero + LOUD, so the box's onstart/error-trap (where the
  # image wires one, e.g. run_with_error_trap.sh) self-destroys, or the startup watchdog / operator
  # catches it — instead of a silent crash-loop that burns $/hr making zero progress.
  if [ "$fails" -ge "${ZEN_MAX_FAILS:-3}" ]; then
    ferr "$fails consecutive failed passes on $WORKER — exiting non-zero (fail-fast) instead of crash-looping"
    exit 5
  fi

  # Drain / pause bookkeeping (unchanged semantics; now announced).
  if printf '%s\n' "$out" | grep -qiE 'run control ='; then
    hb "run control active (paused/draining) — holding, not counting toward drain-exit"
    idle=0
  elif printf '%s\n' "$out" | grep -qE 'done=0 '; then
    idle=$((idle + 1))
  else
    idle=0
  fi
  sleep "${ZEN_PASS_SLEEP:-0.2}"
done
prog "$WORKER drained (idle $idle consecutive no-work passes) — exiting clean"
