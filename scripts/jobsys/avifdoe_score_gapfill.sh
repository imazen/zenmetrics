#!/usr/bin/env bash
# avifdoe_score_gapfill.sh — recurring score declaration for the AVIF-DOE wave.
#
# Reduces each DOE encode run's ledger to a fresh pairs table and (re-)declares
# score jobs on whatever is newly DONE. Canonical commands ONLY
# (zenfleet-ctl pairs / declare-scorefiles / compact) — no hand-rolled claim,
# merge or retry logic. declare-scorefiles is idempotent per cell, so re-running
# is always safe and this coexists with the naive-wave loop rather than fighting it.
#
# ── THE HAZARD THIS SCRIPT EXISTS TO GET RIGHT ────────────────────────────────
# A1/A2/A0R encode the 1024-BUDGET CROP corpus; AG encodes the NATIVE corpus.
# The two corpora share ALL 32 FILENAMES. 19 of the 32 are byte-identical
# passthroughs, but 13 are genuinely different pixels (e.g. 6006: native
# 2320x3408 vs crop 1024x1024). Scoring AG against the crop prefix would
# therefore compare native encodes to crop references and silently produce
# garbage on 13/32 of the corpus. AG gets its own --refs-prefix, and the choice
# is PROVEN, not assumed — see AG_REFS below.
#
# Proof of record (2026-09-02, benchmarks/avif_doe_stageA_2026-09-02.md §2):
#   1. corpus:   sha256 of all 32 files in each prefix -> 19 identical, 13 differ
#   2. manifest: AG declared inputs[0] == NATIVE sha 32/32;
#                A0R/A1/A2 declared inputs[0] == CROP sha 32/32
#   3. runtime:  an AG blob for cropped ref 6006 has ispe 2320x3408 (native),
#                not 1024x1024 (crop)
#
# METRIC SET (USER DIRECTIVE 2026-09-02, "you can skip butteraugli even"):
# ssim2,zensim. job_id = JobId::of(kind, inputs) and `kind` carries the metric
# list, so pre-existing 3-metric jobs keep their own job_ids and stay valid
# (their butteraugli rows are extra data, covering only the ~11k pairs scored
# before the 05:42Z boundary — butteraugli is NOT a corpus-wide column).
#
# FAILS LOUD: every round writes a heartbeat with an error counter; any non-zero
# canonical command prints a ❌ line and bumps it. A dead loop is a stale
# heartbeat mtime; a sick loop is a rising errors_total.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
. scripts/lib/s3env.sh
CTL=./target/release/zenfleet-ctl
SFRUN="${ZEN_DOE_SFRUN:-avifdoe-svt-sf-cpu-20260902}"
CROP_REFS="s3://codec-corpus/avif-doe-1024-2026-09-01/"
AG_REFS="s3://codec-corpus/avif-subsample-2026-09-01/"     # NATIVE — see the hazard note above
# The run -> refs-prefix map. OVERRIDABLE so a later wave reuses this loop
# instead of forking it: space-separated `<full-run-name>=<refs-prefix>`.
# Stage B's B-6 run encodes the NATIVE corpus, so it takes AG_REFS — getting
# this wrong is the silent-garbage hazard documented above, not a typo.
ZEN_DOE_RUNS="${ZEN_DOE_RUNS:-avifdoe-svt-a1-20260901=$CROP_REFS avifdoe-svt-a2-20260901=$CROP_REFS avifdoe-svt-a0r-20260901=$CROP_REFS avifdoe-svt-ag-20260901=$AG_REFS}"
PAIRDIR="${ZEN_DOE_PAIRDIR:-$HOME/tmp/avifdoe_pairs}"
HB="${ZEN_DOE_HEARTBEAT:-$HOME/tmp/avifdoe_score_gapfill.heartbeat}"
SLEEP="${ZEN_DOE_SLEEP:-300}"
mkdir -p "$PAIRDIR"
ROUND=0; ERRORS=0
while true; do
  ROUND=$((ROUND+1)); TS=$(date -u +%Y-%m-%dT%H:%M:%SZ); RERR=0
  echo "[$TS] round $ROUND (errors_total=$ERRORS)"
  PARGS=()
  # run -> refs-prefix. AG is LAST and deliberately separate: same filenames,
  # different pixels. Never fold it into the crop loop.
  for spec in $ZEN_DOE_RUNS; do
    run="${spec%%=*}"; refs="${spec#*=}"
    nb=$(aws --endpoint-url "$EP" s3 ls "s3://zentrain/jobs/${run}/blobs/" --recursive 2>/dev/null | wc -l)
    [ "$nb" -eq 0 ] && { echo "[$TS]   $run: 0 encode blobs, skip"; continue; }
    if $CTL pairs --ledger "s3://zentrain/jobs/${run}/ledger/" \
        --refs-prefix "$refs" \
        --blobs-prefix "s3://zentrain/jobs/${run}/blobs/" \
        --out "$PAIRDIR/${run}_pairs" --endpoint "$EP" 2>&1 | tail -1; then
      PARGS+=(--pairs "$PAIRDIR/${run}_pairs.parquet")
    else
      echo "❌ [$TS] pairs FAILED for $run"; RERR=$((RERR+1))
    fi
  done
  if [ ${#PARGS[@]} -gt 0 ]; then
    $CTL declare-scorefiles "${PARGS[@]}" --run "$SFRUN" --bucket zentrain --endpoint "$EP" \
      --metrics ssim2,zensim --full-uri 2>&1 | tail -1 || { echo "❌ [$TS] declare-scorefiles FAILED"; RERR=$((RERR+1)); }
    $CTL compact --run "$SFRUN" --bucket zentrain --endpoint "$EP" --upload 2>&1 | tail -1 || { echo "❌ [$TS] compact FAILED"; RERR=$((RERR+1)); }
  else
    echo "❌ [$TS] no pairs built this round"; RERR=$((RERR+1))
  fi
  sf=$(aws --endpoint-url "$EP" s3 ls "s3://zentrain/jobs/$SFRUN/blobs/" --recursive 2>/dev/null | wc -l)
  echo "[$TS] $SFRUN score_blobs=$sf"
  ERRORS=$((ERRORS+RERR))
  printf '%s round=%s errors_total=%s errors_this_round=%s score_blobs=%s\n' "$TS" "$ROUND" "$ERRORS" "$RERR" "$sf" > "$HB"
  sleep "$SLEEP"
done
