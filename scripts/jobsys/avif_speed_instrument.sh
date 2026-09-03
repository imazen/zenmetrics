#!/usr/bin/env bash
# AVIF speed instrument — the tuning model's third axis (encode TIME), which no
# fleet path persists (plan doc `avif_doe_plan_2026-09-01.md` section 3.6: jobexec
# emits `encode_ms` only on the `metric` job kind, and the DOE runs `encode` +
# `score_file`).  `zenmetrics sweep` DOES persist `encode_ms`
# (crates/zenmetrics-cli/src/sweep/run.rs:1219), so it is the owner and this
# script is a protocol driver around it -- no new timing code.
#
# PROTOCOL (perf discipline, zensim CLAUDE.md "PERF MEASUREMENT"):
#   * ONE binary, runtime arms only (backend + speed + q are all knob-grid values
#     of the same process), so a build-layout shift cannot masquerade as an arm.
#   * Arms INTERLEAVED inside a pass: both backends live in one `--knob-grid`, so
#     the sweep's own Cartesian walk alternates them per image.
#   * min-of-N over N separate PROCESS STARTS with ASLR left on (layout is an
#     input; see zensim benchmarks/era2_perf_break_2026-08-31.md section 22.5).
#   * `--jobs 1`, uncontended dedicated host, nothing else running.
#   * `--no-score`: ENCODE-ONLY.  Scoring between timed encodes is neither free
#     nor neutral -- a multi-threaded perceptual metric on every core between
#     two single-threaded encodes leaves a different boost/thermal state than
#     the encode it precedes.  MEASURED on this instrument's first launch
#     (r7900x, 2026-09-03): 23 min of wall clock carried 15.0 s of `encode_ms`,
#     i.e. scoring + per-cell overhead was ~99% of the run.  Quality for the
#     matched-quality backend comparison comes from the RD wave, not here.
#   * NOT niced: nice/ionice only bite under contention, and on an idle dedicated
#     box they add scheduling artifacts to the thing being measured. r7900x is a
#     dedicated LAN worker (not the shared dev box or the household tower).
#
# Usage: avif_speed_instrument.sh <ladder-dir> <qprobe-dir> <out-dir> [passes]
set -euo pipefail

LADDER="${1:?ladder source dir}"
QPROBE="${2:?q-probe source dir}"
OUT="${3:?output dir}"
PASSES="${4:-3}"
BIN="${ZM_BIN:-$HOME/speedinstr/bin/zenmetrics}"

SPEEDS='1,2,3,4,5,6,7,8,9,10'
BACKENDS='"svt-rs","zenravif"'
GRID="{\"backend\":[${BACKENDS}],\"speed\":[${SPEEDS}]}"

mkdir -p "$OUT"
LOG="$OUT/instrument.log"

hb() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

hb "START host=$(hostname) nproc=$(nproc) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA) passes=$PASSES"
hb "bin=$BIN sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
hb "ladder=$(ls "$LADDER" | wc -l) inputs; qprobe=$(ls "$QPROBE" | wc -l) inputs"

for p in $(seq 1 "$PASSES"); do
  # S1a -- the alpha + beta*pixels ladder, single q (time's q-dependence is S1b's
  # question, not this block's; keeping q at 1 point here is what makes the full
  # 7-rung ladder affordable on the slow backend).
  hb "pass $p S1a start (ladder x 10 speeds x 2 backends x q45)"
  "$BIN" sweep --codec zenavif --sources "$LADDER" --q-grid 45 \
      --knob-grid "$GRID" --jobs 1 \
      --no-score \
      --output "$OUT/s1a_pass${p}.tsv" >>"$LOG" 2>&1
  hb "pass $p S1a done rows=$(( $(wc -l < "$OUT/s1a_pass${p}.tsv") - 1 ))"

  # S1b -- the q-flatness probe.  The plan asserts per-q encode cost is flat
  # (retrofit section 9.2); the brief says VERIFY rather than inherit it.
  hb "pass $p S1b start (3 sizes x 10 speeds x 2 backends x q{15,45,90})"
  "$BIN" sweep --codec zenavif --sources "$QPROBE" --q-grid 15,45,90 \
      --knob-grid "$GRID" --jobs 1 \
      --no-score \
      --output "$OUT/s1b_pass${p}.tsv" >>"$LOG" 2>&1
  hb "pass $p S1b done rows=$(( $(wc -l < "$OUT/s1b_pass${p}.tsv") - 1 ))"
done

hb "COMPLETE passes=$PASSES"
touch "$OUT/COMPLETE"
