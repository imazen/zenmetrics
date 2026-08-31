#!/usr/bin/env bash
# Task #163 — encoder-search-loop size sweep (linear f32 planar).
# Usage: scripts/loopbench_sweep.sh <bin> <out_tsv> <progress_log> <sizes...>
set -uo pipefail
BIN="${1:?bin}"; OUT="${2:?out}"; LOG="${3:?log}"; shift 3
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
note() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
note "START bin=$BIN out=$OUT sizes=$* load=$(uptime | sed 's/.*averages: //')"

# Record the RESOLVED version of every crate under measurement, next to the
# results. Added 2026-08-30 after the first pass published fast-ssim2 numbers
# without recording which fast-ssim2 they came from — the workspace carries TWO
# (a [patch.crates-io] local 0.8.2 for anything wanting ^0.8, and a registry
# 0.7.1 that jxl-encoder's ^0.7.1 requirement pins), and a manifest comment
# claimed there was only one. A declared dep version is NOT the resolved one;
# only `cargo tree` is authoritative. This is cheap and it is the check that
# would have caught it.
PROV="${OUT%.tsv}.versions"
{
  printf '# resolved dependency versions for %s\n' "$(basename "$OUT")"
  printf '# generated %s on %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -s)"
  printf '# source: cargo tree -p cpu-profile -i <crate> (RESOLVED, not the manifest requirement)\n'
  printf '# repo_commit: %s\n' "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  for c in fast-ssim2 butteraugli zensim imgref zenbench; do
    (cd "$ROOT" && nice -n 19 cargo tree -p cpu-profile -i "$c" -e normal 2>/dev/null | head -1) \
      | sed "s/^/${c}\t/"
  done
} > "$PROV"
note "wrote resolved-version provenance -> $PROV"
cat "$PROV" | sed 's/^/  /' | tee -a "$LOG"
for s in "$@"; do
  note "cell size=$s begin"
  printf '%s claude-metricloop-agent %s sweep cell size=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(basename "$BIN")" "$s" > "$ROOT/.workongoing"
  CPU_WALL_NO_GATE=1 LOOP_WALL_NO_GATE=1 nice -n 19 "$BIN" "$s" "$OUT" >>"$LOG" 2>&1
  note "cell size=$s rc=$? rows=$(wc -l < "$OUT" 2>/dev/null || echo 0)"
done
note "DONE rows=$(wc -l < "$OUT")"
