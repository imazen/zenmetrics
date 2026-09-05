#!/usr/bin/env bash
# tmpdir_discipline.sh — TMPDIR discipline guard (ban RAM-backed tmp everywhere).
#
# Sourced by fleet-entrypoint.sh (and directly by tests/tmpdir_discipline_test.sh). Defines exactly one
# function, check_tmpdir_discipline, with NO side effects on source — the caller invokes it explicitly.
# Requires the caller to already have `ferr` (print+return-nonzero-worthy error line) and `prog`
# (progress line) defined; fleet-entrypoint.sh's own hb/prog/ferr/stall helpers satisfy this, and the
# shell test defines trivial stand-ins before sourcing this file.
#
# Every launcher (lan_score_launch.sh, enroll_running_node.sh's systemd unit, ...) bind-mounts a
# disk-backed scratch dir and exports TMPDIR to it before starting the worker container/process —
# never the bare, possibly-tmpfs /tmp. This script and jobexec's source-cache
# (jobexec.rs::src_cache_path, via Rust's std::env::temp_dir() which already honours $TMPDIR) both
# write scratch under $TMPDIR, so an unset or RAM-backed TMPDIR means every pass's
# manifest/snapshot/source-cache fetch silently lands in RAM — exactly the 2026-08-06 ENOSPC
# incident's failure mode, just relocated from disk to memory. Fail loud here, at boot, same spirit
# as fleet-entrypoint.sh's baked-tool check: a missing/tmpfs TMPDIR is a launcher bug, not a runtime
# condition to paper over. See docs/RUNNING_JOBS.md "TMPDIR discipline".
check_tmpdir_discipline(){
  if [ -z "${TMPDIR:-}" ]; then
    ferr "TMPDIR is unset — launch this worker with TMPDIR set to a disk-backed scratch dir (e.g. -e TMPDIR=/scratch -v <host-dir>:/scratch); refusing to fall back to a possibly RAM-backed /tmp"
    return 3
  fi
  mkdir -p "$TMPDIR" 2>/dev/null || { ferr "TMPDIR=$TMPDIR could not be created"; return 3; }
  if ! command -v findmnt >/dev/null; then
    ferr "baked tool 'findmnt' missing — cannot verify TMPDIR=$TMPDIR is disk-backed; image is broken, rebuild"
    return 3
  fi
  local fstype
  fstype="$(findmnt -T "$TMPDIR" -no FSTYPE 2>/dev/null)"
  if [ "$fstype" = "tmpfs" ]; then
    ferr "TMPDIR=$TMPDIR resolves to a tmpfs mount (RAM-backed) — bind-mount a real disk path instead (see docs/RUNNING_JOBS.md 'TMPDIR discipline')"
    return 3
  fi
  prog "TMPDIR=$TMPDIR ok (fstype=${fstype:-unknown})"
  return 0
}
