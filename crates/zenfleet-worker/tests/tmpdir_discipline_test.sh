#!/usr/bin/env bash
# tmpdir_discipline_test.sh — failing-first shell test for check_tmpdir_discipline
# (../tmpdir_discipline.sh, sourced by fleet-entrypoint.sh at boot).
#
# Written BEFORE tmpdir_discipline.sh existed: sourcing a missing file, or an empty
# check_tmpdir_discipline that always returns 0, both make this fail — the four cases
# below are exactly the "ban RAM-backed tmp everywhere" contract (docs/RUNNING_JOBS.md
# "TMPDIR discipline"), not a smoke test of "the file is present".
#
# Pure-logic, no cloud, no GPU, no secrets. The only filesystem-flavor check is reading
# /dev/shm's fstype, which is tmpfs on every Linux box/CI runner by default — this test
# never mounts, unmounts, or touches anything outside $DISK_BASE (under this repo's
# target/, removed at the end) and /dev/shm (read-only probe).
#
# Run: bash crates/zenfleet-worker/tests/tmpdir_discipline_test.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SUT="$HERE/../tmpdir_discipline.sh"
[ -r "$SUT" ] || { echo "FAIL - cannot find check_tmpdir_discipline source at $SUT"; exit 1; }

FAILS=0
pass(){ echo "ok   - $1"; }
fail(){ echo "FAIL - $1"; FAILS=$((FAILS + 1)); }

# Stand-ins for fleet-entrypoint.sh's hb/prog/ferr/stall helpers — check_tmpdir_discipline
# calls ferr (on failure) and prog (on success); production versions add timestamps/emoji,
# irrelevant here.
ferr(){ echo "  [ferr] $*" >&2; }
prog(){ echo "  [prog] $*" >&2; }
export -f ferr prog

run_check(){
  # Subshell: sourcing + calling in one isolated process per case so env/PATH
  # mutations (case 4) and `set -u` inside the sourced file can't leak between cases.
  bash -c '. "$1" && check_tmpdir_discipline' _ "$SUT"
}

DISK_BASE="$ROOT/target/tmpdir_discipline_test.$$"
mkdir -p "$DISK_BASE"
trap 'rm -rf "$DISK_BASE"' EXIT

# ── Case 1: TMPDIR unset -> must fail loud (nonzero), not silently default to /tmp. ─────────────────
out="$(env -u TMPDIR bash -c '. "$1" && check_tmpdir_discipline' _ "$SUT" 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -q 'TMPDIR is unset'; then
  pass "unset TMPDIR is rejected (rc=$rc)"
else
  fail "unset TMPDIR should be rejected with a clear message; rc=$rc out=[$out]"
fi

# ── Case 2: TMPDIR resolves to tmpfs (RAM-backed) -> must fail loud. /dev/shm is tmpfs on ────────────
# every Linux box without mounting anything ourselves; skip gracefully (not silently — this
# prints why) only on a platform where that assumption doesn't hold.
if [ "$(findmnt -no FSTYPE /dev/shm 2>/dev/null)" = "tmpfs" ]; then
  out="$(TMPDIR=/dev/shm bash -c '. "$1" && check_tmpdir_discipline' _ "$SUT" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -q 'RAM-backed'; then
    pass "tmpfs-backed TMPDIR (/dev/shm) is rejected (rc=$rc)"
  else
    fail "TMPDIR=/dev/shm (tmpfs) should be rejected; rc=$rc out=[$out]"
  fi
else
  echo "skip - /dev/shm is not tmpfs on this host; case 2 needs a known-tmpfs probe path"
fi

# ── Case 3: TMPDIR is a real disk-backed dir -> must succeed (rc=0), and must NOT print an ──────────
# error. Uses this repo's own target/ (gitignored, always disk-backed in dev/CI).
out="$(TMPDIR="$DISK_BASE/ok" bash -c '. "$1" && check_tmpdir_discipline' _ "$SUT" 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && ! printf '%s' "$out" | grep -q 'ferr\]'; then
  pass "disk-backed TMPDIR ($DISK_BASE/ok) is accepted (rc=$rc)"
else
  fail "disk-backed TMPDIR should be accepted with no error; rc=$rc out=[$out]"
fi
[ -d "$DISK_BASE/ok" ] || fail "check_tmpdir_discipline should mkdir -p its TMPDIR"

# ── Case 4: findmnt missing from PATH -> must fail loud (never silently skip the fstype check). ─────
NOFINDMNT="$DISK_BASE/nopath"; mkdir -p "$NOFINDMNT"
for b in bash env mkdir cat grep; do ln -sf "$(command -v "$b")" "$NOFINDMNT/$b" 2>/dev/null || true; done
out="$(PATH="$NOFINDMNT" TMPDIR="$DISK_BASE/ok2" bash -c '. "$1" && check_tmpdir_discipline' _ "$SUT" 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -qi 'findmnt'; then
  pass "missing findmnt is rejected, not silently skipped (rc=$rc)"
else
  fail "a PATH without findmnt should be rejected by name; rc=$rc out=[$out]"
fi

echo "---"
if [ "$FAILS" -eq 0 ]; then
  echo "PASS: all tmpdir_discipline checks behaved correctly"
  exit 0
else
  echo "FAIL: $FAILS check(s) failed"
  exit 1
fi
