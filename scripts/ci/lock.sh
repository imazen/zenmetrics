#!/usr/bin/env bash
# lock.sh — check or regenerate Cargo.lock in an ISOLATED snapshot whose sibling repositories
# are checked out at exactly the revisions in ci/sibling-pins.tsv (the same ones CI clones).
#
# WHY THIS EXISTS
# ---------------
# Cargo.lock records every sibling PATH package with its version AND its dependency list.
# A lane that runs cargo in a dev tree resolves against whatever siblings happen to be checked
# out there (dirty, older or newer than CI's pins) and commits a lock that CI's siblings cannot
# reproduce: on 2026-09-25/26 the lock flipped between the zenavif 0.2.0 / jxl-encoder 0.4.0
# state and the 0.1.7 / 0.3.2 state, and every CI job died at dependency resolution.
# So: NEVER regenerate Cargo.lock in a dev tree. Use this script.
#
# USAGE
#   scripts/ci/lock.sh --check              # does the working tree's Cargo.lock resolve UNCHANGED
#                                           #   against the pinned siblings? (exit 0 yes, 1 no)
#   scripts/ci/lock.sh --check --rev REV    # same, for the tree at jj/git revision REV
#                                           #   (safe_push.sh uses this on the outgoing commit)
#   scripts/ci/lock.sh --regen              # re-resolve Cargo.lock against the pinned siblings
#                                           #   and write it back to this tree, then --check it
# Check = `cargo fetch --locked` + `cargo metadata --locked --offline` in the snapshot.
# Neither builds anything. Regen = an unlocked `cargo metadata` there (existing lock entries are
# kept wherever the siblings still satisfy them).
#
# ENV  LOCK_SNAP_DIR   snapshot location (default $HOME/tmp/zenmetrics-lock-snap; never /tmp).
#                      Sibling clones persist there between runs (partial clones, fetched on demand).
# EXIT 0 ok · 1 lock does not match the pinned siblings · 2 bad usage / setup failure
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
snap="${LOCK_SNAP_DIR:-$HOME/tmp/zenmetrics-lock-snap}"
mode=""; rev=""

while [ $# -gt 0 ]; do
  case "$1" in
    --check) mode=check; shift ;;
    --regen) mode=regen; shift ;;
    --rev)   rev="$2"; shift 2 ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *) echo "lock.sh: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
done
[ -n "$mode" ] || { echo "lock.sh: say --check or --regen (see --help)" >&2; exit 2; }
[ "$mode" = regen ] && [ -n "$rev" ] && { echo "lock.sh: --rev is for --check only" >&2; exit 2; }

tree="$snap/work/zenmetrics"
mkdir -p "$snap/work"

export_tree() {
  rm -rf "$tree"; mkdir -p "$tree"
  if [ -n "$rev" ]; then
    # jj hides the git dir in secondary workspaces; `jj git root` finds it.
    local gd id
    gd="$(cd "$repo" && jj git root 2>/dev/null)" || { echo "lock.sh: cannot locate the git store (jj git root)" >&2; exit 2; }
    id="$(cd "$repo" && jj log --no-graph --ignore-working-copy -r "$rev" -T commit_id 2>/dev/null)" \
      || { echo "lock.sh: cannot resolve revision '$rev'" >&2; exit 2; }
    git --git-dir="$gd" archive --format=tar "$id" | tar -x -C "$tree"
  else
    rsync -a --exclude=.git --exclude=.jj --exclude=target --exclude=.workongoing "$repo/" "$tree/"
  fi
  [ -r "$tree/Cargo.lock" ] || { echo "lock.sh: no Cargo.lock in the exported tree" >&2; exit 2; }
  [ -r "$tree/ci/sibling-pins.tsv" ] || { echo "lock.sh: no ci/sibling-pins.tsv in the exported tree" >&2; exit 2; }
}

clone_siblings() {
  CLONE_FILTER=blob:none "$tree/scripts/ci/clone-siblings.sh" --root "$tree" >"$snap/clone.log" 2>&1 \
    || { echo "lock.sh: cloning the pinned siblings failed; see $snap/clone.log:" >&2; tail -5 "$snap/clone.log" >&2; exit 2; }
}

remedy() {
  cat >&2 <<MSG

  This Cargo.lock does not match the siblings pinned in ci/sibling-pins.tsv, so CI would fail
  at dependency resolution. Fix it — do not push it as is:

      scripts/ci/lock.sh --regen      # re-resolves the lock in an isolated snapshot at the pins
      jj describe / jj new            # commit the regenerated Cargo.lock

  Never regenerate Cargo.lock by running cargo in a dev tree (its siblings are not CI's pins).
  If the PINS themselves must move (a new sibling version is needed), edit ci/sibling-pins.tsv
  first, then run --regen. Cargo's complaint is in $snap/cargo.log.
MSG
}

check_snapshot() {
  cd "$tree"
  local n; n=$(grep -cv -e '^#' -e '^$' ci/sibling-pins.tsv || true)
  if ! { cargo fetch --locked && cargo metadata --locked --offline --format-version 1 >/dev/null; } >"$snap/cargo.log" 2>&1; then
    echo "lock.sh --check: FAIL — Cargo.lock does not resolve unchanged against the $n pinned siblings." >&2
    sed 's/\x1b\[[0-9;]*m//g' "$snap/cargo.log" | grep -E '^(error|Caused by|help)' | head -6 >&2 || true
    remedy
    return 1
  fi
  echo "lock.sh --check: PASS — Cargo.lock resolves unchanged against the $n pinned siblings (${rev:-working tree})."
}

case "$mode" in
  check)
    export_tree; clone_siblings
    check_snapshot
    ;;
  regen)
    export_tree; clone_siblings
    ( cd "$tree" && cargo metadata --format-version 1 >/dev/null 2>"$snap/cargo.log" ) \
      || { echo "lock.sh --regen: cargo could not resolve against the pinned siblings:" >&2; sed 's/\x1b\[[0-9;]*m//g' "$snap/cargo.log" | grep -E '^(error|Caused by|help)' | head -6 >&2; exit 2; }
    if cmp -s "$tree/Cargo.lock" "$repo/Cargo.lock"; then
      echo "lock.sh --regen: Cargo.lock already matches the pinned siblings; nothing to write."
    else
      cp "$tree/Cargo.lock" "$repo/Cargo.lock"
      echo "lock.sh --regen: wrote a re-resolved Cargo.lock; review with \`jj diff Cargo.lock\`, then commit it."
    fi
    check_snapshot
    ;;
esac
