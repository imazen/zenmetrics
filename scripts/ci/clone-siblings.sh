#!/usr/bin/env bash
# clone-siblings.sh — clone every sibling repository listed in ci/sibling-pins.tsv at
# its pinned revision, next to a zenmetrics workspace root.
#
# One implementation for CI (every ci.yml job's clone step) and for scripts/ci/lock.sh's
# isolated snapshot, so the two cannot drift. Idempotent: an existing clone is reused,
# fetched only if the pinned revision is missing, and force-checked-out at the pin.
#
# USAGE
#   scripts/ci/clone-siblings.sh                 # workspace root = the repo this script is in
#   scripts/ci/clone-siblings.sh --root DIR      # workspace root = DIR (siblings land at DIR/<dest>)
#   scripts/ci/clone-siblings.sh --pins FILE     # alternate pins file
# Env: CLONE_FILTER=blob:none  make first-time clones partial (lock.sh sets this; CI does not).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
pins=""
while [ $# -gt 0 ]; do
  case "$1" in
    --root) root="$(cd "$2" && pwd)"; shift 2 ;;
    --pins) pins="$2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "clone-siblings: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
[ -n "$pins" ] || pins="$root/ci/sibling-pins.tsv"
[ -r "$pins" ] || { echo "clone-siblings: cannot read $pins" >&2; exit 2; }

n=0
while IFS=$'\t' read -r name url dest rev flags; do
  case "$name" in ''|\#*) continue ;; esac
  [ -n "${url:-}" ] && [ -n "${dest:-}" ] && [ -n "${rev:-}" ] || { echo "clone-siblings: malformed row for '$name' in $pins" >&2; exit 2; }
  flags="${flags:--}"
  parent="$(cd "$root" && mkdir -p "$(dirname "$dest")" && cd "$(dirname "$dest")" && pwd)"
  target="$parent/$(basename "$dest")"
  co=(git -C "$target")
  case ",$flags," in *,sparse-no-benchmarks,*) co+=(-c core.protectNTFS=false) ;; esac

  if [ ! -d "$target/.git" ]; then
    filt=(); [ -n "${CLONE_FILTER:-}" ] && filt=(--filter="$CLONE_FILTER")
    # ${arr[@]+...} keeps an empty array legal under `set -u` on bash 3.2 (macOS /bin/bash).
    git clone --quiet --no-checkout ${filt[@]+"${filt[@]}"} "$url" "$target"
  fi
  if ! git -C "$target" cat-file -e "$rev^{commit}" 2>/dev/null; then
    git -C "$target" fetch --quiet origin
    git -C "$target" cat-file -e "$rev^{commit}" 2>/dev/null \
      || { echo "clone-siblings: $name: revision $rev is not reachable from any branch of $url" >&2; exit 1; }
  fi
  case ",$flags," in
    *,sparse-no-benchmarks,*)
      # jxl-encoder commits sweep PNGs under benchmarks/ whose names are invalid on NTFS;
      # nothing in the build lives there, so skip them (smaller, faster, works on Windows).
      git -C "$target" sparse-checkout init --no-cone
      git -C "$target" sparse-checkout set '/*' '!/benchmarks/'
      ;;
  esac
  "${co[@]}" checkout --quiet --force "$rev"
  n=$((n + 1))
  echo "clone-siblings: $name -> ${rev:0:8}"
done < "$pins"
echo "clone-siblings: $n sibling(s) at their pinned revisions."
