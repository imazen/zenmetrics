#!/usr/bin/env bash
# Run the existing FitCell executor natively with its canonical paths in a private
# mount namespace. No container runtime or changes to the host's data aliases.
# The caller owns heavy admission, memory caps, job manifests and zenfleet claims.
set -euo pipefail
if [[ "${1:-}" == --inside ]]; then
  shift
  native_ns=$1 native_program=$2 native_scratch=$3 native_uid=$4 native_gid=$5 native_home=$6
  shift 6
  mount --bind "$native_ns/opt" /opt
  mount --bind "$native_program" /opt/fleet-fits/program
  mount --bind "$native_ns/var-tmp" /var/tmp
  mount --bind "$native_ns/byconstruction" /mnt/v/output/zensim/byconstruction-2026-10-22
  mount --bind "$native_scratch" /scratch
  exec setpriv --reuid "$native_uid" --regid "$native_gid" --init-groups -- env HOME="$native_home" PATH="$native_home/.local/bin:/usr/local/bin:/usr/bin:/bin" "$@"
fi
native_ns=$(realpath "${1:?namespace directory}")
native_program=$(realpath "${2:?verified program directory}")
native_scratch=$(realpath "${3:?disk scratch directory}")
shift 3
[[ -x "$native_program/fit_cell_exec.py" && -d "$native_scratch" && $# -gt 0 ]]
mkdir -p "$native_ns/opt/fleet-fits/program" "$native_ns/var-tmp" "$native_ns/byconstruction"
# A mountpoint only; no data is stored in the host's root filesystem.
sudo -n mkdir -p /scratch
exec sudo -n unshare --mount --propagation private \
  bash "$(realpath "$0")" --inside "$native_ns" "$native_program" "$native_scratch" \
  "$(id -u)" "$(id -g)" "$HOME" "$@"
