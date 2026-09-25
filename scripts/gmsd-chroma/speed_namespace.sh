#!/usr/bin/env bash
# Call through heavy + record_command.py. The UTS namespace prevents zenbench's
# lock metadata from recording the host's private name. Host UTS is unchanged.
set -euo pipefail
root=/var/tmp/gmsd-chroma
if [[ "${1:-}" == --inside ]]; then
    test "$(id -u)" -eq 0
    hostname gmsd-chroma-benchmark
    shift
    lane_uid=$1
    lane_gid=$2
    shift 2
    exec setpriv --reuid="$lane_uid" --regid="$lane_gid" --init-groups \
        env TMPDIR="$root/tmp" XDG_CACHE_HOME="$root/cache" \
        CARGO_TARGET_DIR="$root/target" CARGO_HOME="$root/cache/cargo" \
        UV_CACHE_DIR="$root/cache/uv" ZENBENCH_NO_SAVE=1 \
        PYTHONDONTWRITEBYTECODE=1 "$@"
fi
python3 scripts/gmsd-chroma/preflight.py
test "$#" -gt 0
lane_script=$(realpath "$0")
exec sudo -n unshare --uts --fork -- bash "$lane_script" --inside \
    "$(id -u)" "$(id -g)" "$@"
