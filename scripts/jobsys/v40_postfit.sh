#!/usr/bin/env bash
# Explicit post-launch harvest and SDR assessment; never schedules a job.
set -euo pipefail
if [[ $# != 2 || ! "$2" =~ ^(control|e29|e32)$ ]]; then
    printf 'usage: %s BUNDLE control|e29|e32\n' "$0" >&2
    exit 2
fi
bundle=$(realpath "$1")
study=$2
jobset="fitv40-$study-20261007"
manifest="$bundle/fit-manifest-$jobset.json"
status="$bundle/postfit-$study.log"
export TMPDIR="${TMPDIR:-$HOME/tmp/v40}"
mkdir -p "$TMPDIR"
# Validate local pins before reading credentials or contacting the ledger.
python3 "$bundle/harvest_driver_v40.py" --bundle "$bundle" "$jobset" "$manifest" --check-config
# The credential file is supplied by the host, outside the source tree.
# shellcheck source=/dev/null
. "$HOME/.config/zen/s3env.sh" >/dev/null 2>&1
exec > >(tee -a "$status") 2>&1
heartbeat() {
    while true; do
        printf 'V40 %s heartbeat %s\n' "$study" "$(date -u +%FT%TZ)"
        sleep 60
    done
}
heartbeat &
heartbeat_pid=$!
trap 'kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true' EXIT
while true; do
    python3 "$bundle/harvest_driver_v40.py" --bundle "$bundle" "$jobset" "$manifest" --install
    if python3 "$bundle/harvest_driver_v40.py" --bundle "$bundle" "$jobset" "$manifest" --install --require-all > "$bundle/postfit-$study.check" 2>&1; then
        cat "$bundle/postfit-$study.check"
        break
    fi
    cat "$bundle/postfit-$study.check"
    if ! rg -q 'all registered cells must be DONE before scoring' "$bundle/postfit-$study.check"; then
        exit 1
    fi
    sleep 60
done
control=/var/tmp/rev4-featpot/v40-control-results
pins="$bundle/V40_CONTROL_PINS.json"
common=(--bundle "$bundle" --control "$control" --tools "$bundle/committed-tools" --control-pins "$pins")
if [[ "$study" == control ]]; then
    python3 "$bundle/score.py" "${common[@]}" --study e29 --freeze-control \
        --results "$control" --root "$bundle/v2e29" --out "$bundle/control-freeze-unused"
    printf '==== V40 CONTROL FROZEN: %s\n' "$pins"
else
    root="$bundle/v2e29"
    [[ "$study" != e32 ]] || root="$bundle/v2e32"
    out="/mnt/v/output/zensim/v40-assessment-$study-2026-10-07"
    python3 "$bundle/score.py" "${common[@]}" --study "$study" \
        --results "/var/tmp/rev4-featpot/v40-$study-results" --root "$root" --out "$out"
    printf '==== V40 %s SDR RESULT: %s\n' "$study" "$out/decision.json"
fi
# HDR/external/UPIQ are separate commands requiring an exact exposure freeze.
