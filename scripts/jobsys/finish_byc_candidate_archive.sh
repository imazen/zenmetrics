#!/usr/bin/env bash
# Caller owns the existing capped scope and shared lock. No claiming or polling.
set -euo pipefail
candidate_fifo=${1:?closed-plan FIFO}
candidate_output=${2:?fresh snapshot directory}
candidate_mirror=${3:?canonical dataset mirror script}
candidate_destination=${4:?fresh mounted NAS destination}
candidate_snapshot=${5:?snapshot script}
printf '%s\n' 'candidate archive: waiting for the verified closed file plan'
IFS= read -r candidate_plan < "$candidate_fifo"
python3 "$candidate_snapshot" --plan "$candidate_plan" --out "$candidate_output"
python3 "$candidate_mirror" --root "$candidate_output" --nfs \
  --destination "$candidate_destination" --artifact-kind training --cold-readback
