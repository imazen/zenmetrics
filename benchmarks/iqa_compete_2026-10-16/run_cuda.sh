#!/usr/bin/env bash
# Controller runs outside the heavy lock; run.py admits one bounded cell at a time.
set -euo pipefail
binary=$1
pairs=$2
output=$3
models=$4
model_map=$5
sizes=${6:-cid22,2k}
mode=${7:-nn}
reference=$8
qualified=$9
extra=(--batch-seconds "${IQA_COMPETE_BATCH_SECONDS:-1100}")
if [ "$mode" = nn ]; then
    site=$(printf '%s\n' "$HOME"/tmp/nnrt-research/venv-cu/lib/python*/site-packages)
    ort=("$site"/onnxruntime/capi/libonnxruntime.so.*)
    [ "${#ort[@]}" = 1 ] && [ -f "${ort[0]}" ]
    libraries=$(printf '%s:' "$site"/nvidia/*/lib)
    extra+=(--engine ort-cuda --ort-library "${ort[0]}" --cuda-libraries "${libraries%:}")
else
    extra+=(--gpu-runtime "$mode")
fi
python3 benchmarks/iqa_compete_2026-10-16/run.py "$binary" "$pairs" "$output" \
    --host i265 --threads 1,20 --sizes "$sizes" --model-root "$models" --models-json "$model_map" \
    --reference-jsonl "$reference" --reference-jsonl "$qualified" "${extra[@]}"
