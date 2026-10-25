#!/usr/bin/env bash
# Runs under the caller's resource admission; consumes only prebuilt binaries.
set -euo pipefail
source_tree=$1
binary_dir=$2
metadata=$3
pins=$4
columns=$5
worker=$6
output=$7
image=$8
control_data=$9
extra_arms=${10}

python3 scripts/jobsys/pack_fit_program.py --profile byc --source "$source_tree" \
  --executor scripts/jobsys/fit_cell_exec.py --bin-dir "$binary_dir" --build-meta "$metadata" \
  --source-pins "$pins" --source-pins-sha256 "$(sha256sum "$pins" | cut -d' ' -f1)" \
  --byc-columns "$columns" --out "$output/program.tar.gz"
FIT_CELL_IMAGE=1 PUSH=1 ZEN_FIT_PROGRAM_TAR="$output/program.tar.gz" \
  ZEN_FIT_PROGRAM_SHA="$(sha256sum "$output/program.tar.gz" | cut -d' ' -f1)" \
  ZEN_WORKER_BIN="$worker" bash scripts/jobsys/build_executor_image.sh "$image"
if [ -n "$control_data" ]; then
  cp --reflink=auto --update=none "$control_data" "$output/data-Mw025.tar.gz"
else
  just byc-fit-data /var/tmp/byc-mixes-20261025/Mw025/root \
    /var/tmp/datalegs-20261025/rev7data/root Mw025 "$output/data-Mw025.tar.gz"
fi
IFS=, read -ra arms <<< "$extra_arms"
for label in "${arms[@]}"; do
  just byc-fit-data "/var/tmp/byc-mixes-20261025/$label/root" \
    /var/tmp/datalegs-20261025/rev7data/root "$label" "$output/data-$label.tar.gz"
done
