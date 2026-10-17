#!/usr/bin/env bash
# run_afine.sh <variant>...: one process per variant, 4 threads each, in parallel.
cd ~/tmp/aic-confsearch
export OMP_NUM_THREADS=${T:-4}
for v in "$@"; do
  [ -s out/afine_$v.tsv ] && [ $(wc -l < out/afine_$v.tsv) -ge 62 ] && continue
  rm -f out/afine_$v.tsv
  venv/bin/python scripts/afine_probe.py ext/afine_up/QuickInference ext/afine_ckpt pairs_dev.tsv out/afine_$v.tsv $v > out/afine_$v.log 2>&1 &
done
wait
