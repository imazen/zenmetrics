#!/usr/bin/env bash
# run_cfgs.sh <cfgfile>: each line "<tag> key=val ..."; runs qaf_vdp.py per line, 8 in parallel.
cd ~/tmp/aic-confsearch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
grep -v '^#' "$1" | grep . | xargs -P ${PAR:-8} -L1 bash -c 'tag=$0; [ -s out/vdp_$tag.tsv ] && [ $(wc -l < out/vdp_$tag.tsv) -ge 62 ] && exit 0; venv38/bin/python scripts/qaf_vdp.py ext/qaf-0628a6b pairs_dev.tsv out/vdp_$tag.tsv "$@" > out/vdp_$tag.log 2>&1; echo "$tag rc=$?"'
