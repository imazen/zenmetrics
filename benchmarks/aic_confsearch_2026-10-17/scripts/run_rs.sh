#!/usr/bin/env bash
# run_rs.sh <cfgfile>: lines "<tag> key=val ..." -> out/rs_<tag>.tsv via vdpprobe, PAR at a time.
cd ~/tmp/aic-confsearch; export RAYON_NUM_THREADS=${RT:-2}
grep -v '^#' "$1" | grep . | xargs -P ${PAR:-8} -L1 bash -c 'tag=$0; [ -s out/rs_$tag.tsv ] && [ $(wc -l < out/rs_$tag.tsv) -ge 62 ] && exit 0; rs/target/release/vdpprobe pairs_dev.tsv out/rs_$tag.tsv "$@" > out/rs_$tag.log 2>&1; echo "$tag rc=$?"'
