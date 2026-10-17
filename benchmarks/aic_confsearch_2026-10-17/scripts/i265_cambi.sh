#!/usr/bin/env bash
cd ~/tmp/aic-confsearch; mkdir -p out-i265
V=~/tmp/aic-variants-scratch/round4/oracles/vmaf-3.2.1/libvmaf/build-aic/tools/vmaf
for v in formula601 ff_420 ff_420+acc ff_420+709 ff_420+full ff_420+10 ff_444 ff_420+acc+10; do
  python3 scripts/cambi_probe.py $V pairs_i265.tsv out-i265/cambi.tsv out-i265/cambi-scratch-$v $v
done
