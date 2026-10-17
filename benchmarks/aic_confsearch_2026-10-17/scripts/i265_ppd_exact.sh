#!/usr/bin/env bash
set -eu
cd ~/tmp/aic-confsearch
PATH=$HOME/.cargo/bin:$PATH ~/work/zen/zenmetrics--confsearch/benchmarks/aic_confsearch_2026-10-17/vdpprobe/build.sh ~/work/zen/zenmetrics--confsearch rs-i265
mkdir -p out-i265
export RAYON_NUM_THREADS=4
P=37.842505866005425
rs-i265/target/release/vdpprobe pairs_i265.tsv out-i265/rs_e_v3_xyz_ppdexact.tsv v=3 enc=xyz black=0 amb=0 ppd=$P &
rs-i265/target/release/vdpprobe pairs_i265.tsv out-i265/rs_e_v2_bits0_ppdexact.tsv v=2 black=0 amb=0 bits=0 ppd=$P &
wait
