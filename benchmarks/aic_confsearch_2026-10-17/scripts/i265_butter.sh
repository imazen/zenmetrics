#!/usr/bin/env bash
cd ~/tmp/aic-confsearch; mkdir -p out-i265
RAYON_NUM_THREADS=2 ~/tmp/iqa-cx-lead/bin/zenmetrics-nn batch --metric butteraugli --pairs pairs_i265.tsv --output out-i265/butteraugli.tsv --jobs 1 > out-i265/butteraugli.log 2>&1
