#!/usr/bin/env python3
# DMM AIC-4 native-convention probe — compares our port's scores against the
# published AIC-4 2026 `DMM` column on the 61-pair crop set.
#
# Scores are produced by the CLI:
#   ZENMETRICS_NN_MODEL_DIR=<models> ZENMETRICS_NN_ENGINE=ort-cuda \
#     zenmetrics batch --metric dmm --pairs <pairs.tsv> --output aic_dmm.tsv
# This script then joins that TSV against the published column and prints
# the deviation stats.
#
# Result (2026-10-16): the published column is NOT reproduced by the upstream
# convention — median |d| 7.47e-2, max 0.417 across 61 pairs.
import csv
import sys

ours_path = sys.argv[1] if len(sys.argv) > 1 else 'aic_dmm.tsv'
pub_path = sys.argv[2] if len(sys.argv) > 2 else 'metrics_subset.csv'

ours = {}
with open(ours_path) as f:
    for r in csv.DictReader(f, delimiter='\t'):
        ours[r['distorted']] = float(r['dmm'])
pub = {}
with open(pub_path) as f:
    for r in csv.DictReader(f):
        pub[r['distorted']] = float(r['DMM'])

ds = sorted(abs(ours[k] - pub[k]) for k in pub if k in ours)
n = len(ds)
print(f'n={n} median|d|={ds[n // 2]:.3e} max|d|={ds[-1]:.3e}')
worst = sorted(((abs(ours[k] - pub[k]), k, ours[k], pub[k]) for k in pub if k in ours))[-5:]
for d, k, o, p in worst:
    print(f'  {k}: ours {o:.6f} pub {p:.6f} |d| {d:.3e}')
