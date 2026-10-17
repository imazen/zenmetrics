#!/usr/bin/env bash
# afine_eval.sh <log.tsv> <variant>...: evaluate A-FINE terms per variant vs the published A-FINE column.
cd ~/tmp/aic-confsearch
log=$1; shift
for v in "$@"; do
  f=out/afine_$v.tsv; [ -s $f ] || continue
  python3 - $f > out/afine_${v}_derived.tsv <<'PY'
import csv, sys, math
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
print('distorted\tfids\tall\tall_m\tnds\tfid\tfids_plus_dn\tall_scale')
for r in rows:
    g = {k: float(r[k]) for k in ('nd','nr','fid','nds','nrs','fids','all','all_m')}
    sc = (100 - 0) / (1 + math.exp(-(g['all'] - (-1.9710)) / abs(-2.3734))) + 0  # upstream scale_finalscore
    print(f"{r['distorted']}\t{g['fids']!r}\t{g['all']!r}\t{g['all_m']!r}\t{g['nds']!r}\t{g['fid']!r}\t{g['fids']+g['nds']-g['nrs']!r}\t{sc!r}")
PY
  python3 scripts/evalcol.py data/crop/metrics_subset.csv data/config.json A-FINE $log $(for t in fids all all_m nds fids_plus_dn all_scale; do echo "${v}:$t=out/afine_${v}_derived.tsv:$t"; done) > /dev/null
done
