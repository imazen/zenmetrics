"""Rebuild the separate port/oracle consistency table from retained observations.
Usage: summarize_controls.py SCRATCH OUTPUT_TSV
"""
import csv
import hashlib
from pathlib import Path
import statistics as st
import sys

scratch,out=map(Path,sys.argv[1:]);repo=Path(__file__).resolve().parents[3]
retained=repo/'benchmarks/aic_refusals_2026-10-18'
cases=[
 ('v2-retained-replay',scratch/'hdr2-retained.tsv','q',retained/'hdrvdp2-octave.tsv','octave_q',61),
 ('v2-mex-vs-fallback',scratch/'hdr2-mex-200-all61.tsv','q',scratch/'hdr2-retained.tsv','q',61),
 ('v2-rust-vs-octave-mex',scratch/'rust2-replay.tsv','v2_q',scratch/'hdr2-mex-200-all61.tsv','q',61),
 ('v3-upstream-vs-retained-rust',scratch/'hdr3-200-10bit-r2.tsv','q',retained/'hdrvdp3-aic2026.tsv','native',61),
 ('v3-rust-replay',scratch/'rust3-replay.tsv','v3_q',retained/'hdrvdp3-aic2026.tsv','native',61),
 ('v3-306-vs-307',scratch/'hdr3-v306-200-screen6-r2.tsv','q',scratch/'hdr3-200-10bit-r2.tsv','q',6),
]
def load(p,field):
 rows=list(csv.DictReader(p.open(),delimiter='\t'))
 assert len({r['distorted'] for r in rows})==len(rows)
 return {r['distorted']:float(r[field]) for r in rows}
with out.open('w') as f:
 w=csv.writer(f,delimiter='\t');w.writerow(['comparison','n','median_abs_q','max_abs_q','left_sha256','right_sha256','scope'])
 for label,left,lf,right,rf,n in cases:
  a,b=load(left,lf),load(right,rf);assert len(a)==n and a.keys()<=b.keys()
  if n==61:assert a.keys()==b.keys()
  d=[abs(v-b[k]) for k,v in a.items()]
  w.writerow([label,n,st.median(d),max(d),hashlib.sha256(left.read_bytes()).hexdigest(),hashlib.sha256(right.read_bytes()).hexdigest(),'full-61-consistency' if n==61 else 'six-pair-version-screen'])
