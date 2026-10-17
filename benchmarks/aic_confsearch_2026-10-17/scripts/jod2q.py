"""jod2q.py <in.tsv> <jodcol> <out.tsv>: invert HDR-VDP-3.0.7's Q_JOD = 10 - 0.52*max(10-Q,0)^1.2812
(hdrvdp3.m lines 724-726) to recover res.Q; exact wherever Q < 10."""
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
with open(sys.argv[3], 'w') as f:
    f.write('distorted\tscore\n')
    for r in rows:
        j = float(r[sys.argv[2]])
        q = 10 - ((10 - j) / 0.52) ** (1 / 1.2812) if j < 10 else 10.0
        f.write(f"{r['distorted']}\t{q!r}\n")
