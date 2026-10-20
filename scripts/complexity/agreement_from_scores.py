#!/usr/bin/env python3
"""Agreement of per-image metric scores with the preliminary AIC2026 subjective JND, as aggregates.

Same method as the published-column agreement in docs/paper/pareto.json: mean over sources of the
within-source Spearman correlation (sign aligned so 1 is full agreement, sources with at least five
images), a 2,000-sample bootstrap over sources (seed 20261017), and the pooled |Spearman|.

    agreement_from_scores.py --jnd LABELS.csv --out OUT.json NAME=SCORES.tsv ...

LABELS.csv: the private reconstruction (columns distorted, source, JND_score); it never leaves the
host. SCORES.tsv: two columns, distorted name (no extension) and score. OUT.json holds per-metric
aggregates only, in the field names brief_data.py --agreement reads.
"""
import argparse, csv, json, math, random, statistics
from pathlib import Path
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--jnd', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('scores', nargs='+', help='NAME=PATH')
    a = ap.parse_args()
    lab = list(csv.DictReader(a.jnd.open()))
    jnd = {r['distorted']: float(r['JND_score']) for r in lab}
    src = {r['distorted']: r['source'] for r in lab}
    out = []
    for spec in a.scores:
        name, path = spec.split('=', 1)
        pts = []
        for line in Path(path).read_text().splitlines():
            d, v = line.split('\t')[:2]
            d = d.rsplit('.', 1)[0] if d.endswith('.png') else d
            v = float(v)
            if d in jnd and math.isfinite(v):
                pts.append((src[d], v, jnd[d]))
        rho = spearmanr([p[1] for p in pts], [p[2] for p in pts])[0]
        sign = -1 if rho < 0 else 1
        by = {}
        for s, v, j in pts:
            by.setdefault(s, []).append((v, j))
        per = {}
        for s, l in by.items():
            if len(l) >= 5:
                r = spearmanr([v for v, _ in l], [j for _, j in l])[0]
                if not math.isnan(r):
                    per[s] = r * sign
        rng = random.Random(20261017)
        keys = sorted(per)
        boots = sorted(statistics.fmean(per[rng.choice(keys)] for _ in keys) for _ in range(2000))
        out.append(dict(metric=name, n=len(pts), sources=len(per), overall_abs_srcc=abs(rho),
                        direction='higher=worse' if rho > 0 else 'higher=better',
                        within_source_srcc=statistics.fmean(per.values()), ci_lo=boots[50], ci_hi=boots[1949]))
        print(f'{name:28} n={len(pts)} src={len(per)} within={out[-1]["within_source_srcc"]:.4f} '
              f'[{out[-1]["ci_lo"]:.4f}, {out[-1]["ci_hi"]:.4f}] |rho|={abs(rho):.4f}')
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
