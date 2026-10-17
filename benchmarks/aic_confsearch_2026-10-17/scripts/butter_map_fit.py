"""Fit proposal-Butteraugli as a function of the site's Butteraugli-3norm column.
usage: butter_map_fit.py <site mcsv_cropped.csv> [<site mcsv_fullres.csv> ...]
Prints the two linear segments of P = max(k_lo*N, k_hi*N + c_hi) and the residuals."""
import csv, sys, numpy as np
for f in sys.argv[1:]:
    rows = [r for r in csv.DictReader(open(f)) if r['proposal-Butteraugli'] not in ('', 'nan') and r['Butteraugli-3norm'] not in ('', 'nan')]
    P = np.array([float(r['proposal-Butteraugli']) for r in rows]); N = np.array([float(r['Butteraugli-3norm']) for r in rows])
    lo = N < 0.5275
    k_lo = np.linalg.lstsq(N[lo][:, None], P[lo], rcond=None)[0][0]
    k_hi, c_hi = np.polyfit(N[~lo], P[~lo], 1)
    pred = np.maximum(k_lo * N, k_hi * N + c_hi); d = np.abs(pred - P)
    print(f'{f}: n={len(rows)} low n={lo.sum()} k_lo={k_lo:.10f} k_hi={k_hi:.9f} c_hi={c_hi:.9f} '
          f'breakpoint={c_hi / (k_lo - k_hi):.6f} |d| median={np.median(d):.3g} p99={np.percentile(d, 99):.3g} max={d.max():.3g}')
