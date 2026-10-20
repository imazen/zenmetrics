#!/usr/bin/env python3
"""Diff an assembled zenmetrics CSV against the published July 2026 AIC2026 objective scores.

    july_diff.py OURS.csv[.gz] PUBLISHED.csv REPORT.tsv WORST.txt

OURS is assemble.py output (published header and row order, our values in the metric columns); it may be gzipped.
Writes the same per-column report and worst-row list that assemble.py writes, from the CSV alone, so a committed
reproduction can be re-checked against the published file without the per-metric score tables.
"""
import csv, gzip, io, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aic_columns import COLS, write_report  # noqa: E402


def read_csv(path):
    raw = open(path, 'rb').read()
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    return list(csv.DictReader(io.StringIO(raw.decode())))


def main():
    ours_path, pub_path, report, worst = sys.argv[1:5]
    ours_rows, pub = read_csv(ours_path), read_csv(pub_path)
    header = list(pub[0].keys())
    if list(ours_rows[0].keys()) != header:
        sys.exit('header differs from the published file')
    if [r['distorted'] for r in ours_rows] != [r['distorted'] for r in pub]:
        sys.exit('row order differs from the published file')
    ours = {c: {r['distorted']: float(r[c]) for r in ours_rows if r[c] not in ('', 'nan', 'NaN')} for c in COLS if c in header}
    write_report(pub, header, ours, report, worst)
    print('wrote', report, worst)


if __name__ == '__main__':
    main()
