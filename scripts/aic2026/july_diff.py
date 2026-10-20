#!/usr/bin/env python3
"""Diff our assembled AIC2026 scores against a published reference table.

    july_diff.py OURS.csv[.gz] REFERENCE.csv[.gz] REPORT.tsv WORST.txt [--view cropped|fullres] [--plan PLAN.json]

OURS: assemble.py or `just aic-rescore` output (a reference layout with our values in the metric columns).
REFERENCE: the July `metrics_<set>.csv`, the online export, or the July/drop union `combined.csv`
(s3://zentrain/aic2026/csv/july-drop-2026-10-20/, sha256 91d7c2df…), whose `view` column needs --view.
Rows are matched by `distorted` name (the union orders cropped rows differently); both files must hold the same
names. Every reference metric column with our values is compared, plus the COLS columns (empty ones as nodata).
--plan takes the routes from an aic-rescore plan; otherwise from aic_columns.COLS.
"""
import argparse, csv, gzip, io, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aic_columns import REF_META, write_report  # noqa: E402


def read_csv(path):
    raw = open(path, 'rb').read()
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))


def select_view(rows, view, what):
    if rows and 'view' in rows[0]:
        if not view:
            sys.exit(f'{what} has a view column: pass --view')
        rows = [r for r in rows if r['view'] == view]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('ours'); ap.add_argument('reference'); ap.add_argument('report'); ap.add_argument('worst')
    ap.add_argument('--view'); ap.add_argument('--plan')
    a = ap.parse_args()
    ours_rows = select_view(read_csv(a.ours), a.view, 'OURS')
    pub = select_view(read_csv(a.reference), a.view, 'REFERENCE')
    header = list(pub[0].keys())
    if {r['distorted'] for r in ours_rows} != {r['distorted'] for r in pub} or len(ours_rows) != len(pub):
        sys.exit('OURS and REFERENCE hold different distorted names')
    routes = None
    if a.plan:
        plan = json.load(open(a.plan))
        routes = {c: v['route'] for c, v in plan.get('columns', {}).items() if isinstance(v, dict) and v.get('route')}
    cols = [c for c in ours_rows[0] if c not in REF_META]
    ours = {c: {r['distorted']: float(r[c]) for r in ours_rows if r[c] not in ('', 'nan', 'NaN')} for c in cols}
    ours = {c: v for c, v in ours.items() if v}
    write_report(pub, header, ours, a.report, a.worst, routes)
    print('wrote', a.report, a.worst)


if __name__ == '__main__':
    main()
