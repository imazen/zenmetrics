#!/usr/bin/env python3
"""Compare our tallies with the Pedzisz 2026 draft's weighted totals at 840x944.

Our tallies count zenmetrics' implementations; the draft counted reference or
production implementations (libvmaf, MATLAB, Python, C++). Differences are
expected; a large ratio marks stages to recheck, not an error in either.
The draft values were transcribed from partial screenshots of a preliminary
draft; no accuracy is claimed for them.

  compare_draft.py [--tallies GLOB] [--prefer-counter swe2,swe,lead]
"""
import argparse, glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate, validate, parse_shape  # noqa: E402
from correlate import weighted  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tallies', default=str(ROOT/'benchmarks/iqa_complexity_2026-10-17/tallies/*.json'))
    ap.add_argument('--prefer-counter', default='swe2,swe,lead')
    a = ap.parse_args()
    draft = json.loads((HERE/'pedzisz_2026_draft_values.json').read_text())
    weights = json.loads((HERE/'weights_pedzisz_2026_draft.json').read_text())['weights']
    w, h = parse_shape(draft['shape'])
    prefer = a.prefer_counter.split(',')
    best = {}
    for f in glob.glob(a.tallies):
        t = json.loads(Path(f).read_text())
        if validate(t):
            continue
        r = prefer.index(t['counter']) if t['counter'] in prefer else len(prefer)
        if t['metric'] not in best or r < best[t['metric']][0]:
            best[t['metric']] = (r, t)
    print('draft\tdraft_value\tzenmetrics\tours\tratio_ours_over_draft\tcounter\tunweighted')
    for v in draft['values']:
        name = v['zenmetrics']
        if not name or name not in best:
            print(f"{v['draft_name']}\t{v['value']}\t{name or 'not implemented'}\t\t\t\t{'no tally' if name else ''}")
            continue
        t = best[name][1]
        ours, missing = weighted(evaluate(t, w, h), weights)
        print(f"{v['draft_name']}\t{v['value']}\t{name}\t{ours:.2f}\t{ours / v['value']:.2f}\t{t['counter']}\t{';'.join(missing)}")


if __name__ == '__main__':
    main()
