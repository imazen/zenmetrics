#!/usr/bin/env python3
"""Tally coverage table for benchmarks/iqa_complexity_2026-10-17/README.md.

Lists every CLI metric name (crates/zenmetrics-cli/src/metrics/mod.rs) with
the counters that tallied it and totals at one shape, and rewrites the block
between the README's coverage markers. `--check` fails if the block is stale.

  coverage.py [--check] [--shape 1024x1024]
"""
import argparse, glob, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate, validate, parse_shape  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT/'benchmarks/iqa_complexity_2026-10-17'
README = DIR/'README.md'
BEGIN, END = '<!-- coverage:begin -->', '<!-- coverage:end -->'


def cli_metrics():
    mod = (ROOT/'crates/zenmetrics-cli/src/metrics/mod.rs').read_text()
    return sorted(set(re.findall(r'MetricKind::\w+ => "([a-z0-9-]+)"', mod)))


def table(shape):
    w, h = parse_shape(shape)
    by = {}
    for f in sorted(glob.glob(str(DIR/'tallies/*.json'))):
        t = json.loads(Path(f).read_text())
        if validate(t):
            raise SystemExit(f'invalid tally {f}')
        r = evaluate(t, w, h)
        by.setdefault(t['metric'], []).append((t['counter'], r))
    names = cli_metrics()
    counted = [n for n in names if n in by]
    lines = [f'Tallied CLI metric names: **{len(counted)} of {len(names)}** (shape {shape}; flops per original pixel, '
             'intrinsic + implementation classes; halo in original pixels). Two counters on one metric are independent '
             'counts; a difference between them is unresolved until listed in `reconcile/`.', '',
             '| metric | counter | flops/px | intrinsic | implementation | halo px | work/span | barriers |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for n in counted:
        for counter, r in sorted(by[n]):
            ws = r['work_over_span']
            lines.append(f"| {n} | {counter} | {r['flops_per_px']:.1f} | {r['intrinsic_flops_per_px']:.1f} | "
                         f"{r['implementation_flops_per_px']:.1f} | {r['halo_px']:.0f} | "
                         f"{ws:.3g} | {r['barriers']} |" if ws else
                         f"| {n} | {counter} | {r['flops_per_px']:.1f} | {r['intrinsic_flops_per_px']:.1f} | "
                         f"{r['implementation_flops_per_px']:.1f} | {r['halo_px']:.0f} | – | {r['barriers']} |")
    lines += ['', '**Not yet tallied:** ' + ', '.join(n for n in names if n not in by) + '.']
    extra = sorted(set(by) - set(names))
    if extra:
        lines += ['', '**Tallies for names that are not CLI metrics:** ' + ', '.join(extra) + '.']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--shape', default='1024x1024')
    a = ap.parse_args()
    text = README.read_text()
    block = BEGIN + '\n' + table(a.shape) + '\n' + END
    new = re.sub(re.escape(BEGIN) + '.*?' + re.escape(END), lambda _: block, text, flags=re.S) if BEGIN in text \
        else text.rstrip() + '\n\n## Coverage\n\n' + block + '\n'
    if a.check:
        if new != text:
            raise SystemExit('README coverage block is stale: run scripts/complexity/coverage.py')
        print('coverage block current')
        return
    README.write_text(new)
    print('updated', README)


if __name__ == '__main__':
    main()
