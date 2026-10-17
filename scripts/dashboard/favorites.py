#!/usr/bin/env python3
"""Owner priority list ("favorites"), ordered by first implementation in zenmetrics.

The metric set is owner-curated (2026-10-17) from AIC-4 discussion material.
For each CLI metric name the earliest of three git events on master is used:
the crate's Cargo.toml being added, the `MetricKind` variant first appearing,
or the CLI name mapping first appearing. Needs git history, so the result is
committed as scripts/dashboard/favorites.json and `--check` recomputes it.

  scripts/dashboard/favorites.py [--check] [--git-dir DIR] [--rev master@origin-equivalent]
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'scripts/dashboard/favorites.json'
# (label as discussed, CLI names, crate dir or None). Order here is not used.
FAVORITES = [
    ('PSNR', ['psnr', 'psnr-y'], None),
    ('SSIM', ['ssim', 'ssim-libvmaf'], 'msssim'),
    ('MS-SSIM', ['msssim', 'msssim-libvmaf'], 'msssim'),
    ('FSIM / FSIMc', ['fsim-y', 'fsim', 'fsim-k4'], 'fsim'),
    ('HaarPSI', ['haarpsi-y', 'haarpsi'], 'haarpsi'),
    ('PSNR-HVS', ['psnrhvs-y', 'psnrhvs'], 'psnrhvs'),
    ('CVVDP', ['cvvdp'], 'cvvdp'),
    ('VMAF', ['vmaf', 'vmaf-neg'], 'vmaf'),
    ('Butteraugli', ['butteraugli'], None),
    ('SSIMULACRA2', ['ssim2'], None),
    ('SSIMULACRA', ['ssimulacra1'], None),
    ('DSSIM', ['dssim'], None),
    ('VSI', ['vsi', 'vsi-k4'], 'vsi'),
    ('GMSD', ['gmsd'], 'gmsd'),
    ('VIF', ['vif'], 'vif'),
    ('NLPD', ['nlpd'], 'nlpd'),
    ('mDCT-PSNR', ['mdctpsnr'], 'mdctpsnr'),
    ('MAD', ['mad'], 'mad-iqa'),
    ('IW-SSIM', ['iwssim', 'iwssim-piq'], 'iwssim'),
    ('HDR-VDP-2 / -3', ['hdrvdp', 'hdrvdp3'], 'hdrvdp'),
    ('CW-SSIM', ['cw-ssim'], None),
    ('CIEDE2000', ['ciede2000'], None),
    ('CAMBI', ['cambi'], None),
    ('PieAPP', ['pieapp'], None),
    ('Wasserstein Distortion', ['wd'], None),
    ('WS-HV', ['ws-hv'], None),
]
NOT_IMPLEMENTED = ['XPSNR', 'DVIFM', 'IDFIQA', 'TOPIQ', 'HaarXYB / HaarYCoCg / HaarY delta-L variants']


def git(args, git_dir):
    return subprocess.run(['git', '--git-dir', git_dir] + args, capture_output=True, text=True, check=True).stdout


def first(git_dir, rev, args):
    lines = [l for l in git(['log', rev, '--reverse', '--format=%H %aI'] + args, git_dir).splitlines() if l]
    if not lines:
        return None
    sha, date = lines[0].split()
    return {'commit': sha[:12], 'date': date}


def compute(git_dir, rev):
    mod = git(['show', f'{rev}:crates/zenmetrics-cli/src/metrics/mod.rs'], git_dir)
    variants = {name: v for v, name in re.findall(r'MetricKind::(\w+) => "([a-z0-9-]+)"', mod)}
    out = []
    for label, names, crate in FAVORITES:
        events = {}
        for name in names:
            if name not in variants:
                raise SystemExit(f'{name}: not a CLI metric at {rev}')
            for basis, args in [('cli_name', ['-S', f'=> "{name}"', '--', '*.rs']),
                                ('enum_variant', ['-G', f'MetricKind::{variants[name]}[^0-9A-Za-z_]', '--', '*.rs'])]:
                e = first(git_dir, rev, args)
                if e:
                    events[f'{basis}:{name}'] = e
        if crate:
            e = first(git_dir, rev, ['--diff-filter=A', '--', f'crates/{crate}/Cargo.toml'])
            if e:
                events[f'crate:{crate}'] = e
        basis = min(events, key=lambda k: events[k]['date'])
        out.append({'label': label, 'cli_names': names, 'first': events[basis], 'basis': basis})
    out.sort(key=lambda r: r['first']['date'])
    return {'schema': 1, 'note': 'Owner priority list (2026-10-17), ordered by first implementation in zenmetrics; '
            'first = earliest of crate added, MetricKind variant, or CLI name on the stated revision. '
            'Code that predates this repository (for example sibling crates) is not dated here.',
            'revision': git(['rev-parse', rev], git_dir).strip(), 'favorites': out, 'not_implemented': NOT_IMPLEMENTED}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--git-dir', default=None)
    ap.add_argument('--rev', default='origin/master')
    a = ap.parse_args()
    git_dir = a.git_dir or subprocess.run(['jj', 'git', 'root'], cwd=ROOT, capture_output=True, text=True).stdout.strip() \
        or str(ROOT/'.git')
    if a.check:
        old = json.loads(OUT.read_text())
        new = compute(git_dir, old['revision'])
        if new != old:
            raise SystemExit('favorites.json differs from recomputation at its recorded revision')
        print('favorites.json reproduces at', old['revision'][:12])
        return
    OUT.write_text(json.dumps(compute(git_dir, a.rev), indent=1) + '\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
