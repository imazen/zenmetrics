#!/usr/bin/env python3
"""Print each gmsd-chroma headline number straight from its retained raw file.

Usage: recompute_headlines.py <name>   (no name lists the names)
Raw files live under /var/tmp/gmsd-chroma (never in a repo).
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path('/var/tmp/gmsd-chroma')


def load(rel):
    return json.loads((ROOT / rel).read_text())


def mdsi_parity():
    a = load('parity.json')
    r = a['records']
    return dict(pairs=len(r), passed=a['passed'],
                max_abs=max(abs(x['author'] - x['rust']) for x in r),
                max_rel=max(abs(x['author'] - x['rust']) / abs(x['author']) for x in r if x['author'] != 0),
                wrong_constant_rejections=a['negative_control_failures'])


def ms_parity():
    b = load('ms_parity.json')
    out = {}
    for m in ('ms_gmsd', 'ms_gmsdc'):
        v = [x[m] for x in b['records']]
        out[m] = dict(pairs=len(v), passed=b['passed'],
                      max_abs=max(abs(x['numpy'] - x['rust']) for x in v),
                      max_rel=max(abs(x['numpy'] - x['rust']) / abs(x['numpy']) for x in v if x['numpy'] != 0),
                      author_software_parity=b['author_software_parity'])
    return out


def c8_author_cs():
    d = load('c8/author_cs_v4.json')
    r = d['records']
    return dict(pairs=d['pairs'], passed=d['passed'], negative_control_rejections=d['negative_control_rejections'],
                max_abs=max(x['max_abs'] for x in r), max_rel=max(x['max_rel'] for x in r))


def c8_identity():
    d = load('c8/identity_v3/report_v4.json')
    return dict(compared_cells=d['compared_cells'], differing_cells=d['differing_cells'],
                modes=len(d['modes']), baseline_binary=d['baseline_binary_sha256'],
                candidate_binary=d['candidate_binary_sha256'])


def c8_dead_slots():
    d = ROOT / 'c8/identity_v3/v4_on_native_mt1'
    rows = []
    for s in ('cid22', 'safesyn', 'kadid'):
        with (d / f'{s}.csv').open() as f:
            rows += list(csv.DictReader(f))
    slots = [k for k in rows[0] if k.startswith('f') and k[1:].isdigit() and int(k[1:]) >= 1322]
    nz = [sum(1 for r in rows if float(r[k]) != 0.0) for k in slots]
    return dict(rows=len(rows), slots=len(slots), dead=[k for k, n in zip(slots, nz) if n == 0],
                min_nonzero=min(nz), max_nonzero=max(nz))


def cost():
    out = {}
    for t in ('mt1', 'mt8'):
        # the fit log carries both; recompute from its own recorded pairs
        p = ROOT / 'logs/c8_cost_fit_both_v3.log'
        for line in p.read_text().splitlines():
            d = json.loads(line)
            if d['path'].endswith(f'cost_v3_{t}.zenbench'):
                off, on = d['pairs']['1024']['off_ns'], d['pairs']['1024']['on_ns']
                out[t] = dict(raw_1024_pct=100 * (on - off) / off, fit_1024_pct=d['fit_pct_at1024'],
                              alpha_ns=d['alpha_ns'], beta_ns_per_px=d['beta_ns_per_px'], r2=d['r2'])
    return out


def colour():
    rows = load('colour_v3/report.json')['rows']
    return [dict(set=r['set'], distortion=r['distortion'], arm=r['arm'], n=r['n'],
                 srocc=r['srocc_signed'], ci95=r['ci95'])
            for r in rows if r['distortion'] == 'ALL_COLOUR']


def mdsi_maps():
    p = load('parity.json')['records']
    return dict(pairs=len(p), gcs_max_abs=max(x['gcs']['max_abs'] for x in p),
                gcs_max_rel=max(x['gcs']['max_rel'] for x in p),
                cs_max_abs=max(x['cs']['max_abs'] for x in p),
                cs_max_rel=max(x['cs']['max_rel'] for x in p),
                bitwise_checks=sorted({c for x in p for c in x['bitwise_checks']}))


def mdsi_vs_reference():
    with (ROOT / 'mdsi_vs_reference.tsv').open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def speed():
    return load('speed/report.json')


def calibration():
    return load('c8/calibration/chroma_report.json')


NAMES = dict(mdsi_parity=mdsi_parity, mdsi_maps=mdsi_maps, mdsi_vs_reference=mdsi_vs_reference, ms_parity=ms_parity, c8_author_cs=c8_author_cs,
             c8_identity=c8_identity, c8_dead_slots=c8_dead_slots, cost=cost, colour=colour,
             speed=speed, calibration=calibration)

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in NAMES:
        raise SystemExit('names: ' + ' '.join(NAMES))
    print(json.dumps(NAMES[sys.argv[1]](), sort_keys=True))
