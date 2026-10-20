#!/usr/bin/env python3
"""Integrity checks for the committed AIC2026 reproduction (benchmarks/aic2026_full_repro_*).

    repro_check.py [--reference combined.csv] [--json]

Without --reference (CI, any host) it checks what the repository alone can prove:
- each committed `metrics_<set>.zenmetrics.csv.gz` decompresses to the sha256 its metrics.pointer.md records,
  and each provenance file matches its recorded sha256;
- the CSV has unique rows, the nine metadata columns first, values only in mapped columns, no values in columns
  produced by unpublished implementations (dvifmish, private ports), and every JND_<col> equal to the site mapping
  of our own value;
- compare_<set>.tsv covers exactly the mapped columns, and its n never exceeds our non-empty values;
- the routes used (compare `cli`, provenance) agree with scripts/complexity/aic2026_columns.json.
With --reference (the July/drop union `combined.csv`, s3://zentrain/aic2026/csv/july-drop-2026-10-20/) it also
requires that file's sha256 to be the one each set's metrics.pointer.md names as its reference, re-derives every
compare_<set>.tsv with july_diff on the matching view, and requires it to be identical to the committed one.

Exit status 1 when any error-level check fails. --json prints the findings for brief_data.py.
"""
import argparse, csv, gzip, hashlib, io, json, os, re, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from aic_columns import COLS, JNDS, META, REF_META, REFERENCE, sitejnd  # noqa: E402

# Columns whose values come from implementations that are not published here: never commit per-image values.
UNPUBLISHED = ('proposal-DVIFM', 'proposal-DVIFM-0.2', 'proposal-DVIFM-0.2-use_chroma', 'proposal-DVIFM-2d', 'proposal-DVIFM-min',
               'proposal-IDFIQA', 'MSSWD', 'TOPIQ', 'TOPIQ-PIPAL')


def finding(out, level, check, detail):
    out.append(dict(level=level, check=check, detail=detail))


def pointer_shas(text):
    """`file` | bytes | `sha` rows of a metrics.pointer.md table -> {file: sha256}."""
    out = {}
    for l in text.splitlines():
        m = re.match(r'\|\s*`([^`]+)`\s*\|[^|]*\|\s*`([0-9a-f]{64})`', l)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def check_set(d, s, out, reference):
    gz = d / f'metrics_{s}.zenmetrics.csv.gz'
    raw = gzip.decompress(gz.read_bytes())
    shas = pointer_shas((d / 'metrics.pointer.md').read_text()) if (d / 'metrics.pointer.md').exists() else {}
    want = shas.get(f'metrics_{s}.zenmetrics.csv')
    got = hashlib.sha256(raw).hexdigest()
    if want != got:
        finding(out, 'error', f'{d.name}/{s}: CSV hash', f'decompressed sha256 {got[:12]} but metrics.pointer.md records {str(want)[:12]}')
    prov = d / f'provenance_{s}.json'
    wantp = shas.get(f'provenance_{s}.json') or (shas.get('provenance.json') if s == 'cropped' else None)
    if prov.exists() and wantp:
        if hashlib.sha256(prov.read_bytes()).hexdigest() != wantp:
            finding(out, 'error', f'{d.name}/{s}: provenance hash', 'provenance file differs from the pointer\'s sha256')
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    header = list(rows[0].keys())
    if not set(META) <= set(header):
        finding(out, 'error', f'{d.name}/{s}: header', 'metadata columns missing')
    names = [r['distorted'] for r in rows]
    if len(set(names)) != len(names):
        finding(out, 'error', f'{d.name}/{s}: rows', f'{len(names) - len(set(names))} duplicate distorted names')
    filled = {c for c in header if c not in REF_META and any(r[c] not in ('', 'nan', 'NaN') for r in rows)}
    leak = sorted(c for c in filled if c in UNPUBLISHED)
    if leak:
        finding(out, 'error', f'{d.name}/{s}: unpublished implementations', 'per-image values present for ' + ', '.join(leak))
    bad = 0
    for jc, (base, mk) in JNDS.items():
        if jc not in header or base not in header:
            continue
        for r in rows:
            v = r[base]
            want_j = '' if v in ('', 'nan', 'NaN') else sitejnd(mk, float(v))
            if (r[jc] == '') != (want_j == '') or (r[jc] != '' and float(r[jc]) != want_j):
                bad += 1
    if bad:
        finding(out, 'error', f'{d.name}/{s}: JND columns', f'{bad} JND_<col> values differ from the site mapping of our value')
    cmp_path = d / f'compare_{s}.tsv'
    if not cmp_path.exists():
        finding(out, 'error', f'{d.name}/{s}: compare', f'{cmp_path.name} missing')
        return
    cmp = {r['column']: r for r in csv.DictReader(cmp_path.open(), delimiter='\t')}
    mapped = [c for c in header if c in COLS or (c in filled and not c.startswith('JND_'))]
    missing = [c for c in mapped if c not in cmp]
    if missing:
        finding(out, 'error', f'{d.name}/{s}: compare coverage', 'no compare row for ' + ', '.join(missing))
    for c in mapped:
        r = cmp.get(c)
        if not r:
            continue
        ours_n = sum(1 for x in rows if x[c] not in ('', 'nan', 'NaN'))
        if r['n'].isdigit() and int(r['n']) > ours_n:
            finding(out, 'error', f'{d.name}/{s}: compare n', f'{c}: compare n {r["n"]} exceeds our {ours_n} values')
        if c in COLS and r['cli'] != COLS[c][0] and not (d / f'plan_{s}.json').exists():
            finding(out, 'error', f'{d.name}/{s}: compare route', f'{c}: compare says {r["cli"]}, aic_columns.py says {COLS[c][0]}')
    cmap = json.loads((ROOT / 'scripts/complexity/aic2026_columns.json').read_text())['map']
    for c in mapped:
        used, listed = (cmp[c]['cli'] if c in cmp else COLS[c][0]), cmap.get(c)
        if listed and used != listed and not used.startswith('@'):
            finding(out, 'warn', f'{d.name}/{s}: route differs from column map',
                    f'{c} was scored with {used}; scripts/complexity/aic2026_columns.json maps it to {listed}')
    ref_sha = re.search(r'[Rr]eference[\s\S]{0,160}?`([0-9a-f]{64})`', (d / 'metrics.pointer.md').read_text()) if (d / 'metrics.pointer.md').exists() else None
    if not ref_sha:
        finding(out, 'error', f'{d.name}/{s}: reference', 'metrics.pointer.md names no reference table sha256')
    elif ref_sha.group(1) != REFERENCE['sha256']:
        finding(out, 'warn', f'{d.name}/{s}: reference', f'compared against {ref_sha.group(1)[:12]}, not the current reference {REFERENCE["sha256"][:12]} (aic_columns.REFERENCE)')
    if reference:
        pub = Path(reference)
        got_ref = hashlib.sha256(pub.read_bytes()).hexdigest()
        if ref_sha and got_ref != ref_sha.group(1):
            finding(out, 'error', f'{d.name}/{s}: reference', f'{pub.name} sha256 {got_ref[:12]} is not the recorded reference {ref_sha.group(1)[:12]}')
            return
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR', str(Path.home() / 'tmp'))) as t:
            gzp = Path(t) / 'ours.csv'
            gzp.write_bytes(raw)
            import subprocess
            plan = d / f'plan_{s}.json'
            r = subprocess.run([sys.executable, str(HERE / 'july_diff.py'), str(gzp), str(pub), f'{t}/cmp.tsv', f'{t}/worst.txt', '--view', s]
                               + (['--plan', str(plan)] if plan.exists() else []), capture_output=True, text=True)
            if r.returncode:
                finding(out, 'error', f'{d.name}/{s}: july_diff', r.stderr.strip() or r.stdout.strip())
            elif Path(f'{t}/cmp.tsv').read_text() != cmp_path.read_text():
                finding(out, 'error', f'{d.name}/{s}: compare stale', f'{cmp_path.name} differs from july_diff on the committed CSV')


def run(reference=None):
    out = []
    dirs = sorted(p.parent for p in ROOT.glob('benchmarks/aic2026_*/metrics_*.zenmetrics.csv.gz'))
    if not dirs:
        finding(out, 'error', 'reproduction', 'no committed metrics_<set>.zenmetrics.csv.gz')
    for d in sorted(set(dirs)):
        for gz in sorted(d.glob('metrics_*.zenmetrics.csv.gz')):
            check_set(d, gz.name.split('_', 1)[1].split('.')[0], out, reference)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--reference', help='the July/drop union combined.csv (view column selects each set)')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    out = run(a.reference)
    if a.json:
        print(json.dumps(out, indent=1))
    else:
        for f in out:
            print(f"{f['level'].upper():5} {f['check']}: {f['detail']}")
        print(f"{sum(f['level'] == 'error' for f in out)} error(s), {sum(f['level'] == 'warn' for f in out)} warning(s)")
    return 1 if any(f['level'] == 'error' for f in out) else 0


if __name__ == '__main__':
    sys.exit(main())
