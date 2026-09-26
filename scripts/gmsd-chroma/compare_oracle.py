#!/usr/bin/env python3
"""Apply the preregistered f64 score/map gate, preserving every failure."""
import array
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/var/tmp/gmsd-chroma')


def table(path):
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    result = {r['id']: r for r in rows}
    assert len(result) == len(rows), 'duplicate IDs'
    return result


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def values(path):
    import sys
    result = array.array('d')
    result.frombytes(path.read_bytes())
    if sys.byteorder != 'little':
        result.byteswap()
    return result


def delta(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    absolute = abs(a-b)
    relative = absolute/abs(a) if a else (0.0 if b == 0 else None)
    return absolute, relative


def main():
    build = json.loads((ROOT/'build_examples_latest.json').read_text())
    assert sha(ROOT/'target/release/examples/mdsi_oracle') == build['binary_sha256']['mdsi_oracle']
    author = table(ROOT/'octave/scores.tsv')
    ours = table(ROOT/'rust/rust.tsv')
    expected = table(ROOT/'oracle_inputs.tsv')
    assert author.keys() == ours.keys() == expected.keys(), 'missing/extra pair IDs'
    records = []
    failures = []
    negative_failed = 0
    negative_gcs_samples = 0
    for key in expected:
        a, b = float(author[key]['mdsi']), float(ours[key]['mdsi'])
        absolute, relative = delta(a, b)
        passed = (absolute <= 1e-12) if a == 0 else (relative <= 1e-9)
        checks = ours[key]['bitwise_checks'].split(',')
        assert {'scalar','threads1','threads8'} <= set(checks)
        record = dict(id=key, author=a, rust=b, absolute=absolute, relative=relative,
                      passed=passed, bitwise_checks=checks)
        wrong = float(ours[key]['wrong_constant'])
        wa, wr = delta(a, wrong)
        negative_failed += int(wa > 1e-12 if a == 0 else wr > 1e-9)
        if not passed:
            failures.append(key+':score')
        for map_name in ('cs', 'gcs'):
            ap = ROOT/'octave'/f'{key}.{map_name}.f64'
            bp = ROOT/'rust'/f'{key}.{map_name}.f64'
            av, bv = values(ap), values(bp)
            if map_name == 'gcs':
                record['negative_gcs_samples'] = sum(x < 0 for x in av)
                negative_gcs_samples += record['negative_gcs_samples']
            assert len(av) == len(bv) == int(ours[key]['width'])*int(ours[key]['height'])
            max_abs = 0.0
            max_rel = 0.0
            near_zero_max_abs = 0.0
            for x, y in zip(av, bv):
                da, dr = delta(x, y)
                max_abs = max(max_abs, da)
                if abs(x) > 1e-12:
                    max_rel = max(max_rel, dr)
                else:
                    near_zero_max_abs = max(near_zero_max_abs, da)
            ok = max_abs <= 1e-12 and max_rel <= 1e-9
            record[map_name] = dict(samples=len(av), max_abs=max_abs, max_rel=max_rel,
                                    near_zero_max_abs=near_zero_max_abs, passed=ok,
                                    author_sha256=sha(ap), rust_sha256=sha(bp))
            if not ok:
                failures.append(key+':'+map_name)
        records.append(record)
    if negative_gcs_samples == 0:
        failures.append('coverage:no-negative-GCS-in-author-maps')
    result = dict(schema='gmsd-chroma-parity-v1', pairs=len(records), failures=failures,
                  max_score_abs=max(r['absolute'] for r in records),
                  max_score_rel=max(r['relative'] for r in records if r['relative'] is not None),
                  negative_control_failures=negative_failed,
                  negative_gcs_samples=negative_gcs_samples,
                  passed=not failures and negative_failed > 0,
                  inputs_sha256=sha(ROOT/'oracle_inputs.tsv'), records=records,
                  binary_sha256=build['binary_sha256']['mdsi_oracle'],
                  source_sha256=build['source_sha256'])
    (ROOT/'parity.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},sort_keys=True))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
