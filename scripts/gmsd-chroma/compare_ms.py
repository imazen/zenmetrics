#!/usr/bin/env python3
"""Qualify the declared paper variant against its independent NumPy oracle."""
import json
from compare_oracle import ROOT, delta, sha, table, values


def main():
    build = json.loads((ROOT/'build_examples_latest.json').read_text())
    assert sha(ROOT/'target/release/examples/mdsi_oracle') == build['binary_sha256']['mdsi_oracle']
    expected = table(ROOT/'oracle_inputs.tsv')
    numpy = table(ROOT/'ms_numpy/scores.tsv')
    rust = table(ROOT/'rust/ms_rust.tsv')
    assert expected.keys() == numpy.keys() == rust.keys()
    records = []
    failures = []
    negative = 0
    for key, row in expected.items():
        record = dict(id=key)
        assert {'scalar','threads1','threads8'} <= set(rust[key]['bitwise_checks'].split(','))
        for metric in ['ms_gmsd','ms_gmsdc']:
            a,b = float(numpy[key][metric]),float(rust[key][metric])
            absolute,relative = delta(a,b)
            passed = absolute <= 1e-12 if a == 0 else relative <= 1e-9
            record[metric] = dict(numpy=a,rust=b,absolute=absolute,relative=relative,passed=passed)
            if not passed:
                failures.append(key+':'+metric)
        a = float(numpy[key]['ms_gmsdc'])
        da,dr = delta(a,float(rust[key]['wrong_constant']))
        negative += int(da > 1e-12 if a == 0 else dr > 1e-9)
        w,h = int(row['width']),int(row['height'])
        record['maps'] = []
        for scale in range(4):
            ap = ROOT/'ms_numpy'/f'{key}.s{scale}.f64'
            bp = ROOT/'rust'/f'{key}.ms_s{scale}.f64'
            av,bv = values(ap),values(bp)
            assert len(av) == len(bv) == w*h
            max_abs,max_rel = 0.0,0.0
            for a,b in zip(av,bv):
                da,dr = delta(a,b)
                max_abs = max(max_abs,da)
                if abs(a)>1e-12:
                    max_rel = max(max_rel,dr)
            passed = max_abs<=1e-12 and max_rel<=1e-9
            record['maps'].append(dict(scale=scale,samples=w*h,max_abs=max_abs,
                                      max_rel=max_rel,passed=passed,
                                      numpy_sha256=sha(ap),rust_sha256=sha(bp)))
            if not passed:
                failures.append(f'{key}:s{scale}')
            w,h = (w+1)//2,(h+1)//2
        records.append(record)
    maxima = {metric: dict(max_abs=max(r[metric]['absolute'] for r in records),
                          max_rel=max(r[metric]['relative'] or 0 for r in records))
              for metric in ['ms_gmsd','ms_gmsdc']}
    result = dict(schema='gmsd-chroma-ms-paper-parity-v1',oracle='independent NumPy paper transcription',
                  author_software_parity=False,pairs=len(records),failures=failures,
                  negative_control_failures=negative,passed=not failures and negative>0,
                  maxima=maxima,records=records,inputs_sha256=sha(ROOT/'oracle_inputs.tsv'),
                  binary_sha256=build['binary_sha256']['mdsi_oracle'],
                  source_sha256=build['source_sha256'])
    (ROOT/'ms_parity.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},sort_keys=True))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
