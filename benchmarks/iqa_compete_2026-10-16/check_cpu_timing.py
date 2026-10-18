#!/usr/bin/env python3
"""Exercise PSNR refusal and GMSD scalar/map cold/warm identity at frozen 512."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from prepare_durable import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--lock', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    bundle, results = a.output / 'bundle', a.output / 'results'
    subprocess.run([sys.executable, str(Path(__file__).with_name('select_durable.py')),
        str(a.bundle), str(bundle), '--metrics', 'psnr,gmsd', '--sizes', 'cid22', '--threads', '1'], check=True)
    command = [sys.executable, str(bundle / 'durable.py'), 'run', str(bundle / 'manifest.json'),
        str(results), '--run-heavy', str(bundle / 'tools/run-heavy'), '--lock', str(a.lock),
        '--memory', '12G', '--jobs', '8', '--timeout-seconds', '120', '--max-attempts', '1']
    with (a.output / 'controller.log').open('xb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads((results / 'report.json').read_bytes())
    if not report['complete']:
        raise RuntimeError('CPU control incomplete')
    bits = {}
    for cell in report['cells']:
        c, row = cell['config'], cell['row']
        expected = 'unsupported' if (c['metric_name'], c['only_mode']) == ('psnr', 'score+map') else 'ok'
        if cell['status'] != expected:
            raise RuntimeError('CPU control outcome differs: ' + str(c))
        if expected == 'ok':
            bits.setdefault((c['metric_name'], c['only_mode']), set()).add(row['score_bits'])
    if len(bits) != 3 or any(len(values) != 1 for values in bits.values()):
        raise RuntimeError('CPU cold/warm score bits differ')
    if bits['gmsd', 'score'] != bits['gmsd', 'score+map']:
        raise RuntimeError('GMSD map/scalar bits differ')
    receipt = dict(cells=len(report['cells']), report_sha256=sha(results / 'report.json'),
        binary_sha256=json.loads((bundle / 'manifest.json').read_bytes())['binary_sha256'],
        score_bits={metric + '/' + mode: next(iter(value)) for (metric, mode), value in bits.items()},
        qualification='Frozen 512 CPU timing protocol and exact cold/warm/map identity; not independent metric parity')
    (a.output / 'control.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
