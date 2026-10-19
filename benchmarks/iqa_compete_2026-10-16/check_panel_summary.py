#!/usr/bin/env python3
"""Negative controls for publishing a retained scalar panel."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('panel', type=Path)
    p.add_argument('evidence', type=Path)
    a = p.parse_args()
    a.evidence.mkdir()
    evidence = a.evidence.resolve()
    source = a.panel.resolve(strict=True)
    identity = json.loads((source / 'identity.json').read_text())
    sample = sorted(source.glob('*/outcome.json'))[0]
    outcome = json.loads(sample.read_text())
    identity['input_selection'] = [r for r in identity['input_selection']
                                   if r['pair'] == outcome['pair'] and r['size_name'] == outcome['size']]
    identity['families'] = [outcome['family']]
    identity['factors'] = [outcome['factor']]
    identity['threads'] = [outcome['threads']]
    checks = []
    for control in ['incomplete', 'corrupt-artifact', 'wrong-comparison']:
        root = evidence / control
        root.mkdir()
        write(root / 'identity.json', identity)
        write(root / 'summary.json', {'execution_complete': control != 'incomplete',
                                     'counts': {outcome['status']: 1}})
        if control != 'incomplete':
            directory = root / sample.parent.name
            shutil.copytree(sample.parent, directory)
            if control == 'corrupt-artifact':
                with (directory / 'canonical/stdout.log').open('a') as f:
                    f.write('\nDELIBERATE NEGATIVE-CONTROL MUTATION\n')
            else:
                altered = json.loads((directory / 'outcome.json').read_text())
                altered['comparison']['comparisons'][0]['ours'] += 0.1
                write(directory / 'outcome.json', altered)
        output = evidence / (control + '-must-remain-absent')
        command = [sys.executable, '-B', str(Path(__file__).with_name('summarize_octave_panel.py')),
                   str(root), str(output), '--label', control]
        with (evidence / (control + '.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode != 0 and not output.exists(), control
        message = (evidence / (control + '.log')).read_text()
        expected = {'incomplete': 'panel is incomplete', 'corrupt-artifact': 'panel artifact changed',
                    'wrong-comparison': 'AssertionError'}[control]
        assert expected in message, (control, message)
        checks.append(control + ': refused before output creation')
    write(evidence / 'results.json', {'controls': checks, 'count': len(checks),
                                     'scope': 'artifact integrity; no scorer executed'})
    print(json.dumps(checks))


if __name__ == '__main__':
    main()
