#!/usr/bin/env python3
"""Run the ordered crop study; preserve every attempt and publish local progress."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def groups(manifest):
    learned = ['maniqa', 'musiq', 'liqe', 'hyperiqa', 'arniqa', 'paq2piq',
               'wadiqam-fr-live', 'wadiqam-fr-tid', 'wadiqam-nr-live', 'wadiqam-nr-tid',
               'lpips-squeeze', 'pieapp', 'dmm', 'ahiq']
    nr = ['brisque', 'niqe', 'piqe']
    classical = [name for name in manifest['metrics'] if name not in learned+nr]
    return [('classical', classical), ('addendum', list(manifest['addendum'])), ('nr', nr), ('learned', learned)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--describe', action='store_true')
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    stages = groups(manifest)
    if a.describe:
        print(json.dumps(stages, indent=2))
        return
    published = manifest.get('published_comparison')
    if not published or not all(key in published for key in ('path', 'sha256')):
        raise ValueError('manifest requires published_comparison.path and .sha256 for the cropped union')
    here = Path(__file__).parent
    root = Path(manifest['scratch'])
    for stage, names in stages:
        command = [sys.executable, str(here/'canonical_run.py'), str(a.manifest), '--metrics', *names]
        if stage == 'addendum':
            command.append('--addendum')
        print('starting '+stage, flush=True)
        log = root/(stage+'.campaign.log')
        if log.exists():
            raise ValueError('existing campaign log must be retained under a new attempt identity: '+str(log))
        with log.open('x') as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
        compare = ('compare_fsim_vsi.py' if stage == 'addendum' else 'canonical_compare.py')
        aggregate = root/('fsim_vsi_implementation.tsv' if stage == 'addendum' else 'compare_canonical_cropped.tsv')
        command = [manifest['statistics_python'], str(here/compare), str(a.manifest)]
        command += [str(aggregate)] if stage == 'addendum' else ['--summary', str(aggregate)]
        if stage == 'addendum':
            command += ['--published-reference', published['path'], '--reference-sha256', published['sha256']]
        with (root/(stage+'.aggregate.log')).open('x') as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
        with (root/'report.md').open('a') as report:
            report.write('\nCompleted execution stage: '+stage+'. Aggregate: '+str(aggregate)+'.\n')
            report.write('```tsv\n'+aggregate.read_text()+'```\n')
        (root/(stage+'.stage-complete')).write_text('execution and aggregation completed; inspect coverage and refusals\n')
        print('completed '+stage, flush=True)


if __name__ == '__main__':
    main()
