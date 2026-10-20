#!/usr/bin/env python3
"""Generate one persistent Octave evaluation of unchanged author functions."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess


def quote(value):
    return "'"+str(value).replace("'", "''")+"'"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('metric')
    p.add_argument('pairs', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    spec = cfg['octave'][a.metric]
    rows = list(csv.DictReader(a.pairs.open(), delimiter='\t'))
    if a.limit is not None:
        rows = rows[:a.limit]
    prior = set()
    if a.output.exists():
        prior = {r['distorted'] for r in csv.DictReader(a.output.open(), delimiter='\t')}
    script = ["more off; pkg load image; fftw('threads',1);",
              "warning('off','Octave:possible-matlab-short-circuit-operator');"]
    for path in spec['paths']:
        script.append(f"addpath({quote(path)},'-begin');")
    for name, path in spec['functions'].items():
        script.append(f'assert(strcmp(which({quote(name)}),{quote(path)}));')
    script += [f"out=fopen({quote(a.output)},'a'); assert(out>=0);"]
    if not a.output.exists():
        script += ["fprintf(out,'distorted\\tvalue\\terror\\n'); fflush(out);"]
    ref_path = None
    for index, row in enumerate(rows):
        if row['distorted'] in prior:
            continue
        if ref_path != row['ref_path']:
            ref_path = row['ref_path']
            script.append(f'R=imread({quote(ref_path)}); assert(isequal(size(R),[944 840 3]));')
        script += [f'D=imread({quote(row["dist_path"])}); assert(isequal(size(D),size(R)));',
                   'try', spec['ingress'], spec['call'],
                   "assert(isreal(score)&&isfinite(score));",
                   f"fprintf(out,'%s\\t%.17g\\t\\n',{quote(row['distorted'])},score);",
                   'catch err',
                   "message=strrep(strrep(err.message,sprintf('\\n'),' '),sprintf('\\t'),' ');",
                   f"fprintf(out,'%s\\t\\t%s\\n',{quote(row['distorted'])},message);",
                   'end; fflush(out); clear reference distorted score;']
        if index % 100 == 0:
            script += [f"fprintf('{a.metric}: {index+1}/{len(rows)}\\n'); fflush(stdout);"]
    script += ['assert(fclose(out)==0);']
    entry = a.output.with_suffix('.m')
    entry.write_text('\n'.join(script)+'\n')
    env = dict(os.environ)
    for key in ('DISPLAY', 'WAYLAND_DISPLAY', 'XAUTHORITY'):
        env.pop(key, None)
    subprocess.run(['octave-cli', '--no-gui', '--no-window-system', '--no-history',
                    '--no-init-file', '--quiet', str(entry)], env=env, check=True)


if __name__ == '__main__':
    main()
