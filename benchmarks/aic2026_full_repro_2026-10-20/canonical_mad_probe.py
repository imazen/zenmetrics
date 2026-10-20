#!/usr/bin/env python3
"""Observe MAD components and replay only MATLAB's documented comparisons.

This is a separate diagnostic. It never edits author sources, production
manifests/results, the supplied binary or numerical acceptance bounds.
The replay is labeled explicitly; it is not a native MATLAB measurement.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

from canonical_chunks import scoring_environment
from canonical_octave import quote
from canonical_run import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('identities', nargs='+')
    p.add_argument('--source-bindings', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('retain prior diagnostic; supply a new directory')
    m = json.loads(a.manifest.read_text())
    config = json.loads(Path(m['config']).read_text())
    spec = config['octave']['mad']
    sources = json.loads(a.source_bindings.read_text())
    for path in spec['functions'].values():
        if path not in sources or sha(path) != sources[path]:
            raise ValueError('unbound or changed author function: '+path)
    with Path(m['pairs']).open() as stream:
        rows = {row['distorted']: row for row in csv.DictReader(stream, delimiter='\t')}
    selected = [rows[key] for key in a.identities]
    if len(set(a.identities)) != len(selected):
        raise ValueError('duplicate requested diagnostic identity')
    a.output.mkdir(parents=True)
    native_path = Path(m['output'])/'mad.native.tsv'
    with native_path.open() as stream:
        native = {row['distorted']: row for row in csv.DictReader(stream, delimiter='\t')
                  if row['distorted'] in a.identities}
    original = 'hi=hi_index(reference,distorted);'
    if spec['call'].count(original) != 1:
        raise ValueError('expected one explicit primary HI call')
    call = spec['call'].replace(original, '[hi,mp,imgs]=hi_index(reference,distorted);')
    result = a.output/'observations.jsonl'
    raw = a.output/'observations.tsv'
    strings = ['distorted','hi_author','lo_author','sig_author','score_author','replay_label','replay_hi','replay_score']
    counts = ['negative_mean_cells','complex_reference_contrast','complex_distortion_contrast',
              'reference_contrast_imag_cells','distortion_contrast_imag_cells',
              'original_interior_inf_cells','replay_interior_inf_cells',
              'octave_gt_first','octave_gt_second','real_gt_first','real_gt_second']
    columns = strings+counts
    script = ["more off; pkg load image; fftw('threads',1);"]
    script += [f"addpath({quote(path)},'-begin');" for path in spec['paths']]
    script += [f'assert(strcmp(which({quote(name)}),{quote(path)}));' for name,path in spec['functions'].items()]
    script += [f"out=fopen({quote(raw)},'w'); assert(out>=0);"]
    header = '\\t'.join(columns)+'\\n'
    script += [f"fprintf(out,{quote(header)});"]
    fmt = '\\t'.join(['%s']*len(strings)+['%.17g']*len(counts))+'\\n'
    arguments = ','.join('obs.'+name for name in columns)
    for row in selected:
        script += [f"R=imread({quote(row['ref_path'])}); D=imread({quote(row['dist_path'])});",
            'assert(isequal(size(R),[944 840 3])&&isequal(size(D),size(R)));',
            spec['ingress'], call,
            '[std2,std1,mu]=ical_std(imgs.dst-imgs.ref,imgs.ref);',
            'cr=log(std1./mu); cd=log(std2./mu); cd(find(mu<.5))=-inf;',
            # Preserve the primary's contrast arithmetic and assignments.
            # Only comparison operands are projected to their real parts,
            # matching MATLAB gt/le documentation. No clipping is introduced.
            'idx1=find((real(cr)>-5)&(real(cd)>real(1*(cr-(-5))+(-5))));',
            'idx2=find((real(cr)<=-5)&(real(cd)>-5));',
            'mask=zeros(size(cd));',
            'mask(idx1)=(cd(idx1)-(1*(cr(idx1)-(-5))+(-5)))/1;',
            'mask(idx2)=(cd(idx2)-(-5))/1;',
            "lmse=imfilter((double(reference)-double(distorted)).^2,ones(16)/16^2,'symmetric','same','conv');",
            'replay_mp=mask.*lmse; inside=replay_mp(17:end-17,17:end-17);',
            'replay_hi=norm(inside(:),2)/sqrt(length(inside(:)))*10;',
            'replay_sig=1/(1+exp(-2.55/3.35)*replay_hi^(1/(log(10)*3.35)));',
            'replay_score=replay_hi^replay_sig*lo^(1-replay_sig);',
            'original_inside=mp(17:end-17,17:end-17);',
            f"obs=struct('distorted',{quote(row['distorted'])},'hi_author',num2str(hi,17),'lo_author',num2str(lo,17),'sig_author',num2str(sig,17),'score_author',num2str(score,17));",
            "obs.replay_label='Octave replay of MATLAB documented comparison semantics; no MATLAB execution';",
            "obs.replay_hi=num2str(replay_hi,17);obs.replay_score=num2str(replay_score,17);",
            'obs.negative_mean_cells=sum(mu(:)<0);obs.complex_reference_contrast=~isreal(cr);obs.complex_distortion_contrast=~isreal(cd);',
            'obs.reference_contrast_imag_cells=sum(imag(cr(:))~=0);obs.distortion_contrast_imag_cells=sum(imag(cd(:))~=0);',
            'obs.original_interior_inf_cells=sum(isinf(original_inside(:)));obs.replay_interior_inf_cells=sum(isinf(inside(:)));',
            'probe=complex([-inf,1],[0,1]);octave_gt=probe>-5;real_gt=real(probe)>-5;',
            'obs.octave_gt_first=octave_gt(1);obs.octave_gt_second=octave_gt(2);obs.real_gt_first=real_gt(1);obs.real_gt_second=real_gt(2);',
            f'fprintf(out,{quote(fmt)},{arguments});fflush(out);clear obs;',
        ]
    script += ['assert(fclose(out)==0);']
    entry = a.output/'probe.m'
    entry.write_text('\n'.join(script)+'\n')
    env = scoring_environment(m, m['metrics']['mad'], Path(m['scratch']))
    child_env = dict(os.environ, **env)
    for name in ('DISPLAY','WAYLAND_DISPLAY','XAUTHORITY'):
        child_env.pop(name, None)
    command = ['octave-cli','--no-gui','--no-window-system','--no-history','--no-init-file','--quiet',str(entry)]
    receipt = dict(manifest_sha256=sha(a.manifest), config_sha256=sha(m['config']), worker_sha256=sha(__file__),
        primary_functions={path:sha(path) for path in spec['functions'].values()},
        source_bindings_sha256=sha(a.source_bindings), script_sha256=sha(entry),
        native_result_sha256=sha(native_path), native_rows=native, environment=env, command=command,
        inputs={str(path):sha(path) for row in selected for path in (row['ref_path'],row['dist_path'])},
        documentation=['https://docs.octave.org/v6.3.0/Comparison-Ops.html',
                       'https://www.mathworks.com/help/matlab/ref/double.gt.html'])
    with (a.output/'probe.log').open('x') as log:
        completed = subprocess.run(command, env=child_env, stdout=log, stderr=subprocess.STDOUT)
    receipt['returncode'] = completed.returncode
    receipt['raw_result_sha256'] = sha(raw) if raw.exists() else None
    if completed.returncode == 0:
        with raw.open() as stream:
            observed = list(csv.DictReader(stream, delimiter='\t'))
        if [row['distorted'] for row in observed] != a.identities:
            raise ValueError('diagnostic omitted or changed identities')
        for row in observed:
            for name in counts:
                row[name] = int(float(row[name]))
        with result.open('x') as stream:
            for row in observed:
                stream.write(json.dumps(row)+'\n')
    receipt['result_sha256'] = sha(result) if result.exists() else None
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if completed.returncode:
        raise SystemExit(completed.returncode)
    for line in result.read_text().splitlines():
        print(line)


if __name__ == '__main__':
    main()
