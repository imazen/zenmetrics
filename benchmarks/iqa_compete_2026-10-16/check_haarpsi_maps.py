#!/usr/bin/env python3
"""Compare retained HaarPSI RGB maps with weighted primary orientation maps.

Builds only a probe against existing frozen rlibs, without changing the metric.
The primary's original similarity and weight arrays are retained separately.
There is no documented external map bound: differences remain observations.
Caller supplies both heavy locks, run-heavy and a complete log.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np

from haarpsi_primary import execute
from octave_primary import sha, quoted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['build','source','inputs','output']:
        p.add_argument('--'+name, type=Path, required=True)
    identity_source = p.add_mutually_exclusive_group(required=True)
    identity_source.add_argument('--panel', type=Path, help='Historical scalar-panel identity')
    identity_source.add_argument('--worker-plan', type=Path, help='Current frozen scalar worker plan')
    p.add_argument('--threads', help='Explicit thread counts for a current worker plan')
    a = p.parse_args()
    a.source = a.source.resolve(strict=True); a.inputs = a.inputs.resolve(strict=True)
    if a.worker_plan:
        if not a.threads:
            p.error('--worker-plan requires explicit --threads')
        worker = json.loads(a.worker_plan.read_text())
        for path, expected in worker['bindings'].items():
            assert sha(Path(path)) == expected, 'Frozen worker binding changed: ' + path
        assert str(a.source) in worker['bindings'], 'Original map source absent from worker bindings'
        receipt_path = a.worker_plan
        original = dict(source_revision=worker['rust_source'])
        identity = dict(source_sha256=worker['bindings'][str(a.source)],
                        inputs_sha256=sha(a.inputs), threads=list(map(int, a.threads.split(','))))
        assert identity['threads'] and all(t > 0 for t in identity['threads'])
    else:
        if a.threads:
            p.error('--threads is supplied by the historical scalar-panel identity')
        receipt_path = a.build/'build-receipt.json'
        original = json.loads(receipt_path.read_text())
        identity = json.loads((a.panel/'identity.json').read_text())
        assert original['source_revision'] == 'e2d414c33a6d3c7da50a818fa36e56654d3ebeef'
    assert sha(a.source) == identity['source_sha256'] and sha(a.inputs) == identity['inputs_sha256']
    repo = Path(__file__).resolve().parents[2]
    crate = a.build/'work/zenmetrics/crates/haarpsi'
    adaptations = {}
    if a.worker_plan:
        metadata = a.build/'luma-maps.meta.json'
        for entry in json.loads(metadata.read_text())['files']:
            adaptations[entry['path']] = entry
    for path in crate.rglob('*'):
        if path.is_file():
            relative = path.relative_to(a.build/'work/zenmetrics').as_posix()
            published = subprocess.check_output(['git','-C',str(repo),'show',original['source_revision']+':'+relative])
            if relative in adaptations:
                import hashlib
                entry = adaptations[relative]
                assert hashlib.sha256(published).hexdigest() == entry['original_sha256']
                assert sha(path) == entry['adapted_sha256']
            else:
                assert path.read_bytes() == published
    a.output.mkdir(); root=a.output.resolve()
    deps = a.build/'target/release/deps'
    libs = {}
    for name in ['haarpsi','rayon']:
        matches = list(deps.glob('lib'+name+'-*.rlib')); assert len(matches)==1
        libs[name] = matches[0].resolve()
    binary = root/'haarpsi-map'
    source = Path(__file__).with_name('haarpsi_primary_dump.rs').resolve()
    command = ['rustc','--edition=2024','-C','opt-level=3','-L','dependency='+str(deps.resolve()),
               *[v for k,path in libs.items() for v in ['--extern',k+'='+str(path)]],str(source),'-o',str(binary)]
    subprocess.run(command, check=True)
    receipt = dict(command=command,source_sha256=sha(source),binary_sha256=sha(binary),
                   rlibs={str(p):sha(p) for p in libs.values()},frozen_receipt_sha256=sha(receipt_path),
                   frozen_source=original['source_revision'],
                   luma_map_adapter_sha256=sha(a.build/'luma-maps.meta.json') if a.worker_plan else None)
    (root/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    rows = []
    for item in json.loads(a.inputs.read_text())['pairs']:
        w,h=item['width'],item['height']
        paths={r:(a.inputs.parent/item[r]).resolve(strict=True) for r in ['reference','distorted']}
        assert all(sha(path)==item[r+'_sha256'] for r,path in paths.items())
        for threads in identity['threads']:
            directory=root/f"{item['pair']}-{item['size_name']}-t{threads}";directory.mkdir()
            env=dict(os.environ, LC_ALL='C', TMPDIR=str(directory))
            for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']:env[key]=str(threads)
            for key in ['DISPLAY','WAYLAND_DISPLAY','XAUTHORITY']:env.pop(key,None)
            script=directory/'entry.m'
            script.write_text(f"""more off;
addpath({quoted(a.source.parent)},'-begin');assert(strcmp(which('HaarPSI'),{quoted(a.source)}));
f=fopen({quoted(paths['reference'])},'rb');r=fread(f,Inf,'*uint8');assert(fclose(f)==0);
f=fopen({quoted(paths['distorted'])},'rb');d=fread(f,Inf,'*uint8');assert(fclose(f)==0);
r=permute(reshape(r,[3,{w},{h}]),[3,2,1]);d=permute(reshape(d,[3,{w},{h}]),[3,2,1]);
[score,s,m]=HaarPSI(r,d);
f=fopen({quoted(directory/'similarities.f64')},'wb','ieee-le');fwrite(f,permute(s,[2,1,3]),'double');assert(fclose(f)==0);
f=fopen({quoted(directory/'weights.f64')},'wb','ieee-le');fwrite(f,permute(m,[2,1,3]),'double');assert(fclose(f)==0);
fprintf('MAP %d %d %d SCORE %.17g\\n',size(s,2),size(s,1),size(s,3),score);
""")
            rc,_=execute(['octave-cli','--no-gui','--no-window-system','--no-history','--no-init-file','--quiet',str(script)],env,directory,'canonical',300)
            rr,_=execute([str(binary),str(w),str(h),str(paths['reference']),str(paths['distorted']),str(threads),str(directory/'rust.f32')],env,directory,'rust',120)
            assert rc==rr==0
            line=next(s for s in (directory/'canonical-stdout.log').read_text().splitlines() if s.startswith('MAP ')).split()
            mw,mh,n=map(int,line[1:4]); score=float(line[5]); assert n==3
            rustline=next(s for s in (directory/'rust-stdout.log').read_text().splitlines() if s.startswith('HAARPSI ')).split()
            assert list(map(int,rustline[3:]))==[mw,mh]
            if a.panel:
                prior=json.loads((a.panel/f"haarpsi-{item['pair']}-{item['size_name']}-t{threads}"/'results.json').read_text())
                assert score==prior['reference_values'][0] and float(rustline[1])==prior['ours']
            s=np.fromfile(directory/'similarities.f64',dtype='<f8').reshape(n,mh,mw)
            m=np.fromfile(directory/'weights.f64',dtype='<f8').reshape(n,mh,mw)
            with np.errstate(invalid='ignore',divide='ignore'):canonical=(s*m).sum(axis=0)/m.sum(axis=0)
            ours=np.fromfile(directory/'rust.f32',dtype='<f4').reshape(mh,mw)
            finite=np.isfinite(canonical)&np.isfinite(ours)
            matching_nonfinite=(np.isnan(canonical)&np.isnan(ours)) | (np.isinf(canonical)&(canonical==ours))
            row=dict(pair=item['pair'],size=item['size_name'],threads=threads,width=mw,height=mh,
                     rust_source=original['source_revision'], rust_score=float(rustline[1]), canonical_score=score,
                     finite_pixels=int(finite.sum()),matching_nonfinite_pixels=int(matching_nonfinite.sum()),
                     nonfinite_mismatches=int((~finite&~matching_nonfinite).sum()),
                     max_abs_delta=float(np.abs(ours[finite]-canonical[finite]).max()) if finite.any() else None,
                     mean_abs_delta=float(np.abs(ours[finite]-canonical[finite]).mean()) if finite.any() else None,
                     bound='UNBOUND',qualified_ratio=False,
                     files={p.name:sha(p) for p in directory.iterdir() if p.is_file()})
            (directory/'results.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
            print('MAP_CASE',directory.name,row['max_abs_delta'],row['nonfinite_mismatches'],flush=True)
            assert sha(a.source)==identity['source_sha256'] and sha(binary)==receipt['binary_sha256']
    (root/'completion.json').write_text(json.dumps(dict(cases=rows,source_sha256=sha(a.source),
        inputs_sha256=sha(a.inputs),build_receipt_sha256=sha(root/'build-receipt.json'),worker_sha256=sha(Path(__file__)),
        map_scope='weighted mean of original primary orientation/chroma similarities using original weights; no logistic applied',
        bound='UNBOUND',qualified_timing_cells=0),indent=2)+'\n')


if __name__=='__main__':main()
