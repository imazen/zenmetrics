#!/usr/bin/env python3
"""Separate unchanged author RGB plane calls into individual budgeted processes.

Keeps the original 300-second per-process limit. Recombines three per-plane
scalar pairs in the original RGB mean order, first checking completed small
controls. Original whole-RGB timeouts are retained. No metric gate changes.
Controller contains each plane call with both heavy locks and run-heavy.
Caller saves the complete controller log.
"""
import argparse
import json
import os
from pathlib import Path
import time

from contained_primary import execute
from octave_primary import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('panel',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--wait-for-panel',action='store_true')
    a=p.parse_args();a.panel=a.panel.resolve(strict=True)
    if a.wait_for_panel:
        deadline=time.monotonic()+7200
        print('WAIT_PANEL',str(a.panel/'completion.json'),flush=True)
        while not (a.panel/'completion.json').is_file():
            assert time.monotonic()<deadline, 'panel completion wait budget exhausted'
            time.sleep(1)
    identity=json.loads((a.panel/'identity.json').read_text())
    assert identity['absolute_bound']==1e-3 and identity['compat_warning_policy']=='off:Octave:possible-matlab-short-circuit-operator'
    original=json.loads((a.panel/'completion.json').read_text())['cases']
    controls=[r for r in original if r['metric']=='psnrhvs' and r['pair']=='pairA' and r['size']=='64']
    targets=[r for r in original if r['metric']=='psnrhvs' and r['canonical_returncode']==124]
    assert len(controls)==2 and all(r['size']=='4k' and r['rust_returncode']==0 for r in targets)
    a.output.mkdir();root=a.output.resolve();rows=[]
    for item in controls+targets:
        case=f"psnrhvs-{item['pair']}-{item['size']}-t{item['threads']}"
        old=Path(item.get('origin_directory',str(a.panel/case))).resolve(strict=True)
        directory=root/case;directory.mkdir()
        prior=(old/'entry.m').read_text()
        needle="for c=1:size(r,3);[masked,unmasked]=psnrhvsm(r(:,:,c),d(:,:,c));values(c,:)=[unmasked,masked];end"
        assert prior.count(needle)==1 and prior.count('values=zeros(size(r,3),2);')==1
        values=[];calls=[]
        for channel in range(1,4):
            part=directory/f'channel{channel}';part.mkdir()
            text=prior.replace('values=zeros(size(r,3),2);','values=zeros(1,2);').replace(needle,
                f'for c={channel};[masked,unmasked]=psnrhvsm(r(:,:,c),d(:,:,c));values(1,:)=[unmasked,masked];end')
            entry=part/'entry.m';entry.write_text(text)
            env=dict(os.environ,LC_ALL='C',TMPDIR=str(part))
            for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']:env[key]=str(item['threads'])
            for key in ['DISPLAY','WAYLAND_DISPLAY','XAUTHORITY']:env.pop(key,None)
            assert all(sha(Path(path))==identity['source_sha256'][name] for name,path in identity['sources'].items())
            rc,command=execute([identity['octave'],'--no-gui','--no-window-system','--no-history','--no-init-file','--quiet',str(entry)],env,part,'canonical',300)
            scored=[list(map(float,s.split()[1:])) for s in (part/'canonical-stdout.log').read_text().splitlines() if s.startswith('SCORES ')]
            if rc==0:
                assert len(scored)==1 and len(scored[0])==2;values.append(scored[0])
            else:assert not scored
            calls.append(dict(channel=channel,returncode=rc,command=command,
                files={p.name:sha(p) for p in part.iterdir() if p.is_file()}))
            print('PLANE_DONE',case,channel,rc,flush=True)
        row=dict(case=case,control=item in controls,calls=calls,original_result_sha256=sha(old/'results.json'),
            original_entry_sha256=sha(old/'entry.m'),bound=1e-3,qualified_ratio=False,status='process-finding')
        if len(values)==3:
            scores=[(values[0][c]+values[1][c]+values[2][c])/3 for c in range(2)]
            if item in controls:
                assert scores==[r['canonical'] for r in item['scores']], 'split-plane RGB recombination changed the control bits'
                ours=[r['ours'] for r in item['scores']]
            else:
                import struct
                probes=[json.loads(s.removeprefix('IQA_PROBE ')) for s in (old/'rust-stdout.log').read_text().splitlines() if s.startswith('IQA_PROBE ')]
                assert len(probes)==1 and len(probes[0]['scores'])==2
                ours=[struct.unpack('>d',bytes.fromhex(r[1]))[0] for r in probes[0]['scores']]
            delta=max(abs(x-y) for x,y in zip(ours,scores,strict=True))
            row.update(canonical=scores,ours=ours,max_abs_delta=delta,status='pass' if delta<=1e-3 else 'numeric-finding')
        (directory/'result.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
        # A failed control is never used as permission to continue a supplement.
        if item in controls:assert row['status']=='pass'
    (root/'completion.json').write_text(json.dumps(dict(identity_sha256=sha(a.panel/'identity.json'),
        source_completion_sha256=sha(a.panel/'completion.json'),worker_sha256=sha(Path(__file__)),cases=rows,
        qualified_timing_cells=0),indent=2)+'\n')


if __name__=='__main__':main()
