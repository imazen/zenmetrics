#!/usr/bin/env python3
"""Admitted-only original/resident HDR controls; never promotes timing scope.

Requires the caller's shared locks, run-heavy admission and disk TMPDIR. Runs
preflight first, then requested score pairs in fresh processes in ABBA order.
No native metric is executed. Every attempt and failure remains on disk.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct

from first_boundary_hdrvdp_resident import sha, verify, write
from first_boundary_hdrvdp_admitted import capture, execute


def scalar_bits(text, samples, resident):
    rows=[line.split() for line in text.splitlines() if line.startswith(('COLD ','SAMPLE '))]
    hexadecimal=[line.split() for line in text.splitlines() if line.startswith('SCORE_HEX ')]
    if len(rows)!=samples or len(hexadecimal)!=samples:
        raise ValueError('complete original scalar/hex observations required')
    bits=[]
    for row,hex_row in zip(rows,hexadecimal):
        if len(row)!=3 or len(hex_row)!=2:
            raise ValueError('malformed scalar/hex observation')
        elapsed,score=float(row[1]),float(row[2])
        if not math.isfinite(elapsed) or elapsed<=0 or not math.isfinite(score):
            raise ValueError('nonfinite score or invalid evaluator duration')
        value=struct.pack('>d',score).hex()
        if value!=hex_row[1].lower():
            raise ValueError('decimal and original scalar bits differ')
        bits.append(value)
    if len(set(bits))!=1:
        raise ValueError('score bits vary within original evaluation process')
    if resident and text.splitlines().count('PARAMETER_CACHE_UNCHANGED passed=1')!=1:
        raise ValueError('post-evaluation read-only array check missing')
    return bits[0]


def preflight_ok(item, text):
    lines=text.splitlines()
    if item['kind']=='arrays':
        arrays=[line for line in lines if line.startswith('PARAMETER_ARRAY_EXACT ')]
        return (len(arrays)==item['keys'] and
                lines.count(f"PARAMETER_ARRAY_COMPLETE keys={item['keys']} passed=1")==1)
    return lines.count(f"PARAMETER_MISSING_EXACT index={item['index']} passed=1")==1


def run(plan_path, output, panel, preflight_only):
    plan_path=plan_path.resolve(strict=True)
    plan_bytes=plan_path.read_bytes()
    plan=json.loads(plan_bytes)
    bundle=plan_path.parent
    verify(plan['bindings'])
    if (plan['schema']!=1 or len(plan['cases'])!=104 or len(plan['preflight'])!=9
            or plan['qualification']!='unmatched_scope' or plan['matched_scope'] is not False
            or plan['expected_control_order']!=['default','resident','resident','default']
            or len({case['id'] for case in plan['cases']})!=104
            or {case['threads'] for case in plan['cases']}!={1,28}):
        raise ValueError('explicit complete diagnostic resident plan required')
    output.mkdir()
    root=output.resolve()
    (root/'scratch').mkdir()
    (root/'plan.json').write_bytes(plan_bytes)
    result=dict(schema=1,plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        controller_sha256=sha(__file__),variant=plan['variant'],panel=panel,
        preflight=[],cases=[],passed=False,matched_scope=False,
        qualified_timing_cells=0,qualification='unmatched_scope',
        limit='Exact original/resident control only; runtime/native numerical and remaining scope qualification are separate')
    baseline=None

    def child(command, name, threads, check):
        nonlocal baseline
        verify(plan['bindings'])
        if plan_path.read_bytes()!=plan_bytes:
            raise ValueError('resident plan changed')
        if list((bundle/'execution-cwd').iterdir()):
            raise ValueError('frozen evaluator working directory is not empty')
        directory=root/name;directory.mkdir()
        scratch=directory/'scratch';scratch.mkdir()
        env=dict(os.environ,TMPDIR=str(scratch),LC_ALL='C')
        for key in ('RAYON_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS','RTEN_NUM_THREADS'):
            env[key]=str(threads)
        overrides=[key for key in ('OCTAVE_PATH','OCTAVE_EXEC_PATH','OCTAVE_HOME',
            'OCTAVE_SITE_INITFILE','OCTAVE_VERSION_INITFILE','LD_PRELOAD','LD_AUDIT') if env.get(key)]
        if overrides:
            raise ValueError('unbound runtime overrides: '+', '.join(overrides))
        before=capture(plan,bundle,directory/'runtime-before',env)
        invariant=dict(files=before['files'],dependencies=before['dependencies'])
        if baseline is None:
            baseline=invariant
        elif invariant!=baseline:
            raise ValueError('runtime/source closure changed before child')
        observation=execute(command,env,directory,'control',plan['seconds_per_stage'])
        # Save child outcome before any post-attempt verification can fail.
        write(directory/'exit.json',observation)
        verify(plan['bindings'])
        after=capture(plan,bundle,directory/'runtime-after',env)
        if before!=after:
            raise ValueError('runtime/source closure changed during child')
        text=(directory/'control-stdout.log').read_text()
        if observation['returncode']!=0:
            observation.update(passed=False,error='original control process exited nonzero')
        else:
            try:
                observation.update(check(text),passed=True)
            except Exception as error:
                observation.update(passed=False,error=str(error))
        write(directory/'result.json',observation)
        return observation

    try:
        for item in plan['preflight']:
            def check_preflight(text, item=item):
                if not preflight_ok(item,text):
                    raise ValueError('required exact array/key refusal marker missing')
                return dict(control=item['id'])
            observed=child(item['command'],'preflight-'+item['id'],1,check_preflight)
            result['preflight'].append(dict(id=item['id'],**observed))
            print('RESIDENT_PREFLIGHT '+json.dumps(result['preflight'][-1]),flush=True)
        if not all(row['passed'] for row in result['preflight']):
            raise ValueError('resident preflight failed; score-pair controls refused')
        if not preflight_only:
            for case in plan['cases']:
                if panel!='all' and case['panel']!=panel:
                    continue
                row=dict(id=case['id'],metric=case['metric'],phase=case['phase'],
                         threads=case['threads'],observations=[])
                for ordinal,role in enumerate(plan['expected_control_order']):
                    def check_score(text, role=role, case=case):
                        return dict(score_bits=scalar_bits(text,case['samples'],role=='resident'))
                    observed=child(case['arms'][role]['command'],case['id']+f'-{ordinal}-{role}',
                                   case['threads'],check_score)
                    row['observations'].append(dict(role=role,ordinal=ordinal,**observed))
                accepted=[v['score_bits'] for v in row['observations'] if v['passed']]
                row['passed']=len(accepted)==4 and len(set(accepted))==1
                result['cases'].append(row)
                write(root/(case['id']+'-result.json'),row)
                print('RESIDENT_EXACT_CONTROL '+json.dumps(row),flush=True)
        expected=0 if preflight_only else (104 if panel=='all' else (8 if panel=='photo512' else 96))
        if len(result['cases'])!=expected:
            raise ValueError('requested case inventory incomplete')
        result['passed']=(len(result['preflight'])==9 and all(row['passed'] for row in result['preflight'])
                          and all(row['passed'] for row in result['cases']))
        result['coverage']=dict(score_cases_completed=len(result['cases']),score_cases_required=104,
                                preflight_only=preflight_only)
        verify(plan['bindings'])
        if plan_path.read_bytes()!=plan_bytes:
            raise ValueError('resident plan changed during controls')
    except Exception as error:
        result['passed']=False
        result['error']=str(error)
    write(root/'result.json',result)
    print('RESIDENT_CONTROL_RESULT '+json.dumps(result),flush=True)
    return 0 if result['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--panel',choices=('all','photo512','original12'),default='all')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    raise SystemExit(run(args.plan,args.output,args.panel,args.preflight_only))


if __name__=='__main__':main()
