#!/usr/bin/env python3
"""One admitted HDR arm, or passed-preflight sealing; no admission loop here.

Only the successor coordinator calls this worker, inside shared admission.
Every validation/hash/capture and evaluator call stays inside that admission.
Frozen original v3 scripts, metrics and preflight outputs are read only.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct

from first_boundary_hdrvdp_resident import sha, verify, write
from first_boundary_hdrvdp_resident_run import preflight_ok, scalar_bits
from first_boundary_hdrvdp_admitted import capture, execute


ORDER = ['default','resident','resident','default']
THREAD_ENV = {'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','RTEN_NUM_THREADS'}


def load(path):
    return json.loads(Path(path).read_text())


def files_under(root):
    found={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('receipt symlink refused: '+str(path))
        if path.is_file():
            found[str(path)]=sha(path)
        elif not path.is_dir():
            raise ValueError('nonregular receipt artifact: '+str(path))
    return found


def invariant(snapshot):
    return dict(files=snapshot['files'],dependencies=snapshot['dependencies'],
                environment={k:v for k,v in snapshot['environment'].items() if k not in THREAD_ENV})


def context(root, expected):
    path=root/'cohort.json'
    if sha(path)!=expected:
        raise ValueError('successor cohort hash differs from explicit admission argument')
    cohort=load(path)
    verify(cohort['bindings'])
    if sha(cohort['parent_plan'])!=cohort['parent_plan_sha256']:
        raise ValueError('original resident plan changed')
    plan=load(cohort['parent_plan'])
    if (plan['expected_control_order']!=ORDER or len(plan['cases'])!=104 or len(plan['preflight'])!=9
            or len({c['id'] for c in plan['cases']})!=104
            or {c['threads'] for c in plan['cases']}!={1,28}
            or plan['matched_scope'] is not False or plan['qualification']!='unmatched_scope'
            or plan['seconds_per_stage']!=1100):
        raise ValueError('unchanged complete resident protocol required')
    verify(plan['bindings'])
    if list((Path(cohort['parent_plan']).parent/'execution-cwd').iterdir()):
        raise ValueError('original evaluator working directory is not empty')
    selected_cases(cohort,plan)
    return cohort,plan


def selected_cases(cohort,plan):
    cases=load(cohort['case_specs'])
    if cohort['kind']=='original104':
        if cases!=plan['cases']:
            raise ValueError('original 104-case contract differs')
        return cases
    if cohort['kind']!='photo512-t8-supplement' or len(cases)!=4:
        raise ValueError('explicit original104 or four-case eight-thread supplement required')
    originals={case['id']:case for case in plan['cases']}
    identities=set()
    for case in cases:
        old=originals[case['derived_from_case_id']]
        if (old['panel']!='photo512' or old['threads']!=1 or old['width']!=512 or old['height']!=512
                or case['threads']!=8 or case['id']!=old['id'].replace('-t1-','-t8-',1)+'-supplement'):
            raise ValueError('eight-thread photo source/identity differs')
        for key in ('panel','metric','phase','width','height','samples','reference','distorted',
                    'absolute_bound','relative_bound','matched_scope','evaluator_body_sha256'):
            if case[key]!=old[key]:raise ValueError('supplement changed retained contract: '+key)
        for role in ('default','resident'):
            new_command=case['arms'][role]['command'];old_command=old['arms'][role]['command']
            if new_command[:-1]!=old_command[:-1] or sha(new_command[-1])!=sha(old_command[-1]):
                raise ValueError('eight-thread reference script/body changed')
        expected_native=dict(load(old['native_config']),threads=8)
        if load(case['native_config'])!=expected_native or case['native']['command']!=[old['binary'],case['native_config']]:
            raise ValueError('explicit eight-thread native configuration differs')
        if case['environment']!={key:'8' for key in old['environment']}:
            raise ValueError('eight-thread environment differs')
        identities.add((case['metric'],case['phase']))
    if identities!={('hdrvdp','first'),('hdrvdp','warm'),('hdrvdp3','first'),('hdrvdp3','warm')}:
        raise ValueError('both base metrics and first/warm required at eight threads')
    return cases


def validate_preflight(cohort, plan, expected):
    path=Path(cohort['preflight_receipt']).resolve(strict=True)
    if sha(path)!=expected:
        raise ValueError('preflight result hash differs from explicit seal argument')
    receipt=load(path)
    old_controller=Path(cohort['parent_plan']).parent/'helpers/first_boundary_hdrvdp_resident_run.py'
    if (receipt['passed'] is not True or receipt['plan_sha256']!=cohort['parent_plan_sha256']
            or receipt['controller_sha256']!=plan['bindings'][str(old_controller)]
            or receipt['variant']!=plan['variant'] or receipt['cases']!=[]
            or receipt['coverage']!=dict(score_cases_completed=0,score_cases_required=104,preflight_only=True)
            or receipt['qualified_timing_cells']!=0 or receipt['matched_scope'] is not False
            or receipt['qualification']!='unmatched_scope' or len(receipt['preflight'])!=9):
        raise ValueError('exact passed preflight-only receipt/provenance required')
    directory=path.parent
    if sha(directory/'plan.json')!=cohort['parent_plan_sha256']:
        raise ValueError('preflight retained plan differs')
    baseline=None
    for item,row in zip(plan['preflight'],receipt['preflight']):
        if (row['id']!=item['id'] or row['control']!=item['id'] or row['passed'] is not True
                or type(row['returncode']) is not int or row['returncode']!=0):
            raise ValueError('preflight identity or explicit passed gate differs')
        child=directory/('preflight-'+item['id'])
        if load(child/'result.json')!={k:v for k,v in row.items() if k!='id'}:
            raise ValueError('preflight nested receipt differs')
        for stream in ('stdout','stderr'):
            if sha(child/('control-'+stream+'.log'))!=row[stream+'_sha256']:
                raise ValueError('preflight process log hash differs')
        if not preflight_ok(item,(child/'control-stdout.log').read_text()):
            raise ValueError('preflight array/key gate missing from original output')
        before=load(child/'runtime-before/snapshot.json')
        after=load(child/'runtime-after/snapshot.json')
        if before!=after:
            raise ValueError('preflight before/after runtime closure differs')
        current=invariant(before)
        if baseline is None:
            baseline=current
        elif current!=baseline:
            raise ValueError('preflight runtime closure varies across controls')
    artifacts=files_under(directory)
    if artifacts.get(str(path))!=expected:
        raise ValueError('preflight changed during seal validation')
    return baseline,artifacts


def seal(root, expected_cohort, expected_preflight):
    cohort,plan=context(root,expected_cohort)
    baseline,artifacts=validate_preflight(cohort,plan,expected_preflight)
    verify(artifacts)
    context(root,expected_cohort)
    record=dict(schema=1,passed=True,cohort_sha256=expected_cohort,
        parent_plan_sha256=cohort['parent_plan_sha256'],preflight_sha256=expected_preflight,
        preflight_receipt=cohort['preflight_receipt'],artifacts=artifacts,baseline=baseline,
        controls=9,matched_scope=False,qualified_timing_cells=0)
    path=root/'preflight-seal.json'
    write(path,record)
    path.chmod(0o444)
    print(json.dumps(dict(seal=str(path),seal_sha256=sha(path),preflight_sha256=expected_preflight,
                          passed=True,controls=9)),flush=True)
    return 0


def checked_seal(root, cohort, expected_cohort, expected_seal):
    path=root/'preflight-seal.json'
    if sha(path)!=expected_seal:
        raise ValueError('preflight seal differs from explicit queue argument')
    seal=load(path)
    if (seal['passed'] is not True or seal['controls']!=9 or seal['cohort_sha256']!=expected_cohort
            or seal['parent_plan_sha256']!=cohort['parent_plan_sha256']
            or seal['preflight_receipt']!=cohort['preflight_receipt']
            or seal['artifacts'].get(cohort['preflight_receipt'])!=seal['preflight_sha256']):
        raise ValueError('sealed preflight provenance differs')
    verify(seal['artifacts'])
    if set(files_under(Path(cohort['preflight_receipt']).parent))!=set(seal['artifacts']):
        raise ValueError('preflight artifact inventory changed')
    return seal


def arm_path(root, case, ordinal):
    return root/'arms'/case['id']/f'{ordinal}-{ORDER[ordinal]}'


def verified_observation(directory, case, ordinal, cohort, expected_cohort, expected_seal, baseline):
    row=load(directory/'result.json')
    if (row['case']!=case['id'] or row['ordinal']!=ordinal or row['role']!=ORDER[ordinal]
            or row['cohort_sha256']!=expected_cohort or row['seal_sha256']!=expected_seal
            or row['parent_plan_sha256']!=cohort['parent_plan_sha256']
            or row['source_runtime_verified'] is not True):
        raise ValueError('prior arm identity/source/runtime proof differs')
    verify(row['artifacts'])
    if set(files_under(directory))!={*row['artifacts'],str(directory/'result.json')}:
        raise ValueError('prior arm artifact inventory differs')
    before=load(directory/'runtime-before/snapshot.json')
    if before!=load(directory/'runtime-after/snapshot.json') or invariant(before)!=baseline:
        raise ValueError('prior arm runtime differs from sealed preflight')
    exit_record=load(directory/'exit.json')
    for stream in ('stdout','stderr'):
        if sha(directory/('control-'+stream+'.log'))!=exit_record[stream+'_sha256']:
            raise ValueError('prior arm exit/log hash differs')
    passed=False;bits=None
    if exit_record['returncode']==0:
        try:
            bits=scalar_bits((directory/'control-stdout.log').read_text(),case['samples'],ORDER[ordinal]=='resident')
            passed=True
        except ValueError:
            pass
    if row['passed'] is True:
        value=row.get('score_bits')
        if (not isinstance(value,str) or len(value)!=16
                or any(ch not in '0123456789abcdef' for ch in value)
                or not math.isfinite(struct.unpack('>d',bytes.fromhex(value))[0])):
            raise ValueError('passed retained observation requires finite explicit f64 bits')
    if row['passed'] is not passed or row.get('score_bits')!=bits:
        raise ValueError('prior arm scalar gate differs from original retained output')
    return dict(ordinal=ordinal,role=ORDER[ordinal],passed=passed,score_bits=bits,
                result=str(directory/'result.json'),result_sha256=sha(directory/'result.json'))


def audit(root, expected_cohort, expected_seal, case_id, ordinal, output):
    """Read-only artifact audit under admission; never invokes an evaluator."""
    cohort,plan=context(root,expected_cohort)
    seal=checked_seal(root,cohort,expected_cohort,expected_seal)
    selected=[case for case in selected_cases(cohort,plan) if case['id']==case_id]
    if len(selected)!=1 or ordinal not in range(4):
        raise ValueError('one exact retained case and ABBA ordinal required')
    output=output.resolve()
    if not output.is_relative_to(root/'admission') or output.name!='audit-result.json':
        raise ValueError('new audit output must be inside owned admission directory')
    case=selected[0]
    observations=[verified_observation(arm_path(root,case,n),case,n,cohort,expected_cohort,
        expected_seal,seal['baseline']) for n in range(ordinal+1)]
    passed=all(row['passed'] is True for row in observations)
    if ordinal==3:
        bits=[row['score_bits'] for row in observations if row['passed'] is True]
        exact=len(bits)==4 and len(set(bits))==1
        expected=dict(case=case_id,cohort_sha256=expected_cohort,seal_sha256=expected_seal,
            observations=observations,passed=exact,matched_scope=False,
            qualified_timing_cells=0,qualification='unmatched_scope')
        if load(root/'cases'/(case_id+'.json'))!=expected:
            raise ValueError('retained ABBA case differs from audited original observations')
        passed=passed and exact
    # Recheck the source and sealed preflight closure before releasing admission.
    context(root,expected_cohort)
    checked_seal(root,cohort,expected_cohort,expected_seal)
    record=dict(schema=1,operation='read-only-retained-audit',case=case_id,ordinal=ordinal,
        cohort_sha256=expected_cohort,seal_sha256=expected_seal,observations=observations,
        passed=passed,source_runtime_verified=True,metric_inference_executed=False,
        matched_scope=False,qualified_timing_cells=0)
    write(output,record)
    print(json.dumps(record),flush=True)
    return 0 if passed else 1


def arm(root, expected_cohort, expected_seal, case_id, ordinal):
    cohort,plan=context(root,expected_cohort)
    seal=checked_seal(root,cohort,expected_cohort,expected_seal)
    selected=[case for case in selected_cases(cohort,plan) if case['id']==case_id]
    if len(selected)!=1 or ordinal not in range(4):
        raise ValueError('one exact case and ABBA ordinal required')
    case=selected[0]
    prior=[verified_observation(arm_path(root,case,n),case,n,cohort,expected_cohort,
                               expected_seal,seal['baseline']) for n in range(ordinal)]
    directory=arm_path(root,case,ordinal)
    directory.parent.mkdir(parents=True,exist_ok=True)
    directory.mkdir()
    scratch=directory/'scratch';scratch.mkdir()
    record=dict(schema=1,case=case_id,ordinal=ordinal,role=ORDER[ordinal],
        cohort_sha256=expected_cohort,seal_sha256=expected_seal,
        parent_plan_sha256=cohort['parent_plan_sha256'],passed=False,
        source_runtime_verified=False,matched_scope=False,qualified_timing_cells=0)
    try:
        env=dict(os.environ,TMPDIR=str(scratch),LC_ALL='C')
        for key in THREAD_ENV:env[key]=str(case['threads'])
        overrides=[key for key in ('OCTAVE_PATH','OCTAVE_EXEC_PATH','OCTAVE_HOME',
            'OCTAVE_SITE_INITFILE','OCTAVE_VERSION_INITFILE','LD_PRELOAD','LD_AUDIT') if env.get(key)]
        if overrides:raise ValueError('unbound runtime overrides: '+', '.join(overrides))
        before=capture(plan,Path(cohort['parent_plan']).parent,directory/'runtime-before',env)
        if invariant(before)!=seal['baseline']:
            raise ValueError('runtime/source closure differs from passed preflight')
        exit_record=execute(case['arms'][ORDER[ordinal]]['command'],env,directory,'control',1100)
        write(directory/'exit.json',exit_record)
        # All revalidation remains admitted and precedes lock release.
        after=capture(plan,Path(cohort['parent_plan']).parent,directory/'runtime-after',env)
        if before!=after:raise ValueError('runtime/source closure changed during arm')
        context(root,expected_cohort)
        checked_seal(root,cohort,expected_cohort,expected_seal)
        record.update(source_runtime_verified=True,returncode=exit_record['returncode'])
        if exit_record['returncode']!=0:
            record['error']='original metric process exited nonzero'
        else:
            try:
                record['score_bits']=scalar_bits((directory/'control-stdout.log').read_text(),
                                                 case['samples'],ORDER[ordinal]=='resident')
                record['passed']=True
            except ValueError as error:
                record['error']=str(error)
    except Exception as error:
        record['error']=str(error)
    record['artifacts']=files_under(directory)
    write(directory/'result.json',record)
    if not record['source_runtime_verified']:
        print(json.dumps(record),flush=True)
        return 2
    current=verified_observation(directory,case,ordinal,cohort,expected_cohort,expected_seal,seal['baseline'])
    if ordinal==3:
        observations=prior+[current]
        bits=[row['score_bits'] for row in observations if row['passed']]
        result=dict(case=case_id,cohort_sha256=expected_cohort,seal_sha256=expected_seal,
            observations=observations,passed=len(bits)==4 and len(set(bits))==1,
            matched_scope=False,qualified_timing_cells=0,qualification='unmatched_scope')
        (root/'cases').mkdir(exist_ok=True)
        write(root/'cases'/(case_id+'.json'),result)
    print(json.dumps(record),flush=True)
    return 0 if record['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('seal','arm','audit'))
    parser.add_argument('root',type=Path)
    parser.add_argument('--cohort-sha',required=True)
    parser.add_argument('--preflight-sha')
    parser.add_argument('--seal-sha')
    parser.add_argument('--case-id')
    parser.add_argument('--ordinal',type=int)
    parser.add_argument('--audit-output',type=Path)
    args=parser.parse_args()
    if args.operation=='seal':
        if not args.preflight_sha:parser.error('explicit passed-preflight SHA256 required')
        code=seal(args.root.resolve(strict=True),args.cohort_sha,args.preflight_sha)
    else:
        if not args.seal_sha or not args.case_id or args.ordinal is None:
            parser.error('explicit seal SHA256, case ID and ABBA ordinal required')
        if args.operation=='audit':
            if args.audit_output is None:parser.error('explicit new audit output required')
            code=audit(args.root.resolve(strict=True),args.cohort_sha,args.seal_sha,
                       args.case_id,args.ordinal,args.audit_output)
        else:
            code=arm(args.root.resolve(strict=True),args.cohort_sha,args.seal_sha,args.case_id,args.ordinal)
    raise SystemExit(code)


if __name__=='__main__':main()
