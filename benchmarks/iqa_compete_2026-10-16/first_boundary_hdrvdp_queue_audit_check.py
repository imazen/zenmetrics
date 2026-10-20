#!/usr/bin/env python3
"""Admitted file-only controls for old acceptance and successor audit refusal.

Synthetic artifacts only; no worker main, metric, Octave or subprocess is run.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def fixture(root, worker):
    root.mkdir();(root/'execution-cwd').mkdir();(root/'admission').mkdir()
    cases=[dict(id=f'synthetic-{n}',threads=1 if n%2==0 else 28,samples=1) for n in range(104)]
    plan=root/'plan.json'
    put(plan,dict(expected_control_order=worker.ORDER,cases=cases,preflight=[{}]*9,
        matched_scope=False,qualification='unmatched_scope',seconds_per_stage=1100,bindings={}))
    specs=root/'case-specs.json';put(specs,cases)
    preflight=root/'preflight/result.json';put(preflight,dict(synthetic=True))
    cohort=root/'cohort.json'
    put(cohort,dict(bindings={str(specs):sha(specs)},parent_plan=str(plan),parent_plan_sha256=sha(plan),
        case_specs=str(specs),kind='original104',preflight_receipt=str(preflight)))
    cohort_sha=sha(cohort)
    seal=root/'preflight-seal.json'
    baseline=dict(files={},dependencies={},environment={})
    put(seal,dict(passed=True,controls=9,cohort_sha256=cohort_sha,parent_plan_sha256=sha(plan),
        preflight_receipt=str(preflight),preflight_sha256=sha(preflight),
        artifacts={str(preflight):sha(preflight)},baseline=baseline))
    seal_sha=sha(seal);case=cases[0];observations=[]
    for ordinal,role in enumerate(worker.ORDER):
        directory=worker.arm_path(root,case,ordinal);directory.mkdir(parents=True)
        stdout=directory/'control-stdout.log'
        text='SAMPLE 100 1\nSCORE_HEX 3ff0000000000000\n'
        if role=='resident':text+='PARAMETER_CACHE_UNCHANGED passed=1\n'
        stdout.write_text(text);(directory/'control-stderr.log').write_text('')
        put(directory/'runtime-before/snapshot.json',baseline)
        put(directory/'runtime-after/snapshot.json',baseline)
        put(directory/'exit.json',dict(returncode=0,stdout_sha256=sha(stdout),
            stderr_sha256=sha(directory/'control-stderr.log')))
        result=directory/'result.json'
        put(result,dict(case=case['id'],ordinal=ordinal,role=role,cohort_sha256=cohort_sha,
            seal_sha256=seal_sha,parent_plan_sha256=sha(plan),source_runtime_verified=True,
            passed=True,score_bits='3ff0000000000000',artifacts=worker.files_under(directory)))
        observations.append(dict(ordinal=ordinal,role=role,passed=True,score_bits='3ff0000000000000',
            result=str(result),result_sha256=sha(result)))
    put(root/'cases'/(case['id']+'.json'),dict(case=case['id'],cohort_sha256=cohort_sha,
        seal_sha256=seal_sha,observations=observations,passed=True,matched_scope=False,
        qualified_timing_cells=0,qualification='unmatched_scope'))
    return case,cohort_sha,seal_sha


def mutate_bits(root,case,worker,value):
    path=root/'cases'/(case['id']+'.json');row=json.loads(path.read_text())
    for ordinal in range(4):
        result=worker.arm_path(root,case,ordinal)/'result.json'
        arm=json.loads(result.read_text());arm['score_bits']=value
        result.write_text(json.dumps(arm))
        row['observations'][ordinal].update(score_bits=value,result_sha256=sha(result))
    path.write_text(json.dumps(row))


def run(old_helpers,new_helpers,output):
    sys.path.insert(0,str(new_helpers))
    old=module('old_hdr_queue',old_helpers/'first_boundary_hdrvdp_arm_queue.py')
    new=module('new_hdr_queue',new_helpers/'first_boundary_hdrvdp_arm_queue.py')
    worker=module('new_hdr_worker',new_helpers/'first_boundary_hdrvdp_arm_worker.py')
    output.mkdir();results=[]
    for name in ('valid','changed-log','null-bits','nan-bits','changed-preflight','changed-seal'):
        root=output/name;case,cohort_sha,seal_sha=fixture(root,worker)
        if name=='changed-log':
            (worker.arm_path(root,case,0)/'control-stdout.log').write_text('changed retained raw output\n')
        if name=='null-bits':mutate_bits(root,case,worker,None)
        if name=='nan-bits':mutate_bits(root,case,worker,'7ff8000000000000')
        if name=='changed-preflight':(root/'preflight/result.json').write_text('{}')
        if name=='changed-seal':(root/'preflight-seal.json').write_text('{}')
        old_accepted=old.checked_case(root,case,cohort_sha,seal_sha)['passed']
        if old_accepted is not True:raise AssertionError('old retained summary no longer reproduces')
        try:
            code=worker.audit(root,cohort_sha,seal_sha,case['id'],3,root/'admission/audit-result.json')
            accepted=code==0;error=None
        except ValueError as exc:accepted=False;error=str(exc)
        if accepted != (name=='valid'):raise AssertionError('successor audit outcome differs: '+name)
        if name in ('null-bits','nan-bits'):
            try:new.checked_case(root,case,cohort_sha,seal_sha)
            except ValueError:pass
            else:raise AssertionError('new summary accepted absent/nonfinite exact bits')
        results.append(dict(control=name,old_summary_accepted=old_accepted,
            new_admitted_audit_accepted=accepted,rejection=error,passed=True))
    put(output/'result.json',dict(passed=True,controls=results,metric_inference_executed=False,
        synthetic_receipts=True,old_queue_sha256=sha(old_helpers/'first_boundary_hdrvdp_arm_queue.py'),
        new_queue_sha256=sha(new_helpers/'first_boundary_hdrvdp_arm_queue.py'),
        new_worker_sha256=sha(new_helpers/'first_boundary_hdrvdp_arm_worker.py'),
        controller_sha256=sha(__file__)))
    print(json.dumps(dict(passed=True,controls=len(results),metric_inference_executed=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('old_helpers','new_helpers','output'):parser.add_argument(name,type=Path)
    args=parser.parse_args();run(args.old_helpers.resolve(),args.new_helpers.resolve(),args.output.resolve())
