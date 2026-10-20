#!/usr/bin/env python3
"""Prepare a NEW headless parent and four warm photo controls; no execution.

Keeps the complete source parent inventory and original M arithmetic. Only new
wrapper paths and --no-window-system change; the visible queue selects four
warm cases at one/eight threads. Nine fresh preflights must precede sealing.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

from first_boundary_hdrvdp_arm_queue import prepare as prepare_queue
from first_boundary_hdrvdp_resident import sha, verify, write


def load(path):
    return json.loads(Path(path).read_text())


def headless(command,old_root,new_root):
    if command.count('--no-gui')!=1 or '--no-window-system' in command:
        raise ValueError('exact original GUI-disabled predecessor required')
    result=[part.replace(str(old_root),str(new_root)) for part in command]
    result.insert(result.index('--no-gui')+1,'--no-window-system')
    return result


def body(text):
    marker='clock=tic();\n'
    if text.count(marker)!=1 or text.count('elapsed=toc(clock)')!=1:
        raise ValueError('single original timed body required')
    start=text.index(marker)+len(marker)
    return text[start:text.index('elapsed=toc(clock)',start)]


def prepare(parent_path,expected,output,wrapper,locks):
    parent_path=parent_path.resolve(strict=True)
    if sha(parent_path)!=expected:raise ValueError('explicit predecessor SHA256 differs')
    old=load(parent_path);verify(old['bindings'])
    if (old['variant']!='resize2201' or len(old['cases'])!=104 or len(old['preflight'])!=9
            or old['matched_scope'] is not False or old.get('headless') is True
            or old['expected_control_order']!=['default','resident','resident','default']):
        raise ValueError('unchanged non-headless resident parent required')
    output.mkdir();root=output.resolve();old_root=parent_path.parent
    bindings={str(parent_path):expected}
    for name,expected_file in old['bindings'].items():
        original=Path(name)
        if original.is_relative_to(old_root):
            dest=root/original.relative_to(old_root);dest.parent.mkdir(parents=True,exist_ok=True)
            raw=original.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=expected_file:raise ValueError('source changed during copy')
            if original.is_relative_to(old_root/'variants') and original.suffix=='.m':
                raw=raw.replace(str(old_root).encode(),str(root).encode())
            with dest.open('xb') as stream:stream.write(raw)
            bindings[str(dest)]=sha(dest)
        else:bindings[name]=expected_file
    helpers=root/'helpers';helpers.mkdir(exist_ok=True)
    # New copies only: preserve the original resident runner/capture helpers,
    # add the corrected audited queue/worker and this source-only preparer.
    for name in ('first_boundary_hdrvdp_arm_queue.py','first_boundary_hdrvdp_arm_worker.py',
                 'first_boundary_hdrvdp_headless.py'):
        source=Path(__file__).with_name(name);dest=helpers/name
        with dest.open('xb') as stream:stream.write(source.read_bytes())
        bindings[str(dest)]=sha(dest)
    (root/'execution-cwd').mkdir()
    plan=copy.deepcopy(old)
    for case in plan['cases']:
        for role in ('default','resident'):
            prior=case['arms'][role]['command']
            command=headless(prior,old_root,root)
            original_body=body(Path(prior[-1]).read_text())
            if (body(Path(command[-1]).read_text())!=original_body or
                    hashlib.sha256(original_body.encode()).hexdigest()!=case['evaluator_body_sha256']):
                raise ValueError('headless relocation changed evaluator body')
            case['arms'][role]['command']=command
    for item in plan['preflight']:item['command']=headless(item['command'],old_root,root)
    plan['runtime_roots']=[p.replace(str(old_root),str(root)) for p in old['runtime_roots']]
    plan.update(bindings=bindings,headless=True,source_parent_plan=str(parent_path),
        source_parent_plan_sha256=expected,execution_ready=False,matched_scope=False,
        qualification='unmatched_scope',qualified_timing_cells=0,
        note='New headless source parent; nine new preflights required. Warm queue selects four cases only.')
    plan_path=root/'resize2201-headless.plan.json';write(plan_path,plan)
    preflight=root/'preflight-results/result.json'
    queued=prepare_queue(plan_path,preflight,root/'warm-queue',wrapper,locks,warm=True)
    queue=root/'warm-queue';cohort_sha=queued['cohort_sha256']
    python=sys.executable
    preflight_command=[]
    for lock in locks:preflight_command += ['/usr/bin/flock',str(lock)]
    preflight_command += [str(wrapper),'--mem','12G','--jobs','1','--','env',
        'TMPDIR='+str(root/'preflight-wrapper-scratch'),'PYTHONDONTWRITEBYTECODE=1',python,'-B',
        str(helpers/'first_boundary_hdrvdp_resident_run.py'),str(plan_path),str(preflight.parent),'--preflight-only']
    (root/'preflight-wrapper-scratch').mkdir()
    commands=dict(preflight=preflight_command,
        seal=[python,'-B',str(queue/'helpers/first_boundary_hdrvdp_arm_queue.py'),'seal',str(queue),
            '--cohort-sha',cohort_sha,'--preflight-sha','{passed_headless_preflight_result_sha256}'],
        warm_four=[python,'-B',str(queue/'helpers/first_boundary_hdrvdp_arm_queue.py'),'queue',str(queue),
            '--cohort-sha',cohort_sha,'--seal-sha','{passed_headless_seal_sha256}'],
        note='Root execution only. Seal/queue acquire per-operation admission internally; no outer shared lock.',
        matched_scope=False,qualified_timing_cells=0)
    write(root/'commands.json',commands)
    result=dict(schema=1,parent_plan=str(plan_path),parent_plan_sha256=sha(plan_path),
        cohort=queued['cohort'],cohort_sha256=cohort_sha,commands=str(root/'commands.json'),
        preflight_controls=9,warm_cases=4,warm_arms=16,source_parent_cases=104,
        metric_inference_executed=False,matched_scope=False,qualification='unmatched_scope',
        bound_v2=0.005,bound_v3=None,
        source_files={str(p):sha(p) for p in sorted(helpers.iterdir()) if p.is_file()})
    write(root/'index.json',result)
    # Queue metadata remains writable for future owned root execution; its
    # source/config files were frozen by the existing queue preparer.
    for path in root.rglob('*'):
        if path.is_file() and not path.is_relative_to(queue):path.chmod(0o444)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','output'):parser.add_argument(name,type=Path)
    parser.add_argument('--parent-sha',required=True)
    parser.add_argument('--heavy-wrapper',type=Path,required=True)
    parser.add_argument('--lock',type=Path,action='append',required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.parent,args.parent_sha,args.output,args.heavy_wrapper,args.lock)),flush=True)


if __name__=='__main__':main()
