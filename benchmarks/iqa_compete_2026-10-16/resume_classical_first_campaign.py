#!/usr/bin/env python3
"""Resume original three-block constructor/first-call CPU campaigns."""
import argparse
import copy
import datetime
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from compare_canonical import sha, compare
from resume_canonical_campaign import _attempt, _completed
from supervision import timeout_command


def original(plan, helper, expected):
    helper=Path(helper).resolve(strict=True)
    if plan['bindings'].get(str(helper)) != expected or sha(helper) != expected:
        raise ValueError('original first-call helper binding differs')
    spec=importlib.util.spec_from_file_location('retained_first_recovery_helper',helper)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(result):
    value=copy.deepcopy(result)
    # The original coordinator adds its child's numerical/error exit to the
    # summary, while leaving the raw block result unchanged.
    if value.get('errors') or value['passed'] is False:
        value.setdefault('errors',{})['wrapper']='exit 1'
    return value


def validate(block,case,number):
    if block['phase'] != 'scorer_construct_first_eval' or block['order'] != (['canonical','native'] if number%2==0 else ['native','canonical']):
        raise ValueError('retained first-call phase/order differs')
    if set(block['arms']) == {'canonical','native'}:
        delta,passed=compare(block['arms']['native']['score'],block['arms']['canonical']['score'],case['absolute_bound'],case.get('relative_bound',0))
        if block['delta'] != delta or block['passed'] is not passed:
            raise ValueError('retained numerical classification differs')
    elif not block.get('errors'):
        raise ValueError('missing arm without retained error')


def summary(module,case,blocks,digest):
    reasons=sorted({reason for block in blocks for reason in block.get('scope_reasons',[])})
    return dict(case=case['id'],metric=case['metric'],phase=module.PHASE,blocks=blocks,first_call_scope=case['first_call_scope'],scope_reasons=reasons,
                absolute_bound=case['absolute_bound'],relative_bound=case.get('relative_bound',0),ratio=module.summarize(blocks,reasons,3),plan_sha256=digest,rust_source=case['rust_source'],
                qualification='descriptive evaluator first calls; no CI/regression acceptance; null bound means no assigned numerical qualification')


def execute(plan,case,number,output,helper,digest):
    command=['flock',str(Path.home()/'tmp/lilith-heavy.lock'),'flock',str(Path.home()/'tmp/lilith-wsl-heavy.lock'),
             str(Path.home()/'work/claudehints/scripts/run-heavy'),'--mem','12G','--jobs',str(case['threads']),'--',*timeout_command('Linux',1100),
             sys.executable,'-B',str(helper),str(plan),str(output),'--block-case',case['id'],'--block-number',str(number),'--expected-plan-sha',digest]
    with Path(str(output)+'-command.json').open('x') as stream:
        json.dump(command,stream,indent=2)
    with Path(str(output)+'-wrapper.log').open('x') as stream:
        completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,TMPDIR=str(output.parent),PYTHONDONTWRITEBYTECODE='1'))
    if not (output/'result.json').exists():
        raise RuntimeError('interrupted first-call attempt retained: '+str(output))
    result=json.loads((output/'result.json').read_text())
    if 'admission_or_binding' in result.get('errors',{}):
        raise ValueError('original source/admission refusal retained: '+str(output))
    expected=1 if result.get('errors') or result['passed'] is False else 0
    if completed.returncode != expected:
        raise ValueError('unexpected original helper exit; retain result and log: '+str(output))


def resume(plan_path,root,helper,expected,executor=execute):
    plan_path=Path(plan_path).resolve(strict=True);root=Path(root).absolute()
    plan_bytes=plan_path.read_bytes();plan=json.loads(plan_bytes);digest=sha(plan_path)
    if plan['phase'] != 'scorer_construct_first_eval':
        raise ValueError('wrong original phase')
    module=original(plan,helper,expected)
    root.mkdir(exist_ok=True);frozen=root/'plan.json'
    if frozen.exists():
        if frozen.read_bytes()!=plan_bytes:
            raise ValueError('frozen plan differs')
    else:
        frozen.write_bytes(plan_bytes)
    with (root/'recovery-coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=root/('recovery-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));journal.mkdir()
        (journal/'request.json').write_text(json.dumps(dict(plan_sha256=digest,helper=str(helper),helper_sha256=expected,source_sha256=sha(__file__),blocks=3,qualification='Retained original three-block CPU constructor/first-call campaign only; no fixed confirmation stages.'),indent=2)+'\n')
        rows=[]
        for case in plan['cases']:
            directory=root/case['id'];directory.mkdir(exist_ok=True)
            for path in directory.glob('block-*'):
                match=re.match(r'block-(\d+)(?:$|[-.])',path.name)
                if match and int(match[1])>=3:
                    raise ValueError('fixed confirmation stage requires its own controller')
            blocks=[];paths=[]
            for number in range(3):
                result=_completed(directory,number)
                if result is None:
                    output=_attempt(directory,number,'warm')
                    executor(frozen,case,number,output,helper,digest)
                    result=output/'result.json'
                block=json.loads(result.read_text());validate(block,case,number)
                blocks.append(normalize(block));paths.append(dict(path=str(result),sha256=sha(result)))
            value=summary(module,case,blocks,digest);target=directory/'summary.json'
            retained=target.exists()
            if retained:
                if json.loads(target.read_text())!=value:
                    raise ValueError('original completed summary differs; preserve it: '+str(target))
            else:
                module.write(target,value)
            with (journal/'cases.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(case=case['id'],blocks=paths,retained_summary=retained))+'\n')
            rows.append(value)
            if not retained:
                print('CASE_DONE',case['id'],'ratio',value['ratio'],flush=True)
            if frozen.read_bytes()!=plan_bytes:
                raise ValueError('frozen plan changed during recovery')
        aggregate=root/'summary.json'
        if aggregate.exists():
            if json.loads(aggregate.read_text())!=rows:
                raise ValueError('original aggregate differs')
        else:
            module.write(aggregate,rows)
        result=dict(completed=len(rows),failed=sum(any(b.get('errors') or b['passed'] is False for b in row['blocks']) for row in rows),plan_sha256=digest)
        module.write(journal/'result.json',result)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--original-helper',type=Path,required=True)
    parser.add_argument('--original-helper-sha256',required=True)
    args=parser.parse_args()
    print(json.dumps(resume(args.plan,args.output,args.original_helper,args.original_helper_sha256)),flush=True)


if __name__=='__main__':
    main()
