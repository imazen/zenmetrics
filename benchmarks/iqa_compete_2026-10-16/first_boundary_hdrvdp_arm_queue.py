#!/usr/bin/env python3
"""Visible HDR ABBA queue with a fresh shared admission for each scalar arm.

prepare writes new source/queue files only. seal and queue execute admitted
workers and therefore belong to the coordinator, never to the preparer.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys


ORDER=['default','resident','resident','default']


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def load(path):
    return json.loads(Path(path).read_text())


def prepare(plan_path, preflight, output, wrapper, locks, supplement=False):
    plan_path=plan_path.resolve(strict=True)
    plan=load(plan_path)
    if (len(plan['cases'])!=104 or len(plan['preflight'])!=9 or plan['expected_control_order']!=ORDER
            or plan['matched_scope'] is not False or plan['qualification']!='unmatched_scope'
            or len({c['id'] for c in plan['cases']})!=104 or {c['threads'] for c in plan['cases']}!={1,28}):
        raise ValueError('complete frozen 104-case resident plan required')
    if len(locks)!=2 or len({str(p.resolve()) for p in locks})!=2:
        raise ValueError('two distinct explicit shared admission locks required')
    output.mkdir();root=output.resolve()
    helpers=root/'helpers';helpers.mkdir()
    bindings={}
    for name in ('first_boundary_hdrvdp_arm_queue.py','first_boundary_hdrvdp_arm_worker.py'):
        dest=helpers/name;shutil.copy2(Path(__file__).with_name(name),dest)
        bindings[str(dest)]=sha(dest)
    for name in ('first_boundary_hdrvdp_admitted.py','first_boundary_hdrvdp_resident.py',
                 'first_boundary_hdrvdp_resident_run.py'):
        original=plan_path.parent/'helpers'/name
        if sha(original)!=plan['bindings'][str(original)]:
            raise ValueError('frozen predecessor helper differs')
        dest=helpers/name;shutil.copy2(original,dest);bindings[str(dest)]=sha(dest)
    wrapper=wrapper.resolve(strict=True)
    python=Path(sys.executable).resolve(strict=True)
    flock=Path('/usr/bin/flock').resolve(strict=True)
    for path in (wrapper,python,flock):bindings[str(path)]=sha(path)
    cases=copy.deepcopy(plan['cases'])
    if supplement:
        originals=[c for c in cases if c['panel']=='photo512' and c['threads']==1]
        if len(originals)!=4:raise ValueError('four one-thread photo predecessors required')
        cases=[]
        scripts=root/'thread8-scripts';scripts.mkdir()
        configs=root/'thread8-native';configs.mkdir()
        for old in originals:
            case=copy.deepcopy(old)
            case['derived_from_case_id']=old['id']
            case['id']=old['id'].replace('-t1-','-t8-',1)+'-supplement'
            case['threads']=8
            case['environment']={key:'8' for key in old['environment']}
            for role in ORDER[:2]:
                original=Path(old['arms'][role]['command'][-1])
                if sha(original)!=plan['bindings'][str(original)]:
                    raise ValueError('photo predecessor reference script changed')
                dest=scripts/(case['id']+'-'+role+'.m');shutil.copy2(original,dest)
                if sha(dest)!=sha(original):raise ValueError('supplement script copy changed')
                bindings[str(dest)]=sha(dest);dest.chmod(0o444)
                case['arms'][role]['command'][-1]=str(dest)
            original=Path(old['native_config'])
            if sha(original)!=plan['bindings'][str(original)]:
                raise ValueError('photo predecessor native config changed')
            dest=configs/(case['id']+'.json')
            write(dest,dict(load(original),threads=8));bindings[str(dest)]=sha(dest);dest.chmod(0o444)
            case['native_config']=str(dest)
            case['native']['command']=[old['binary'],str(dest)]
            cases.append(case)
    case_specs=root/'case-specs.json';write(case_specs,cases)
    bindings[str(case_specs)]=sha(case_specs)
    catalog=[]
    for case in cases:
        if Path(case['id']).name!=case['id'] or case['id'] in ('.','..'):
            raise ValueError('single-component case IDs required')
        catalog.append({k:case[k] for k in ('id','panel','metric','phase','threads','samples')})
    catalog_path=root/'queue-catalog.json'
    write(catalog_path,dict(cases=catalog,arms_per_case=ORDER,total_cases=len(cases),total_arms=4*len(cases)))
    bindings[str(catalog_path)]=sha(catalog_path)
    cohort=dict(schema=1,parent_plan=str(plan_path),parent_plan_sha256=sha(plan_path),
        preflight_receipt=str(preflight.resolve()),bindings=bindings,catalog=str(catalog_path),case_specs=str(case_specs),
        kind='photo512-t8-supplement' if supplement else 'original104',
        admission=dict(wrapper=str(wrapper),locks=[str(p.resolve()) for p in locks],
                       python=str(python),flock=str(flock),memory='12G'),
        expected_control_order=ORDER,seconds_per_stage=1100,automatic_retries=0,
        source_status='prepared only; no tests/inference/admission executed',
        matched_scope=False,qualified_timing_cells=0,qualification='unmatched_scope')
    path=root/'cohort.json';write(path,cohort)
    for directory in ('admission','arms','cases','events','views'):(root/directory).mkdir()
    render(root,cohort)
    for path in [root/'cohort.json',catalog_path,case_specs,*helpers.iterdir()]:path.chmod(0o444)
    return dict(cohort=str(root/'cohort.json'),cohort_sha256=sha(root/'cohort.json'),
                cases=len(cases),arms=4*len(cases),executed=False)


def check_cohort(root, expected):
    if sha(root/'cohort.json')!=expected:
        raise ValueError('explicit successor cohort SHA256 differs')
    cohort=load(root/'cohort.json')
    # Only small coordinator/source/admission-tool files are read outside
    # admission. Original inputs, runtime artifacts and receipt closure are
    # validated by the worker after both shared locks and run-heavy admission.
    for path,expected_file in cohort['bindings'].items():
        if sha(path)!=expected_file:raise ValueError('successor source binding differs: '+path)
    return cohort


def event(root,kind,**fields):
    sequence=len(list((root/'events').glob('*.json')))
    write(root/'events'/f'{sequence:06d}.json',dict(sequence=sequence,
        utc=datetime.now(timezone.utc).isoformat(),kind=kind,**fields))


def render(root,cohort):
    rows=[]
    for case in load(cohort['catalog'])['cases']:
        arms=[]
        for ordinal,role in enumerate(ORDER):
            path=root/'arms'/case['id']/f'{ordinal}-{role}'/'result.json'
            state='pending' if (root/'preflight-seal.json').is_file() else 'awaiting_preflight_seal'
            passed=None
            if path.is_file():
                try:
                    result=load(path)
                    state='complete' if result.get('source_runtime_verified') is True else 'blocked'
                    passed=result.get('passed')
                except json.JSONDecodeError:
                    state='writing_receipt'
            elif path.parent.exists():
                state='incomplete_or_running'
            arms.append(dict(ordinal=ordinal,role=role,state=state,passed=passed,result=str(path)))
        rows.append(dict(case=case,arms=arms))
    history=sorted((root/'events').glob('*.json'))
    latest=load(history[-1]) if history else None
    view=dict(schema=1,cases=rows,total_cases=len(rows),total_arms=4*len(rows),latest_event=latest,
              matched_scope=False,qualified_timing_cells=0)
    sequence=len(list((root/'views').glob('*.json')))
    archive=root/'views'/f'{sequence:06d}.json';write(archive,view)
    temporary=root/'queue-status.next.json'
    write(temporary,view)
    os.replace(temporary,root/'queue-status.json')
    return view


def dispatch(root,cohort,expected,operation,label,threads,arguments):
    directory=root/'admission'/label
    directory.parent.mkdir(parents=True,exist_ok=True)
    directory.mkdir()  # No overwrite, automatic retry or inferred recovery.
    scratch=directory/'scratch';scratch.mkdir()
    cfg=cohort['admission']
    command=[]
    for lock in cfg['locks']:command += [cfg['flock'],'-x',lock]
    command += [cfg['wrapper'],'--mem',cfg['memory'],'--jobs',str(threads),'--',
                'env','TMPDIR='+str(scratch),cfg['python'],'-B',
                str(root/'helpers/first_boundary_hdrvdp_arm_worker.py'),operation,str(root),
                '--cohort-sha',expected,*arguments]
    write(directory/'command.json',command)
    event(root,'waiting_for_shared_admission',operation=operation,label=label,threads=threads,
          log=str(directory/'wrapper.log'))
    render(root,cohort)
    print('HDR_ARM_WAIT '+json.dumps(dict(label=label,threads=threads,log=str(directory/'wrapper.log'))),flush=True)
    with (directory/'wrapper.log').open('x') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        write(directory/'process.json',dict(pid=process.pid,command=command))
        event(root,'owned_dispatch',label=label,pid=process.pid,process=str(directory/'process.json'))
        render(root,cohort)
        # The visible owned arm remains bounded by its existing timeout and
        # resource wrapper; no next arm is launched until this scope returns.
        code=process.wait()
    write(directory/'exit.json',dict(returncode=code))
    event(root,'shared_admission_released',operation=operation,label=label,returncode=code)
    render(root,cohort)
    return code


def checked_case(root,case,expected,seal):
    row=load(root/'cases'/(case['id']+'.json'))
    if (row['case']!=case['id'] or row['cohort_sha256']!=expected or row['seal_sha256']!=seal
            or len(row['observations'])!=4):
        raise ValueError('retained case identity or complete ABBA inventory differs')
    bits=[]
    for ordinal,observation in enumerate(row['observations']):
        path=root/'arms'/case['id']/f'{ordinal}-{ORDER[ordinal]}'/'result.json'
        arm=load(path)
        if (observation['ordinal']!=ordinal or observation['role']!=ORDER[ordinal]
                or observation['result']!=str(path) or observation['result_sha256']!=sha(path)
                or arm['source_runtime_verified'] is not True or arm['cohort_sha256']!=expected
                or arm['seal_sha256']!=seal or arm['case']!=case['id'] or arm['ordinal']!=ordinal
                or arm['role']!=ORDER[ordinal] or observation['passed'] is not arm['passed']
                or observation['score_bits']!=arm.get('score_bits')):
            raise ValueError('retained ABBA observation or arm receipt hash differs')
        if arm['passed'] is True:
            value=arm.get('score_bits')
            if (not isinstance(value,str) or len(value)!=16
                    or any(ch not in '0123456789abcdef' for ch in value)
                    or not math.isfinite(struct.unpack('>d',bytes.fromhex(value))[0])):
                raise ValueError('retained passed arm requires finite explicit f64 bits')
            bits.append(value)
    passed=len(bits)==4 and len(set(bits))==1
    if row['passed'] is not passed:raise ValueError('retained ABBA exact-bit gate differs')
    return row


def coordinate(args):
    root=args.root.resolve(strict=True)
    cohort=check_cohort(root,args.cohort_sha)
    # This lock protects this queue's metadata only; shared CPU admission is
    # acquired and released independently for each dispatch below.
    with (root/'coordinator.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='seal':
            return dispatch(root,cohort,args.cohort_sha,'seal','preflight-seal',1,
                            ['--preflight-sha',args.preflight_sha])
        catalog=load(cohort['catalog'])['cases']
        selected=[c for c in catalog if args.case_id is None or c['id']==args.case_id]
        if not selected:raise ValueError('case selector did not match exact frozen ID')
        if sha(root/'preflight-seal.json')!=args.seal_sha:
            raise ValueError('explicit passed-preflight seal hash differs')
        failed=False
        for case in selected:
            ordinals=range(4) if args.ordinal is None else [args.ordinal]
            for ordinal in ordinals:
                role=ORDER[ordinal]
                result=root/'arms'/case['id']/f'{ordinal}-{role}'/'result.json'
                if result.is_file():
                    # Audit original artifacts inside shared admission even
                    # when there will be no fresh arm in this selection.
                    event(root,'retained_arm_not_rerun',case=case['id'],ordinal=ordinal,result=str(result))
                    audits=root/'admission'/'audits'
                    number=len(list(audits.glob('*/*'))) if audits.exists() else 0
                    label=f'audits/{case["id"]}/{ordinal}-{role}-{number:06d}'
                    audit=root/'admission'/label/'audit-result.json'
                    code=dispatch(root,cohort,args.cohort_sha,'audit',label,1,
                        ['--seal-sha',args.seal_sha,'--case-id',case['id'],'--ordinal',str(ordinal),
                         '--audit-output',str(audit)])
                    if code not in (0,1) or not audit.is_file():
                        raise ValueError('admitted retained audit failed; further arms refused')
                    row=load(audit)
                    if (row['source_runtime_verified'] is not True or row['case']!=case['id']
                            or row['ordinal']!=ordinal or row['cohort_sha256']!=args.cohort_sha
                            or row['seal_sha256']!=args.seal_sha or (code==0)!=(row['passed'] is True)):
                        raise ValueError('retained audit identity or exit outcome differs')
                    failed |= row['passed'] is not True
                    print('HDR_ARM_RETAINED '+str(result),flush=True)
                    continue
                code=dispatch(root,cohort,args.cohort_sha,'arm',case['id']+f'/{ordinal}-{role}',
                    case['threads'],['--seal-sha',args.seal_sha,'--case-id',case['id'],'--ordinal',str(ordinal)])
                if not result.is_file():
                    raise ValueError('admitted worker produced no terminal arm receipt; retain logs, no automatic retry')
                row=load(result)
                if row['source_runtime_verified'] is not True or code not in (0,1):
                    raise ValueError('source/runtime/admission failure; further arms refused')
                if (code==0)!=(row['passed'] is True):
                    raise ValueError('admission exit and arm numerical outcome disagree')
                failed |= row['passed'] is not True
            if args.ordinal is None:
                case_path=root/'cases'/(case['id']+'.json')
                if not case_path.is_file():
                    raise ValueError('complete ABBA case receipt missing')
                row=checked_case(root,case,args.cohort_sha,args.seal_sha)
                failed |= row['passed'] is not True
        render(root,cohort)
        summary=dict(selected_cases=len(selected),case_selector=args.case_id,ordinal_selector=args.ordinal,
                     numerical_failure_seen=failed,matched_scope=False,qualified_timing_cells=0)
        event(root,'selection_finished',**summary)
        print('HDR_ARM_SELECTION '+json.dumps(summary),flush=True)
        return 1 if failed else 0


def digest(value):
    if len(value)!=64 or any(ch not in '0123456789abcdef' for ch in value):
        raise argparse.ArgumentTypeError('explicit lowercase SHA256 required')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    prep=sub.add_parser('prepare')
    for name in ('plan','preflight','output'):prep.add_argument(name,type=Path)
    prep.add_argument('--heavy-wrapper',type=Path,required=True)
    prep.add_argument('--lock',type=Path,action='append',required=True)
    prep.add_argument('--photo512-threads8',action='store_true')
    for mode in ('seal','queue'):
        active=sub.add_parser(mode);active.add_argument('root',type=Path)
        active.add_argument('--cohort-sha',type=digest,required=True)
        if mode=='seal':active.add_argument('--preflight-sha',type=digest,required=True)
        else:
            active.add_argument('--seal-sha',type=digest,required=True)
            active.add_argument('--case-id')
            active.add_argument('--ordinal',type=int,choices=range(4))
    status=sub.add_parser('status');status.add_argument('root',type=Path)
    args=parser.parse_args()
    if args.mode=='prepare':
        print(json.dumps(prepare(args.plan,args.preflight,args.output,args.heavy_wrapper,args.lock,args.photo512_threads8)))
    elif args.mode=='status':
        print((args.root/'queue-status.json').read_text())
    else:
        if args.mode=='queue' and args.ordinal is not None and not args.case_id:
            parser.error('one-arm ordinal selector requires an exact case ID')
        raise SystemExit(coordinate(args))


if __name__=='__main__':main()
