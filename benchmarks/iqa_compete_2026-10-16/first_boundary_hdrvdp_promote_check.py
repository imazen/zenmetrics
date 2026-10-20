#!/usr/bin/env python3
"""Admitted file-only promoter controls. All scores/receipts here are synthetic.

Uses real frozen evaluator source slices and the real verifier/seal/audit
functions. Never calls worker.arm, cell.cell, Octave, a scorer or a subprocess.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import first_boundary_hdrvdp_arm_queue as queue
import first_boundary_hdrvdp_arm_worker as worker
import first_boundary_hdrvdp_promote as promote
import first_boundary_hdrvdp_resident_cell as cell


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def change(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))


def record(path):return dict(path=str(path),sha256=promote.sha(path))


def fixture(root,source_plan,batch_controller,wrapper,locks):
    root.mkdir();parent=root/'parent';parent.mkdir();(parent/'execution-cwd').mkdir()
    original=json.loads(source_plan.read_text());plan=copy.deepcopy(original)
    helpers=parent/'helpers';helpers.mkdir()
    for name in ('first_boundary_hdrvdp_resident.py','first_boundary_hdrvdp_resident_run.py',
                 'first_boundary_hdrvdp_admitted.py'):
        dest=helpers/name;shutil.copy2(source_plan.parent/'helpers'/name,dest)
        plan['bindings'][str(dest)]=promote.sha(dest)
    plan_path=parent/'plan.json';put(plan_path,plan)
    preflight=root/'preflight';pref_result=preflight/'result.json'
    parent_q=root/'parent-queue'
    queue.prepare(plan_path,pref_result,parent_q,wrapper,locks,warm=True)
    runtime=root/'runtime-fixture.m';runtime.write_text('% synthetic runtime identity\n')
    baseline=dict(files={str(runtime):dict(resolved=str(runtime),sha256=promote.sha(runtime))},
                  dependencies={},environment={})
    preflight.mkdir();shutil.copy2(plan_path,preflight/'plan.json');pref_rows=[]
    for item in plan['preflight']:
        directory=preflight/('preflight-'+item['id']);directory.mkdir()
        if item['kind']=='arrays':
            text=''.join(f'PARAMETER_ARRAY_EXACT key=synthetic-{i}\n' for i in range(item['keys']))
            text+=f"PARAMETER_ARRAY_COMPLETE keys={item['keys']} passed=1\n"
        else:text=f"PARAMETER_MISSING_EXACT index={item['index']} passed=1\n"
        (directory/'control-stdout.log').write_text(text);(directory/'control-stderr.log').write_text('')
        for phase in ('before','after'):put(directory/f'runtime-{phase}/snapshot.json',baseline)
        row=dict(control=item['id'],passed=True,returncode=0,
            stdout_sha256=promote.sha(directory/'control-stdout.log'),
            stderr_sha256=promote.sha(directory/'control-stderr.log'))
        put(directory/'result.json',row);pref_rows.append(dict(id=item['id'],**row))
    put(pref_result,dict(passed=True,plan_sha256=promote.sha(plan_path),
        controller_sha256=promote.sha(helpers/'first_boundary_hdrvdp_resident_run.py'),
        variant=plan['variant'],cases=[],coverage=dict(score_cases_completed=0,score_cases_required=104,preflight_only=True),
        qualified_timing_cells=0,matched_scope=False,qualification='unmatched_scope',preflight=pref_rows))
    parent_cohort=parent_q/'cohort.json';parent_sha=promote.sha(parent_cohort)
    batch=root/'batch';batch.mkdir()
    for name in ('helpers','admission','arms','cases'):(batch/name).mkdir()
    cohort=json.loads(parent_cohort.read_text())
    for source in (parent_q/'helpers').glob('*.py'):
        dest=batch/'helpers'/source.name;shutil.copy2(source,dest);cohort['bindings'][str(dest)]=promote.sha(dest)
    for field in ('case_specs','catalog'):
        dest=batch/Path(cohort[field]).name;shutil.copy2(cohort[field],dest)
        cohort[field]=str(dest);cohort['bindings'][str(dest)]=promote.sha(dest)
    controller=batch/'controller.py';shutil.copy2(batch_controller,controller)
    cohort['bindings'].update({str(controller):promote.sha(controller),str(parent_cohort):parent_sha})
    cohort.update(parent_cohort=str(parent_cohort),parent_cohort_sha256=parent_sha,
                  admission_granularity='one caller-admitted batch, four warm score-control cases')
    cohort_path=batch/'cohort.json';put(cohort_path,cohort);cohort_sha=promote.sha(cohort_path)
    worker.seal(batch,cohort_sha,promote.sha(pref_result))  # File validation only.
    seal=batch/'preflight-seal.json';seal_sha=promote.sha(seal)
    selected=worker.selected_cases(cohort,plan);audits=[]
    for case in selected:
        observations=[]
        for ordinal,role in enumerate(worker.ORDER):
            directory=worker.arm_path(batch,case,ordinal);directory.mkdir(parents=True)
            text='SAMPLE 100 1\nSCORE_HEX 3ff0000000000000\n'*case['samples']
            if role=='resident':text+='PARAMETER_CACHE_UNCHANGED passed=1\n'
            (directory/'control-stdout.log').write_text(text);(directory/'control-stderr.log').write_text('')
            for phase in ('before','after'):put(directory/f'runtime-{phase}/snapshot.json',baseline)
            put(directory/'exit.json',dict(returncode=0,stdout_sha256=promote.sha(directory/'control-stdout.log'),
                stderr_sha256=promote.sha(directory/'control-stderr.log')))
            put(directory/'result.json',dict(case=case['id'],ordinal=ordinal,role=role,
                cohort_sha256=cohort_sha,seal_sha256=seal_sha,parent_plan_sha256=promote.sha(plan_path),
                source_runtime_verified=True,passed=True,score_bits='3ff0000000000000',artifacts=worker.files_under(directory)))
            observations.append(dict(ordinal=ordinal,role=role,passed=True,score_bits='3ff0000000000000',
                result=str(directory/'result.json'),result_sha256=promote.sha(directory/'result.json')))
        put(batch/'cases'/(case['id']+'.json'),dict(case=case['id'],cohort_sha256=cohort_sha,seal_sha256=seal_sha,
            observations=observations,passed=True,matched_scope=False,qualified_timing_cells=0,qualification='unmatched_scope'))
        audit=batch/'admission'/case['id']/'audit-result.json';audit.parent.mkdir()
        worker.audit(batch,cohort_sha,seal_sha,case['id'],3,audit)  # File-only, no evaluator.
        audits.append(dict(case=case['id'],audit=str(audit),sha256=promote.sha(audit),passed=True))
    batch_result=batch/'result.json';put(batch_result,dict(cases=audits,passed=True,matched_scope=False,
        qualified_timing_cells=0,cohort_sha256=cohort_sha,preflight_seal_sha256=seal_sha,
        controller_sha256=promote.BATCH_SHA,worker_sha256=promote.WORKER_SHA))
    candidates=cell.prepare(plan_path,promote.sha(plan_path),seal,seal_sha,root/'cell-candidate')
    cfg=cell.load(candidates['config']);cell_records=[]
    for path in candidates['metric_plans'].values():
        for row in cell.load(path)['cases']:
            control=root/'cell-controls'/row['id'];scratch=control/'cell';scratch.mkdir(parents=True)
            samples=2;spec=cfg['routes'][row['metric']];script=scratch/'hdr-resident-entry.m'
            script.write_text(cell.script_for(spec,samples))
            loader=next(p for p in cfg['bindings'] if p.endswith('/load_spectral_resp.m') and
                        promote.sha(p)=='f51b19ae6a874bf5474a0bee367d9564fac84b091b9d343bc7c7a6e72e6ca560'
                        and ('official-v3' in p if row['metric']=='hdrvdp3' else 'official-v2' in p))
            text=f"PARAMETER_CACHE_SEALED keys={3 if row['metric']=='hdrvdp' else 4} original_helper={loader}\n"
            text+='COLD 100 1\nSCORE_HEX 3ff0000000000000\n'
            text+='SAMPLE 100 1\nSCORE_HEX 3ff0000000000000\n'*samples
            text+='PARAMETER_CACHE_UNCHANGED passed=1\n'
            for location in (scratch/'original-stdout.log',control/'adapter-stdout.log'):location.write_text(text)
            for location in (scratch/'original-stderr.log',control/'adapter-stderr.log'):location.write_text('')
            put(scratch/'hdr-cell-identity.json',dict(config_sha256=candidates['config_sha256'],metric=row['metric'],
                threads=row['threads'],samples=samples,evaluator_body_sha256=cell.BODY_SHA[row['metric']],
                script_sha256=promote.sha(script),matched_scope=False,qualification='unmatched_scope'))
            put(scratch/'hdr-cell-exit.json',dict(returncode=0,stdout_sha256=promote.sha(scratch/'original-stdout.log'),
                stderr_sha256=promote.sha(scratch/'original-stderr.log')))
            put(scratch/'original-command.json',['/usr/bin/time','-v','/usr/bin/timeout','1100s',plan['octave'],
                '--no-gui','--no-window-system','--no-init-file','--no-history','--quiet',str(script)])
            captured=copy.deepcopy(baseline)
            captured['environment']={key:str(row['threads']) for key in
                                    ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
            for phase in ('before','after'):put(scratch/f'runtime-{phase}/snapshot.json',captured)
            envelope=root/(row['id']+'-envelope.json')
            put(envelope,dict(schema=1,kind='hdr-resident-cell-protocol',passed=True,returncode=0,
                case_id=row['id'],metric=row['metric'],threads=row['threads'],samples=samples,
                command=[p.format_map(dict(row,samples=samples,output=str(control))) for p in row['canonical']['command']],
                root=str(control),cell_root=str(scratch),stdout=str(control/'adapter-stdout.log'),
                stderr=str(control/'adapter-stderr.log'),artifacts=worker.files_under(control)))
            cell_records.append(record(envelope))
    native_records=[]
    for threads in (1,8):
        native=root/f'native-t{threads}';native.mkdir();rows=[]
        meta=dict(binary=selected[0]['binary'],binary_sha256=promote.sha(selected[0]['binary']),
                  source_revision=selected[0]['rust_source'],threads=threads,samples=2)
        put(native/'plan.json',meta)
        for metric in ('hdrvdp','hdrvdp3'):
            case=next(c for c in selected if c['metric']==metric and c['threads']==threads)
            directory=native/metric;directory.mkdir();score=1.001 if metric=='hdrvdp' else 2.0
            config={key:case[key] for key in ('width','height','reference','distorted','threads')}
            config.update(metric_name=metric,only_mode='score',timing_phase='warm',required_samples=2)
            put(directory/'config.json',config);put(directory/'command.json',[meta['binary'],str(directory/'config.json')])
            row=dict(config,status='ok',rayon_threads_observed=threads,score=score,score_bits=promote.bits(score),
                self_warmup_score=score,ref_score=score,map_score_matches_scalar_bits=True,n=2,
                round_means_ns=[100,100],zenbench_samples=[dict(gate_clean=False),dict(gate_clean=True)])
            (directory/'stdout.log').write_text('IQA_WARMUP '+json.dumps(dict(completed=True,score_bits=promote.bits(score)))+
                '\nIQA_ROW '+json.dumps(row)+'\n')
            rows.append(dict(metric=metric,observations=dict(extra=row)))
        put(native/'observations.json',dict(plan_sha256=promote.sha(native/'plan.json'),
            source_revision=meta['source_revision'],bindings={meta['binary']:meta['binary_sha256']},rows=rows))
        native_records.append(dict(threads=threads,root=str(native),artifacts=worker.files_under(native)))
    request=root/'request.json';put(request,dict(schema=1,synthetic_fixture=True,cohort=record(cohort_path),seal=record(seal),
        batch_receipt=record(batch_result),candidate_plans=[record(Path(p)) for p in candidates['metric_plans'].values()],
        cell_controls=cell_records,native_controls=native_records))
    return request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source_plan','batch_controller','output'):parser.add_argument(name,type=Path)
    parser.add_argument('--heavy-wrapper',type=Path,required=True)
    parser.add_argument('--lock',type=Path,action='append',required=True)
    args=parser.parse_args();args.output.mkdir();results=[]
    for name in ('valid-unbounded','missing-cell','outer-failure','altered-cell-log','invented-v3-bound',
                 'changed-helper','missing-abba','runtime-drift','wrong-template','wrong-samples','changed-audit'):
        root=args.output/name
        request=fixture(root,args.source_plan,args.batch_controller,args.heavy_wrapper,args.lock)
        data=cell.load(request)
        if name=='missing-cell':data['cell_controls'].pop()
        if name=='outer-failure':
            record_=data['cell_controls'][0];envelope=cell.load(record_['path']);envelope['returncode']=1
            change(Path(record_['path']),envelope);record_['sha256']=promote.sha(record_['path'])
        if name=='altered-cell-log':
            envelope=cell.load(data['cell_controls'][0]['path']);Path(envelope['stdout']).write_text('altered retained log')
        if name=='invented-v3-bound':
            rec=next(r for r in data['candidate_plans'] if cell.load(r['path'])['cases'][0]['metric']=='hdrvdp3')
            plan=cell.load(rec['path']);plan['cases'][0]['absolute_bound']=0.005
            Path(rec['path']).chmod(0o644);change(Path(rec['path']),plan);rec['sha256']=promote.sha(rec['path'])
        if name=='changed-helper':
            cohort_path=Path(data['cohort']['path']);cohort=cell.load(cohort_path)
            helper=cohort_path.parent/'helpers/first_boundary_hdrvdp_resident_run.py';helper.chmod(0o644)
            helper.write_text(helper.read_text()+'\n# synthetic copy differs\n')
            cohort['bindings'][str(helper)]=promote.sha(helper);change(cohort_path,cohort)
            data['cohort']['sha256']=promote.sha(cohort_path)
        if name=='missing-abba':
            case=sorted((root/'batch/arms').iterdir())[0];next(case.glob('0-*/result.json')).unlink()
        if name=='runtime-drift':(root/'runtime-fixture.m').write_text('% changed synthetic runtime\n')
        if name=='wrong-template':
            rec=data['candidate_plans'][0];plan=cell.load(rec['path'])
            plan['cases'][0]['canonical']['command'][6]='another-route'
            Path(rec['path']).chmod(0o644);change(Path(rec['path']),plan);rec['sha256']=promote.sha(rec['path'])
        if name=='wrong-samples':
            rec=data['cell_controls'][0];envelope=cell.load(rec['path']);envelope['samples']+=1
            change(Path(rec['path']),envelope);rec['sha256']=promote.sha(rec['path'])
        if name=='changed-audit':
            result=cell.load(data['batch_receipt']['path']);Path(result['cases'][0]['audit']).write_text('{}\n')
        change(request,data)
        try:
            promoted=promote.promote(request,promote.sha(request),root/'promoted');accepted=True;error=None
        except (ValueError,FileNotFoundError) as exc:accepted=False;error=str(exc)
        if accepted!=(name=='valid-unbounded'):raise AssertionError('promoter outcome differs: '+name+' '+str(error))
        if accepted:
            receipt=cell.load(root/'promoted/scope-promotion.json')
            for row in receipt['cases']:
                if row['metric']=='hdrvdp3':
                    assert row['absolute_bound'] is None and row['native_parity_qualified'] is False
                    assert row['timing_scope_qualified'] is True and row['observed_absolute_difference']==1.0
                assert row['native']['resource_gate_clean']==[False,True] and row['native']['timing_qualified'] is False
        results.append(dict(control=name,passed=True,accepted=accepted,rejection=error))
    put(args.output/'result.json',dict(passed=True,controls=results,synthetic_receipts=True,
        metric_inference_executed=False,controller_sha256=promote.sha(__file__),promoter_sha256=promote.sha(promote.__file__)))
    print(json.dumps(dict(passed=True,controls=len(results),metric_inference_executed=False)),flush=True)


if __name__=='__main__':main()
