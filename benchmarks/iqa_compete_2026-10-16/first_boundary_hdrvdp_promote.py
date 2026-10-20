#!/usr/bin/env python3
"""Verify retained HDR receipts and write NEW descriptive reference plans.

Caller supplies shared admission. This program never executes a scorer, Octave
or a subprocess. Timing-scope agreement and numerical acceptance are separate.
"""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

from first_boundary_hdrvdp_resident import sha, write


WORKER_SHA='8035916fafc25f4e7a4d9b1028827628cc8f5c42138b1565ae98c7fa716778ce'
CELL_SHA='35a252870cd9622372da426807115b0f065a444023463fbc0b94ff0e07403d77'
BATCH_SHA='58123857d6569cd8bb6720f39071e013143b23c3e193b132b917b3226219cac8'
IDENTITIES={(metric,threads) for metric in ('hdrvdp','hdrvdp3') for threads in (1,8)}


def require(condition,message):
    if not condition:raise ValueError(message)


def bits(value):
    require(type(value) in (int,float) and math.isfinite(value),'finite explicit scalar required')
    return struct.pack('>d',value).hex()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class Verifier:
    def __init__(self):self.bindings={}

    def file(self,path,expected=None):
        path=str(Path(path).absolute());actual=sha(path)
        if expected is not None:require(actual==expected,'artifact hash differs: '+path)
        require(path not in self.bindings or self.bindings[path]==actual,'artifact changed during verification: '+path)
        self.bindings[path]=actual
        return actual

    def doc(self,record):
        self.file(record['path'],record['sha256'])
        return json.loads(Path(record['path']).read_text())

    def maps(self,mapping):
        for path,expected in mapping.items():self.file(path,expected)

    def tree(self,root,mapping,worker):
        require(worker.files_under(root)==mapping,'complete artifact inventory/hash differs: '+str(root))
        self.maps(mapping)


def cell_protocol(v,envelope,case,row,config,cell,baseline,reference_bits,worker):
    metric,threads=case['metric'],case['threads'];samples=envelope['samples']
    require(envelope['schema']==1 and envelope['kind']=='hdr-resident-cell-protocol'
        and envelope['passed'] is True and type(envelope['returncode']) is int and envelope['returncode']==0
        and envelope['metric']==metric and envelope['threads']==threads
        and envelope['case_id']==row['id']
        and type(samples) is int and samples>=1,'explicit completed cell control required')
    root=Path(envelope['root']).resolve(strict=True)
    scratch=Path(envelope['cell_root']).resolve(strict=True)
    require(scratch==root/'cell','cell scratch must be inside its retained control root')
    v.tree(root,envelope['artifacts'],worker)
    argv=[part.format_map(dict(row,samples=samples,output=str(root))) for part in row['canonical']['command']]
    require(envelope['command']==argv,'actual cell command differs from candidate template/budget')
    identity=json.loads((scratch/'hdr-cell-identity.json').read_text())
    require(identity['config_sha256']==argv[5] and identity['metric']==metric
        and identity['threads']==threads and identity['samples']==samples
        and identity['evaluator_body_sha256']==case['evaluator_body_sha256'],'cell identity/body differs')
    spec=config['routes'][metric]
    source=Path(spec['predecessor']['arms']['resident']['command'][-1])
    require(sha(spec['original_wrapper'])==sha(source)==spec['original_wrapper_sha256'],
            'original resident wrapper copy differs')
    for name,text in cell.split_original(source.read_text(),metric).items():
        require(Path(spec['parts'][name]).read_text()==text,'original evaluator slice differs: '+name)
    script=scratch/'hdr-resident-entry.m'
    require(script.read_text()==cell.script_for(spec,samples) and sha(script)==identity['script_sha256'],
            'executed cell script differs from original body/protocol')
    command=json.loads((scratch/'original-command.json').read_text())
    expected=[config['routes'][metric]['predecessor']['arms']['resident']['command'][0],
        '--no-gui','--no-window-system','--no-init-file','--no-history','--quiet',str(script)]
    require(command[-len(expected):]==expected,
        'original Octave command/headless flags differ')
    before=json.loads((scratch/'runtime-before/snapshot.json').read_text())
    require(before==json.loads((scratch/'runtime-after/snapshot.json').read_text())
        and worker.invariant(before)==baseline,'cell runtime closure differs from headless preflight')
    require(all(before['environment'].get(key)==str(threads) for key in
                ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')),
            'cell captured thread environment differs from explicit budget')
    v.maps({path:r['sha256'] for path,r in before['files'].items()})
    v.maps({r['resolved']:r['sha256'] for r in before['files'].values()})
    outcome=json.loads((scratch/'hdr-cell-exit.json').read_text())
    require(type(outcome['returncode']) is int and outcome['returncode']==0,'original cell process failed')
    for stream in ('stdout','stderr'):
        original=scratch/('original-'+stream+'.log')
        require(sha(original)==outcome[stream+'_sha256'],'cell process log hash differs')
        outer=Path(envelope[stream]).resolve(strict=True)
        require(outer.is_relative_to(root) and outer.read_bytes()==original.read_bytes(),
                'outer cell output differs from verified forwarded original output')
    text=(scratch/'original-stdout.log').read_text()
    bootstrap=[line for line in text.splitlines() if line.startswith('PARAMETER_CACHE_SEALED ')]
    prefix=f'PARAMETER_CACHE_SEALED keys={3 if metric=="hdrvdp" else 4} original_helper='
    require(len(bootstrap)==1 and bootstrap[0].startswith(prefix),'original-helper bootstrap provenance required')
    loader=bootstrap[0][len(prefix):]
    require(config['bindings'].get(loader)=='f51b19ae6a874bf5474a0bee367d9564fac84b091b9d343bc7c7a6e72e6ca560',
            'bootstrap original-helper binding differs')
    v.file(loader,config['bindings'][loader])
    observations=[line.split() for line in text.splitlines() if line.startswith(('COLD ','SAMPLE '))]
    require([r[0] for r in observations]==['COLD']+['SAMPLE']*samples,'complete COLD/SAMPLE protocol required')
    require(worker.scalar_bits(text,samples+1,True)==reference_bits,'actual cell/default/resident score bits differ')
    return dict(samples=samples,score_bits=reference_bits,source_runtime_verified=True)


def native_observation(v,record,case,plan,worker):
    root=Path(record['root']).resolve(strict=True);v.tree(root,record['artifacts'],worker)
    metadata=json.loads((root/'plan.json').read_text())
    receipt=json.loads((root/'observations.json').read_text())
    require(sha(root/'plan.json')==receipt['plan_sha256'] and metadata['binary']==plan['binary']
        and metadata['source_revision']==receipt['source_revision']==plan['rust_source']
        and metadata['threads']==case['threads'],'native source/plan/budget differs')
    v.file(plan['binary'],metadata['binary_sha256']);v.maps(receipt['bindings'])
    directory=root/case['metric'];config=json.loads((directory/'config.json').read_text())
    command=json.loads((directory/'command.json').read_text())
    require(command[-2:]==[plan['binary'],str(directory/'config.json')],'native command/binary differs')
    rows=[json.loads(line[8:]) for line in (directory/'stdout.log').read_text().splitlines()
          if line.startswith('IQA_ROW ')]
    require(len(rows)==1,'one completed native scalar row required');row=rows[0]
    for key,value in config.items():require(row[key]==value,'native config/row differs: '+key)
    require(row['status']=='ok' and row['metric_name']==case['metric'] and row['threads']==case['threads']
        and row['width']==case['width'] and row['height']==case['height']
        and row['timing_phase']=='warm' and row['only_mode']=='score'
        and row['rayon_threads_observed']==case['threads'],'native scalar/budget contract differs')
    for role in ('reference','distorted'):
        require(sha(config[role])==sha(case[role]),'native/reference input bytes differ')
    observed=bits(row['score'])
    require(row['score_bits']==observed==bits(row['self_warmup_score'])==bits(row['ref_score']),
            'native finite/warm identity differs')
    warmup=[json.loads(line[11:]) for line in (directory/'stdout.log').read_text().splitlines()
            if line.startswith('IQA_WARMUP ')]
    require(len(warmup)==1 and warmup[0]['completed'] is True and warmup[0]['score_bits']==observed
        and row['map_score_matches_scalar_bits'] is True,'native warmup/variant proof differs')
    completed=[r for r in receipt['rows'] if r['metric']==case['metric']]
    require(len(completed)==1 and completed[0]['observations']['extra']==row,'native retained summary differs')
    gates=[sample['gate_clean'] for sample in row['zenbench_samples']]
    require(type(row['required_samples']) is int and row['required_samples']>=1
        and len(gates)==row['required_samples']==metadata['samples']==len(row['round_means_ns'])==row['n']
        and all(type(t) in (int,float) and math.isfinite(t) and t>0 for t in row['round_means_ns'])
        and all(type(g) is bool for g in gates),'complete native resource observations required')
    return dict(score=row['score'],score_bits=observed,resource_gate_clean=gates,
                timing_qualified=False,note='Numerical observation only; future timing attempts retain all resource gates')


def promote(request_path,expected,output):
    v=Verifier();request=v.doc(dict(path=str(request_path),sha256=expected))
    require(request['schema']==1,'explicit request schema required')
    cohort=v.doc(request['cohort']);root=Path(request['cohort']['path']).parent
    v.maps(cohort['bindings'])  # Verify imported helper closure before importing it.
    batch=None
    if 'parent_cohort' in cohort:
        parent=v.doc(dict(path=cohort['parent_cohort'],sha256=cohort['parent_cohort_sha256']))
        v.maps(parent['bindings']);parent_root=Path(cohort['parent_cohort']).parent
        require(cohort['bindings'].get(cohort['parent_cohort'])==cohort['parent_cohort_sha256'],
                'derived cohort does not bind its explicit source parent')
        for key in ('kind','parent_plan','parent_plan_sha256','preflight_receipt','admission',
                    'expected_control_order','seconds_per_stage','automatic_retries'):
            require(cohort[key]==parent[key],'derived cohort changed original contract: '+key)
        for field in ('case_specs','catalog'):
            require(v.file(cohort[field])==v.file(parent[field]),'derived case/catalog copy differs')
        for name in ('first_boundary_hdrvdp_arm_queue.py','first_boundary_hdrvdp_arm_worker.py',
                     'first_boundary_hdrvdp_resident.py','first_boundary_hdrvdp_resident_run.py',
                     'first_boundary_hdrvdp_admitted.py'):
            original=parent_root/'helpers'/name;copied=root/'helpers'/name
            require(v.file(copied)==v.file(original,parent['bindings'][str(original)]),
                    'derived helper differs from verified parent: '+name)
        v.file(root/'controller.py',BATCH_SHA)
        require(cohort['admission_granularity']=='one caller-admitted batch, four warm score-control cases',
                'explicit derived batch admission policy required')
        batch=v.doc(request['batch_receipt'])
        require(batch['passed'] is True and batch['cohort_sha256']==request['cohort']['sha256']
            and batch['preflight_seal_sha256']==request['seal']['sha256']
            and batch['controller_sha256']==BATCH_SHA and batch['worker_sha256']==WORKER_SHA
            and batch['matched_scope'] is False and batch['qualified_timing_cells']==0,
            'complete successful source-bound batch receipt required')
    worker_path=root/'helpers/first_boundary_hdrvdp_arm_worker.py';v.file(worker_path,WORKER_SHA)
    sys.path.insert(0,str(worker_path.parent));worker=module('hdr_receipt_worker',worker_path)
    cohort,source=worker.context(root,request['cohort']['sha256'])
    require(cohort['kind']=='photo512-warm-t1-t8' and source.get('headless') is True,
            'new headless four-warm cohort required')
    seal=worker.checked_seal(root,cohort,request['cohort']['sha256'],request['seal']['sha256'])
    require(Path(request['seal']['path'])==root/'preflight-seal.json','seal path differs')
    baseline,artifacts=worker.validate_preflight(cohort,source,seal['preflight_sha256'])
    require(baseline==seal['baseline'] and artifacts==seal['artifacts'],'fresh nine-preflight closure differs')
    v.file(request['seal']['path'],request['seal']['sha256']);v.maps(artifacts)
    v.file(cohort['parent_plan'],cohort['parent_plan_sha256']);v.maps(cohort['bindings']);v.maps(source['bindings'])
    selected=worker.selected_cases(cohort,source)
    cases={(c['metric'],c['threads']):c for c in selected}
    require(set(cases)==IDENTITIES and len(selected)==4,'exact four-case inventory required')
    audits={} if batch is None else {a['case']:a for a in batch['cases']}
    if batch is not None:
        require(len(batch['cases'])==4 and set(audits)=={c['id'] for c in selected},'four batch audit receipts required')
    abba={}
    for identity,case in cases.items():
        observations=[]
        for ordinal in range(4):
            directory=worker.arm_path(root,case,ordinal)
            observations.append(worker.verified_observation(directory,case,ordinal,cohort,
                request['cohort']['sha256'],request['seal']['sha256'],baseline))
            arm=json.loads((directory/'result.json').read_text());v.maps(arm['artifacts']);v.file(directory/'result.json')
        require(all(o['passed'] is True for o in observations) and len({o['score_bits'] for o in observations})==1,
                'complete exact original/resident ABBA required')
        terminal=root/'cases'/(case['id']+'.json');v.file(terminal)
        require(json.loads(terminal.read_text())==dict(case=case['id'],cohort_sha256=request['cohort']['sha256'],
            seal_sha256=request['seal']['sha256'],observations=observations,passed=True,matched_scope=False,
            qualified_timing_cells=0,qualification='unmatched_scope'),'terminal ABBA receipt differs from raw artifacts')
        if batch is not None:
            declared=audits[case['id']];audit=v.doc(dict(path=declared['audit'],sha256=declared['sha256']))
            require(declared['passed'] is True and audit['passed'] is True
                and audit['operation']=='read-only-retained-audit' and audit['case']==case['id']
                and audit['ordinal']==3 and audit['source_runtime_verified'] is True
                and audit['cohort_sha256']==request['cohort']['sha256']
                and audit['seal_sha256']==request['seal']['sha256'] and audit['observations']==observations,
                'batch terminal audit differs from verified raw observations')
        abba[identity]=observations[0]['score_bits']
    plans=[v.doc(record) for record in request['candidate_plans']]
    require(all(len(p['cases'])==2 and len({r['metric'] for r in p['cases']})==1
        and {r['threads'] for r in p['cases']}=={1,8} for p in plans),'one metric and both budgets per plan required')
    plan_rows={(r['metric'],r['threads']):(p,r) for p in plans for r in p['cases']}
    require(len(plans)==2 and sum(len(p['cases']) for p in plans)==4 and set(plan_rows)==IDENTITIES,
            'exact two-route 1/8 candidate plans required')
    controls=[v.doc(record) for record in request['cell_controls']]
    controls_by_id={(c['metric'],c['threads']):c for c in controls}
    require(len(controls)==4 and set(controls_by_id)==IDENTITIES,'four actual cell protocol receipts required')
    native={entry['threads']:entry for entry in request['native_controls']}
    require(len(request['native_controls'])==2 and set(native)=={1,8},'both explicit native budgets required')
    verified=[]
    for identity,case in cases.items():
        plan,row=plan_rows[identity];v.maps(plan['bindings']);command=row['canonical']['command']
        require(plan['binary']==case['binary'] and plan['rust_source']==case['rust_source'],
                'candidate native worker/source differs from original bound cohort')
        require(len(command)==13 and command[2]=='cell' and command[4]=='--config-sha'
            and command[6:]==[case['metric'],'{width}','{height}','{reference}','{distorted}',
                               '{threads}','{samples}']
            and row['canonical']['protocol']=='samples','explicit cell template required')
        python=Path(command[0]).resolve(strict=True)
        v.file(command[0],v.file(python,plan['bindings'][str(python)]))
        v.file(command[1],CELL_SHA);cell=module('hdr_receipt_cell',Path(command[1]))
        config=v.doc(dict(path=command[3],sha256=command[5]));v.maps(config['bindings'])
        capture=Path(cohort['parent_plan']).parent/'helpers/first_boundary_hdrvdp_admitted.py'
        require(v.file(config['admitted_helper'])==v.file(capture,source['bindings'][str(capture)]),
                'cell runtime-capture helper differs from verified parent')
        require(config['parent_plan']==cohort['parent_plan']
            and config['parent_plan_sha256']==cohort['parent_plan_sha256']
            and config['preflight_seal']==str(root/'preflight-seal.json')
            and config['preflight_seal_sha256']==request['seal']['sha256'],'cell binds another parent/seal')
        for key in ('metric','width','height','reference','distorted','absolute_bound','relative_bound'):
            require(row[key]==case[key],'candidate changed source contract: '+key)
        require(row['matched_scope'] is False and row['absolute_bound']==(0.005 if case['metric']=='hdrvdp' else None),
                'unchanged candidate numerical bound/scope required')
        predecessor=config['routes'][case['metric']]['predecessor']
        require(predecessor in source['cases'] and predecessor['metric']==case['metric']
            and predecessor['phase']=='warm' and predecessor['threads']==1,'original cell predecessor differs')
        protocol=cell_protocol(v,controls_by_id[identity],case,row,config,cell,baseline,abba[identity],worker)
        observation=native_observation(v,native[case['threads']],case,plan,worker)
        reference=struct.unpack('>d',bytes.fromhex(abba[identity]))[0]
        delta=abs(observation['score']-reference)
        bounded=case['absolute_bound'] is not None
        if bounded:require(delta<=max(case['absolute_bound'],case['relative_bound']*abs(reference)),
                           'native score exceeds unchanged assigned bound')
        verified.append(dict(metric=case['metric'],threads=case['threads'],protocol=protocol,
            reference_score=reference,native=observation,observed_absolute_difference=delta,
            absolute_bound=case['absolute_bound'],native_parity_qualified=bounded,
            numerical_acceptance='within_existing_bound' if bounded else 'unbounded_observation',
            timing_scope_qualified=True))
    v.maps(dict(v.bindings))  # Fail if any previously checked artifact changed.
    output.mkdir();destination=output.resolve()
    receipt=destination/'scope-promotion.json'
    write(receipt,dict(schema=1,passed=True,matched_scope=True,timing_scope_qualified=True,
        qualified_timing_cells=0,metric_inference_executed=False,request_sha256=expected,
        controller_sha256=sha(__file__),cases=verified,bindings=v.bindings,
        qualification='Descriptive matched timing scope; numerical acceptance and future resource gates remain separate'))
    final_bindings=dict(v.bindings,**{str(receipt):sha(receipt)})
    summaries={(r['metric'],r['threads']):r for r in verified};metric_plans={}
    for original in plans:
        plan=copy.deepcopy(original);plan['bindings'].update(final_bindings)
        for row in plan['cases']:
            evidence=summaries[(row['metric'],row['threads'])]
            row.update(matched_scope=True,timing_scope_qualified=True,
                native_parity_qualified=evidence['native_parity_qualified'],
                observed_absolute_difference=evidence['observed_absolute_difference'],
                numerical_acceptance=evidence['numerical_acceptance'],
                scope_receipt=str(receipt),scope_receipt_sha256=sha(receipt),
                scope='RGB8 conversion + original call + temporary release; parameters resident; exact protocol controls passed',
                qualification='Descriptive timing scope; numerical acceptance separate',qualified_timing_cells=0)
        plan.update(matched_scope=True,timing_scope_qualified=True,execution_ready=True,
                    qualification='Descriptive matched timing scope; preserve per-case numerical and resource qualifications')
        metric=plan['cases'][0]['metric'];path=destination/(metric+'-reference-plan.json');write(path,plan)
        metric_plans[metric]=str(path)
    write(destination/'index.json',dict(metric_plans=metric_plans,receipt=str(receipt),receipt_sha256=sha(receipt),
        binary=plans[0]['binary'],rust_source=plans[0]['rust_source'],matched_scope=True,
        metric_inference_executed=False,plans_sha256={p:sha(p) for p in metric_plans.values()}))
    for path in destination.iterdir():path.chmod(0o444)
    return dict(plans=metric_plans,receipt_sha256=sha(receipt),metric_inference_executed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('request','output'):parser.add_argument(name,type=Path)
    parser.add_argument('--request-sha',required=True)
    args=parser.parse_args();print(json.dumps(promote(args.request,args.request_sha,args.output)),flush=True)
