#!/usr/bin/env python3
"""Source-bound original HDR warm numerical observations; no timing promotion.

prepare/audit/cell/finish require caller admission. queue acquires both locks
and run-heavy separately for every cell; it never executes a scorer itself.
"""
import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys


INVENTORY_SHA = '7ab6673f4ac17199e16c1b3a1940c950ccf2c7fc3541fbf820759d7ea1d74428'
PARENT_SHA = '1b435281ae018c58bb9b98e8f876a87ff1ca420512405c119bfcf9cdba8852ec'
PREFLIGHT_SHA = 'db63fa2d8a1f1d60acb36abb282d851ae762fb22e83adfe7c2cd65e437710edf'
BINARY_SHA = '19898e546917c9b62badcf1733ec74aacb3e02b4e14d88af1e31923a88a54005'
SOURCE = 'ea767ba1a4029963bc56a74fb824e64455e4599e'
HELPERS = {
    'first_boundary_hdrvdp_admitted.py': '4d5f396ea6f16d753cf2a3daf77458a5e06b8010a21b1a7cdf924f54179656f6',
    'first_boundary_hdrvdp_resident.py': '858f2ae0ad84dfb306de0173af381332c972462b1975f56797d8c0d3a46c091f',
    'first_boundary_hdrvdp_resident_run.py': '41fb01d98ac7dfd33d0ec2fca3a019d7bc8b192566d2d0bb1237123f9172d86d',
    'first_boundary_hdrvdp_arm_worker.py': '8035916fafc25f4e7a4d9b1028827628cc8f5c42138b1565ae98c7fa716778ce',
}
THREADS = ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'MKL_NUM_THREADS', 'RTEN_NUM_THREADS')
OVERRIDES = ('OCTAVE_PATH', 'OCTAVE_EXEC_PATH', 'OCTAVE_HOME',
             'OCTAVE_SITE_INITFILE', 'OCTAVE_VERSION_INITFILE', 'LD_PRELOAD', 'LD_AUDIT')


class Drift(ValueError):
    """Evidence/source change: stop dispatching new cases."""


class Protocol(ValueError):
    """Retained child failure: continue independent cases."""


def require(value, message):
    if not value:
        raise Drift(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def strict(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('nonfinite JSON number')
        if isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)
    result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    finite(result)
    return result


def load(path):
    return strict(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify(bindings):
    # Hashed source/assets are opaque, even if their filename ends in .json.
    for path, expected in bindings.items():
        require(sha(path) == expected, 'bound bytes changed: ' + path)


def pinned(path, expected):
    verify({str(path): expected})
    return load(path)


def modules(directory):
    verify({str(directory/name): digest for name, digest in HELPERS.items()})
    sys.path.insert(0, str(directory))
    names = ('first_boundary_hdrvdp_arm_worker', 'first_boundary_hdrvdp_admitted')
    result = [importlib.import_module(name) for name in names]
    for module in result:
        require(Path(module.__file__).resolve() == (directory/(module.__name__+'.py')).resolve(),
                'helper import resolution differs')
    return result


def environment(threads, scratch):
    require(threads in (1, 28), 'explicit original1/28 budgets required')
    require(not any(os.environ.get(k) for k in OVERRIDES), 'unbound runtime override')
    env = dict(os.environ, TMPDIR=str(scratch), LC_ALL='C', ZENBENCH_NO_SAVE='1')
    env.update({key: str(threads) for key in THREADS})
    return env


def selected(plan, inventory):
    cases = [c for c in plan['cases'] if c['panel']=='original12' and c['phase']=='warm']
    require(len(cases)==48 and len({c['id'] for c in cases})==48, 'unique48 selection required')
    require([c['id'] for c in cases]==[c['id'] for c in inventory['cases']], 'inventory order differs')
    for c, row in zip(cases, inventory['cases']):
        require(c['metric'] in ('hdrvdp','hdrvdp3') and c['threads'] in (1,28)
                and c['samples']==5 and c['matched_scope'] is False, 'warm grid contract differs')
        for key in ('metric','width','height','threads','evaluator_body_sha256','absolute_bound','relative_bound'):
            require(row[key]==c[key], 'inventory field differs: '+key)
        require(c['absolute_bound']==(0.005 if c['metric']=='hdrvdp' else None)
                and c['relative_bound']==0, 'unchanged bound required')
        for role in ('reference','distorted'):
            require(row[role]['path']==c[role]
                    and row[role]['sha256_from_frozen_binding']==plan['bindings'][c[role]],
                    'input identity differs')
        command = c['arms']['default']['command']
        require(command==row['reference_default_command'] and command[1:-1]==[
            '--no-gui','--no-window-system','--no-init-file','--no-history','--quiet'],
            'exact headless original command required')
        require(c['native']['command']==row['native_command']==[c['binary'],c['native_config']]
                and c['environment']=={k:str(c['threads']) for k in THREADS}, 'native argv/budget differs')
        text = Path(command[-1]).read_text()
        require(text.count('clock=tic();\n')==1 and 'for iteration=1:5\n' in text,
                'original five-sample wrapper differs')
        body = text.split('clock=tic();\n')[1].split('elapsed=toc(clock)')[0]
        require(hashlib.sha256(body.encode()).hexdigest()==c['evaluator_body_sha256'],
                'original evaluator body differs')
        config = load(c['native_config'])
        expected = dict(width=c['width'], height=c['height'], threads=c['threads'],
            reference=c['reference'], distorted=c['distorted'], metric_name=c['metric'],
            only_mode='score', timing_phase='warm', probe_only=False, required_samples=5,
            sample_target_ns=10000000, chunk_budget_ms=240000, locked_process_accounting=True)
        require(config==expected, 'exact retained native config required')
    return cases


def native_evidence(request, plan):
    records = {name: pinned(record['path'], record['sha256'])
               for name, record in request['native_evidence'].items()}
    require(set(records)=={'build','verified','reproducible'}, 'three native evidence roots required')
    for record in records.values():
        require(record['source_revision']==SOURCE and record['binary_sha256']==BINARY_SHA,
                'native source/binary receipt differs')
    require(records['build']['adapters_applied'] is False
            and records['build']['features']==['compete','metrics-parallel','zenmetrics-cli/nr'],
            'unchanged production worker required')
    require(plan['rust_source']==SOURCE and plan['binary_sha256']==BINARY_SHA, 'native parent differs')
    verify({str(Path(request['native_source_root'])/k): v
            for k,v in records['reproducible']['source_files'].items()})


def prepare(request_path, expected, root):
    request = pinned(request_path, expected)
    require(request['schema']==1, 'explicit request schema required')
    inventory = pinned(request['inventory'], INVENTORY_SHA)
    plan = pinned(request['parent_plan'], PARENT_SHA)
    pinned(request['preflight_receipt'], PREFLIGHT_SHA)
    verify(request['sources'])
    admission=request['admission']
    require(admission['memory']=='12G' and len(admission['locks'])==2
            and len({str(Path(p).resolve()) for p in admission['locks']})==2
            and all(Path(p).is_absolute() for p in admission['locks'])
            and {admission['wrapper'],admission['python'],'/usr/bin/flock'}<=set(admission['tools']),
            'explicit pinned admission tools and two distinct locks required')
    verify(admission['tools'])
    require(request['sources'].get(str(Path(__file__).resolve()))==sha(__file__), 'controller not bound')
    require(inventory['parent_plan']==request['parent_plan']
            and inventory['parent_plan_sha256']==PARENT_SHA and plan['headless'] is True,
            'exact headless lineage required')
    verify(plan['bindings'])
    cases = selected(plan, inventory)
    native_evidence(request, plan)
    root.mkdir()  # Never reuse output, even an empty directory.
    root = root.resolve()
    helper_dir = root/'helpers'; helper_dir.mkdir()
    bindings = dict(request['sources'])
    for name, digest in HELPERS.items():
        source = Path(request['parent_plan']).parent/'helpers'/name
        verify({str(source):digest})
        target = helper_dir/name; shutil.copy2(source,target);bindings[str(target)]=digest
    controller = helper_dir/Path(__file__).name
    shutil.copy2(__file__,controller);bindings[str(controller)]=sha(__file__)
    request_copy = root/'request.json';shutil.copy2(request_path,request_copy)
    bindings[str(request_copy)]=expected
    for key in ('inventory','preflight_receipt'):
        bindings[request[key]] = sha(request[key])
    for record in request['native_evidence'].values(): bindings[record['path']]=record['sha256']
    for path, digest in request['admission']['tools'].items(): bindings[path]=digest
    write(root/'case-specs.json',plan['cases']);bindings[str(root/'case-specs.json')]=sha(root/'case-specs.json')
    catalog = [dict(index=i,id=c['id'],threads=c['threads'],order=order(i)) for i,c in enumerate(cases)]
    write(root/'catalog.json',catalog);bindings[str(root/'catalog.json')]=sha(root/'catalog.json')
    old = [p for p in plan['bindings'] if Path(p).name=='original12-warm.plan.json']
    require(len(old)==1, 'original gate plan required')
    old_plan = load(old[0])
    gates = {key:dict(text=old_plan[key],status='incomplete')
             for key in ('exact_original_call_gate','native_thread_gate')}
    gates['default_resident_exact']=dict(text='Original default/resident exact control, order '+
        repr(plan['expected_control_order']),status='incomplete')
    gates['native_first_call']=dict(text='Fresh-process first-call numerical qualification',status='incomplete')
    gates['native_timed_score_stability']=dict(text='Every timed CPU return observed',status='unobserved')
    cohort = dict(schema=1,kind='original104',parent_plan=request['parent_plan'],parent_plan_sha256=PARENT_SHA,
        preflight_receipt=request['preflight_receipt'],case_specs=str(root/'case-specs.json'),
        bindings=bindings,request=str(request_copy),catalog=str(root/'catalog.json'),old_gates=gates,
        expected_cases=48,expected_children=96,admission=request['admission'],matched_scope=False,
        qualified_timing_cells=0,ratio=None,protocol='one-default-one-native-alternating-by-case')
    write(root/'cohort.json',cohort)
    result = dict(root=str(root),cohort_sha256=sha(root/'cohort.json'),expected_cases=48,expected_children=96)
    write(root/'prepared.json',result)
    print(json.dumps(result),flush=True)


def order(index):
    return ['default','native'] if index%2==0 else ['native','default']


def context(root, expected):
    cohort = pinned(root/'cohort.json',expected)
    verify(cohort['bindings'])
    worker, admitted = modules(root/'helpers')
    checked, plan = worker.context(root,expected)
    require(checked==cohort, 'helper context disagrees')
    request = load(cohort['request'])
    require(sha(request['inventory'])==INVENTORY_SHA and sha(request['parent_plan'])==PARENT_SHA,
            'pinned selection/parent differs')
    cases = selected(plan,load(request['inventory']))
    require(load(cohort['catalog'])==[dict(index=i,id=c['id'],threads=c['threads'],order=order(i))
                                    for i,c in enumerate(cases)], 'catalog differs')
    native_evidence(request,plan)
    return cohort,plan,cases,worker,admitted


def native_snapshot(admitted, plan, bundle, directory, env, binary):
    # Reuse the original recursive ELF resolver with the native executable as
    # an explicit ELF root, without recursively scanning a Cargo target tree.
    return admitted.capture(dict(plan,runtime_roots=[],octave=binary),bundle,directory,env)


def audit(root, expected):
    cohort,plan,cases,worker,admitted = context(root,expected)
    worker.seal(root,expected,PREFLIGHT_SHA)
    scratch=root/'audit-scratch';scratch.mkdir()
    env=environment(1,scratch)
    bundle=Path(cohort['parent_plan']).parent
    before=admitted.capture(plan,bundle,root/'audit-runtime-before',env)
    preflight=worker.checked_seal(root,cohort,expected,sha(root/'preflight-seal.json'))
    require(worker.invariant(before)==preflight['baseline'], 'installed runtime differs from passed preflight')
    native=native_snapshot(admitted,plan,bundle,root/'audit-native-before',env,cases[0]['binary'])
    after=admitted.capture(plan,bundle,root/'audit-runtime-after',env)
    native_after=native_snapshot(admitted,plan,bundle,root/'audit-native-after',env,cases[0]['binary'])
    require(before==after and native==native_after, 'setup runtime closure changed')
    context(root,expected)
    record=dict(schema=1,passed=True,cohort_sha256=expected,preflight_seal_sha256=sha(root/'preflight-seal.json'),
        native_baseline=worker.invariant(native),runtime_baseline=preflight['baseline'],
        artifacts={str(p):sha(p) for folder in ('audit-runtime-before','audit-runtime-after',
                   'audit-native-before','audit-native-after') for p in (root/folder).rglob('*') if p.is_file()},
        expected_cases=48,expected_children=96,acquired_children=0,ratio=None,matched_scope=False,
        qualified_timing_cells=0,old_gates=cohort['old_gates'])
    write(root/'setup.json',record)
    print(json.dumps(dict(setup_sha256=sha(root/'setup.json'),passed=True,acquired_children=0)),flush=True)


def checked_setup(root, cohort, expected, setup_sha, worker):
    setup=pinned(root/'setup.json',setup_sha)
    require(setup['passed'] is True and setup['cohort_sha256']==expected
            and setup['expected_cases']==48 and setup['expected_children']==96
            and setup['acquired_children']==0 and setup['ratio'] is None
            and setup['matched_scope'] is False and setup['qualified_timing_cells']==0
            and setup['old_gates']==cohort['old_gates'], 'setup provenance differs')
    verify(setup['artifacts'])
    actual={path for folder in ('audit-runtime-before','audit-runtime-after',
            'audit-native-before','audit-native-after') for path in worker.files_under(root/folder)}
    require(actual==set(setup['artifacts']), 'setup artifact inventory differs')
    seal=worker.checked_seal(root,cohort,expected,setup['preflight_seal_sha256'])
    require(setup['runtime_baseline']==seal['baseline'], 'setup/preflight closure differs')
    return setup


def bits(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise Protocol('finite scalar required')
    return struct.pack('>d',value).hex()


def parse_default(text, worker):
    lines=[line.split() for line in text.splitlines() if line.startswith(('COLD ','SAMPLE '))]
    if [row[0] for row in lines]!=['SAMPLE']*5:
        raise Protocol('exactly five warm SAMPLE records required')
    try:
        observed=worker.scalar_bits(text,5,False)
    except (ValueError,OverflowError) as error:
        raise Protocol(str(error)) from error
    return dict(score=float(lines[0][2]),score_bits=observed,
                samples=[dict(elapsed_ns=float(row[1]),score=float(row[2])) for row in lines])


def parse_native(text, config):
    try:
        if config.get('probe_only') is not False:
            raise Protocol('bound config must explicitly disable probe-only execution')
        rows=[strict(line[8:]) for line in text.splitlines() if line.startswith('IQA_ROW ')]
        warm=[strict(line[11:]) for line in text.splitlines() if line.startswith('IQA_WARMUP ')]
        if len(rows)!=1 or len(warm)!=1 or any(line.startswith(('IQA_PARTIAL ','IQA_PROBE ')) for line in text.splitlines()):
            raise Protocol('one complete native scalar row/warmup required')
        row=rows[0]
        # Production rows clone their actual execution config. Existing normal
        # warm configs omit this optional field; exact grid configs include false.
        # Neither shape permits probe execution. All other config keys must emit.
        if any(type(row.get(k)) is not type(v) or row[k]!=v for k,v in config.items() if k!='probe_only'):
            raise Protocol('native config/identity differs')
        if (row.get('probe_only',False) is not False or row.get('mode')!='score'
                or row.get('only_mode')!='score' or row.get('timing_phase')!='warm'
                or row.get('model_preload_adapter') is not False):
            raise Protocol('unexpected native execution mode/adapter')
        observed=bits(row['score'])
        if (row['status']!='ok' or row['rayon_threads_observed']!=config['threads']
                or row['score_bits']!=observed or bits(row['self_warmup_score'])!=observed
                or bits(row['ref_score'])!=observed or warm[0]['completed'] is not True
                or warm[0]['score_bits']!=observed or row['map_score_matches_scalar_bits'] is not True):
            raise Protocol('native warm scalar/budget protocol differs')
        times=row['round_means_ns']; samples=row['zenbench_samples']
        if (type(row['n']) is not int or row['n']!=10 or len(times)!=10 or len(samples)!=10
                or any(type(t) not in (int,float) or not math.isfinite(t) or t<=0 for t in times)
                or any(type(s['gate_clean']) is not bool for s in samples)):
            raise Protocol('complete ten native warm rounds/resource flags required')
        return dict(score=row['score'],score_bits=observed,raw_row=row,
            resource_gate_clean=[s['gate_clean'] for s in samples],
            scalar_scope='initialization/warmup result; timed CPU returns not individually observed')
    except (KeyError,TypeError,ValueError,OverflowError) as error:
        raise Protocol(str(error)) from error


def observation(role, text, returncode, case, worker):
    result=dict(role=role,returncode=returncode,complete=False)
    if type(returncode) is not int or returncode!=0:
        result['error']='nonzero/timeout child exit'
        return result
    try:
        result.update(parse_default(text,worker) if role=='default' else parse_native(text,load(case['native_config'])))
        result['complete']=True
    except (Protocol,ValueError,IndexError,TypeError) as error:
        result['error']=str(error)
    return result


def compare(case, arms):
    complete=all(arms[r]['complete'] is True for r in ('default','native'))
    delta=abs(arms['native']['score']-arms['default']['score']) if complete else None
    overflow=delta is not None and not math.isfinite(delta)
    if overflow:
        complete=False;delta=None
    bounded=(delta<=0.005 if case['metric']=='hdrvdp' and complete else None)
    return dict(observation_complete=complete,absolute_difference=delta,difference_overflow=overflow,
        absolute_bound=case['absolute_bound'],relative_bound=0,bound_pass=bounded,
        native_parity_qualified=False,ratio=None,matched_scope=False,qualified_timing_cells=0)


def snapshots(root, role, plan, cohort, case, setup, worker, admitted, env):
    bundle=Path(cohort['parent_plan']).parent
    runtime=admitted.capture(plan,bundle,root/(role+'-runtime'),env)
    native=native_snapshot(admitted,plan,bundle,root/(role+'-native'),env,case['binary'])
    require(worker.invariant(runtime)==setup['runtime_baseline']
            and worker.invariant(native)==setup['native_baseline'], 'runtime/dependency closure drift')
    return runtime,native


def cell(root, expected, setup_sha, case_id, admission):
    cohort,plan,cases,worker,admitted=context(root,expected)
    setup=checked_setup(root,cohort,expected,setup_sha,worker)
    selected_cases=[(i,c) for i,c in enumerate(cases) if c['id']==case_id]
    require(len(selected_cases)==1,'exact frozen case ID required')
    index,case=selected_cases[0]
    require(admission==cohort['admission'], 'explicit outer admission differs from bound request')
    require(all(os.environ.get(k)==str(case['threads']) for k in THREADS), 'outer admission budget differs')
    base=root/'cells';base.mkdir(exist_ok=True)
    directory=base/case_id;directory.mkdir()
    record=dict(schema=1,id=case_id,index=index,cohort_sha256=expected,setup_sha256=setup_sha,
                order=order(index),arms={},fatal=False,old_gates=cohort['old_gates'])
    try:
        for role in order(index):
            child=directory/role;child.mkdir();scratch=child/'scratch';scratch.mkdir()
            env=environment(case['threads'],scratch)
            context(root,expected);checked_setup(root,cohort,expected,setup_sha,worker)
            before=snapshots(child,'before',plan,cohort,case,setup,worker,admitted,env)
            command=case['arms']['default']['command'] if role=='default' else case['native']['command']
            write(child/'environment.json',env)
            exit_record=admitted.execute(command,env,child,'control',1100)
            write(child/'exit.json',exit_record)
            after=snapshots(child,'after',plan,cohort,case,setup,worker,admitted,env)
            require(before==after,'runtime closure changed during child')
            context(root,expected);checked_setup(root,cohort,expected,setup_sha,worker)
            record['arms'][role]=observation(role,(child/'control-stdout.log').read_text(),
                                            exit_record['returncode'],case,worker)
        record.update(compare(case,record['arms']))
    except Exception as error:
        record.update(fatal=True,error=str(error),ratio=None,matched_scope=False,qualified_timing_cells=0)
    record['artifacts']=worker.files_under(directory)
    write(directory/'result.json',record)
    print(json.dumps(record),flush=True)
    return 2 if record['fatal'] else 0 if record['observation_complete'] and record['bound_pass'] is not False else 1


def audit_cell(root, case, index, expected, setup_sha, worker, setup):
    directory=root/'cells'/case['id'];record=load(directory/'result.json')
    require(record['id']==case['id'] and record['index']==index and record['cohort_sha256']==expected
            and record['setup_sha256']==setup_sha and record['order']==order(index)
            and record['fatal'] is False, 'retained cell identity/fatal outcome differs')
    verify(record['artifacts'])
    require(set(worker.files_under(directory))==set(record['artifacts'])|{str(directory/'result.json')},
            'retained cell artifact inventory differs')
    arms={}
    for role in order(index):
        child=directory/role;exit_record=load(child/'exit.json')
        command=case['arms']['default']['command'] if role=='default' else case['native']['command']
        require(load(child/'control-command.json')==['/usr/bin/time','-v','-o',str(child/'control-memory.txt'),
            '/usr/bin/timeout','--kill-after=20s','1100s',*command], 'retained child argv differs')
        env=load(child/'environment.json')
        require(all(env.get(k)==str(case['threads']) for k in THREADS)
                and not any(env.get(k) for k in OVERRIDES), 'retained environment differs')
        for suffix, baseline in (('runtime',setup['runtime_baseline']),('native',setup['native_baseline'])):
            before=load(child/('before-'+suffix)/'snapshot.json')
            after=load(child/('after-'+suffix)/'snapshot.json')
            require(before==after and worker.invariant(before)==baseline, 'retained runtime closure differs')
        for stream in ('stdout','stderr'):
            require(sha(child/('control-'+stream+'.log'))==exit_record[stream+'_sha256'], 'child log differs')
        arms[role]=observation(role,(child/'control-stdout.log').read_text(),exit_record['returncode'],case,worker)
    require(record['arms']==arms,'retained parsed observations differ')
    for key,value in compare(case,arms).items():require(record[key]==value,'retained numerical gate differs')
    return record


def finish(root, expected, setup_sha):
    cohort,plan,cases,worker,admitted=context(root,expected)
    setup=checked_setup(root,cohort,expected,setup_sha,worker)
    records=[audit_cell(root,c,i,expected,setup_sha,worker,setup) for i,c in enumerate(cases)]
    require({p.name for p in (root/'cells').iterdir()}=={c['id'] for c in cases}, 'exact48 cell directories required')
    comparisons=[]
    for metric in ('hdrvdp','hdrvdp3'):
        for reference in sorted({c['reference'] for c in cases}):
            pair=[(c,r) for c,r in zip(cases,records) if c['metric']==metric and c['reference']==reference]
            native={str(c['threads']):r['arms']['native'].get('score_bits') for c,r in pair}
            comparisons.append(dict(metric=metric,reference=reference,bits_by_threads=native,
                exact_match=(native['1']==native['28'] if all(native.values()) else None),
                discharges_first_warm_thread_gate=False))
    summary=dict(schema=1,cohort_sha256=expected,setup_sha256=setup_sha,
        expected_cases=48,expected_children=96,acquired_cases=len(records),
        acquired_children=sum(len(r['arms']) for r in records),
        complete_observations=sum(r['observation_complete'] for r in records),
        failed_cases=[r['id'] for r in records if not r['observation_complete'] or r['bound_pass'] is False],
        v2_bound_pass_count=sum(r['bound_pass'] is True for r in records),
        v3_unbounded_observations=sum(r['observation_complete'] and c['metric']=='hdrvdp3' for c,r in zip(cases,records)),
        native_warm_thread_comparisons=comparisons,old_gates=cohort['old_gates'],
        ratio=None,matched_scope=False,qualified_timing_cells=0,native_parity_qualified=False,
        cell_results={str(root/'cells'/c['id']/'result.json'):sha(root/'cells'/c['id']/'result.json') for c in cases})
    write(root/'result.json',summary)
    print(json.dumps(summary),flush=True)
    return 1 if summary['failed_cases'] else 0


def check(root, expected, setup_sha, case_id):
    cohort,plan,cases,worker,admitted=context(root,expected)
    setup=checked_setup(root,cohort,expected,setup_sha,worker)
    found=[(i,c) for i,c in enumerate(cases) if c['id']==case_id]
    require(len(found)==1,'exact control case ID required')
    i,c=found[0]
    record=audit_cell(root,c,i,expected,setup_sha,worker,setup)
    print(json.dumps(dict(id=case_id,observation_complete=record['observation_complete'],
        bound_pass=record['bound_pass'],ratio=None,matched_scope=False)),flush=True)
    return 0 if record['observation_complete'] and record['bound_pass'] is not False else 1


def queue(args):
    # Only small scheduling metadata is read here. ALL evidence hashes, runtime
    # scans, retained audits and scoring are inside each admitted child.
    catalog=load(args.root/'catalog.json')
    admission=load(args.root/'cohort.json')['admission']
    require(str(args.heavy_wrapper)==admission['wrapper']
            and [str(p) for p in args.lock]==admission['locks']
            and admission['memory']=='12G' and admission['python']==sys.executable,
            'outer dispatch admission differs')
    require(len(catalog)==48 and [c['index'] for c in catalog]==list(range(48)), '48-case catalog required')
    require(len(args.lock)==2 and args.lock[0]!=args.lock[1], 'two distinct explicit locks required')
    with (args.root/'coordinator.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        chosen=[c for c in catalog if args.case_id is None or c['id']==args.case_id]
        require(bool(chosen), 'case selector did not match')
        directory=args.root/'dispatch';directory.mkdir(exist_ok=True)
        failed=False
        for c in chosen:
            require(c['threads'] in (1,28) and Path(c['id']).name==c['id'], 'unsafe catalog entry')
            logdir=directory/c['id'];logdir.mkdir();scratch=logdir/'scratch';scratch.mkdir()
            command=[]
            for lock in args.lock:command += ['/usr/bin/flock','-x',str(lock)]
            command += [str(args.heavy_wrapper),'--mem','12G','--jobs',str(c['threads']),'--',
                'env','TMPDIR='+str(scratch),*[key+'='+str(c['threads']) for key in THREADS],
                sys.executable,'-I','-B',str(Path(__file__).resolve()),
                'cell',str(args.root),'--cohort-sha',args.cohort_sha,'--setup-sha',args.setup_sha,
                '--case-id',c['id'],'--admission',json.dumps(admission)]
            write(logdir/'command.json',command)
            print(json.dumps(dict(state='waiting_for_admission',id=c['id'],index=c['index'],log=str(logdir/'wrapper.log'))),flush=True)
            with (logdir/'wrapper.log').open('x') as stream:
                child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                write(logdir/'process.json',dict(pid=child.pid));rc=child.wait()
            write(logdir/'exit.json',dict(returncode=rc))
            print(json.dumps(dict(state='returned',id=c['id'],returncode=rc)),flush=True)
            if rc not in (0,1):return rc or 2
            # A wrapper admission refusal can also return1. Only a completed
            # worker with two retained role outcomes is an ordinary cell failure.
            try:
                marker=load(args.root/'cells'/c['id']/'result.json')
                require(marker['id']==c['id'] and marker['fatal'] is False
                    and marker['cohort_sha256']==args.cohort_sha and marker['setup_sha256']==args.setup_sha
                    and set(marker['arms'])=={'default','native'}
                    and type(marker['observation_complete']) is bool
                    and marker['ratio'] is None and marker['matched_scope'] is False
                    and marker['qualified_timing_cells']==0, 'incomplete worker marker')
                require(rc==(0 if marker['observation_complete'] and marker['bound_pass'] is not False else 1),
                        'worker marker/exit differ')
            except Exception as error:
                print(json.dumps(dict(state='incomplete_worker',id=c['id'],error=str(error))),flush=True)
                return 2
            failed |= rc!=0
        return 1 if failed else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare');p.add_argument('request',type=Path);p.add_argument('root',type=Path)
    p.add_argument('--request-sha',required=True)
    for name in ('audit','cell','finish','check','queue'):
        p=sub.add_parser(name);p.add_argument('root',type=Path);p.add_argument('--cohort-sha',required=True)
        if name!='audit':p.add_argument('--setup-sha',required=True)
        if name in ('cell','check','queue'):p.add_argument('--case-id',required=name!='queue')
        if name=='cell':p.add_argument('--admission',required=True)
        if name=='queue':
            p.add_argument('--heavy-wrapper',type=Path,required=True)
            p.add_argument('--lock',type=Path,action='append',required=True)
    a=parser.parse_args()
    if a.mode=='prepare':prepare(a.request,a.request_sha,a.root);return 0
    a.root=a.root.resolve(strict=True)
    if a.mode=='audit':audit(a.root,a.cohort_sha);return 0
    if a.mode=='cell':return cell(a.root,a.cohort_sha,a.setup_sha,a.case_id,strict(a.admission))
    if a.mode=='finish':return finish(a.root,a.cohort_sha,a.setup_sha)
    if a.mode=='check':return check(a.root,a.cohort_sha,a.setup_sha,a.case_id)
    return queue(a)


if __name__=='__main__':
    try:
        code=main()
    except Exception as error:
        print(json.dumps(dict(fatal=True,error=str(error))),flush=True)
        code=2
    raise SystemExit(code)
