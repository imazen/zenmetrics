"""Source/synthetic-only native cohort proposal, separate from Python runtimes."""
import copy
import hashlib
import json
from pathlib import Path
import runtime_closure

ROUTES={
    'butteraugli-original':dict(engine='original-71b18b6',canonical_metric='butteraugli-original',absolute_bound=0.,relative_bound=0.,
        scope='planes + f64 sRGB LUT + ButteraugliInterface inside canonical call; Rust f32 LUT/planes/compute/max inside call',
        blockers=['new source/binary/features binding','f64 versus f32 ingress numeric parity','exact original configuration/version']),
    'butteraugli':dict(engine='libjxl-0.12.0',canonical_metric='butteraugli',absolute_bound=0.,relative_bound=.001,
        scope='canonical BufferToImageBundle x2, metadata/pool/comparator outside; Rust resident-RGB ingress inside',
        blockers=['unmatched ingress/lifecycle','ComputeScore norm versus Rust max unverified','new source/binary/features binding']),
    'ssim2':dict(engine='libjxl-0.12.0',canonical_metric='ssim2',absolute_bound=0.,relative_bound=0.,
        scope='canonical BufferToImageBundle x2 and bundle lifetime outside; Rust resident-RGB ingress inside',
        blockers=['unmatched ingress/lifecycle','new source/binary/features binding','exact numerical parity'])}


def proposal(metric,*,executable,source_receipts,build_receipts,inputs,controller_receipts,loader):
    if metric not in ROUTES:raise ValueError('outside SAME19 native subset')
    groups=dict(source=source_receipts,build=build_receipts,inputs=inputs,controller=controller_receipts)
    if any(not rows for rows in groups.values()):raise ValueError('all identity groups required')
    required={'source_revision','features','overlays','toolchain','build_command','binary_mapping'}
    if {r.get('role') for r in build_receipts}!=required:raise ValueError('native build contract incomplete')
    if len(inputs)!=2 or {r.get('role') for r in inputs}!={'reference','distorted'}:raise ValueError('native pair identity required')
    paths=[dict(role='executable',path=str(executable))]+[r for rows in groups.values() for r in rows]
    files=[]
    for r in paths:
        p=Path(r['path'])
        if not p.is_absolute():raise ValueError('absolute binding required')
        files.append(dict(r,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return dict(schema=1,kind='native-source-proposal',metric=metric,cell=[512,512,1,'score'],
        route=copy.deepcopy(ROUTES[metric]),files=files,loader=copy.deepcopy(loader),
        native_roots=[str(Path(executable).absolute())],runtime_admission=False,scope_qualified=False,
        historical_rows_rebound=False,python_scorer_environment=None)


def synthetic_closure(plan,inspector):
    # No real ldd/ELF loading here. Only explicitly injected fake graph controls.
    if plan.get('kind')!='native-source-proposal':raise ValueError('wrong native plan')
    return runtime_closure.closure(plan['native_roots'],plan['loader'],inspector=inspector)


def verify_files(plan):
    for r in plan['files']:
        if hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()!=r['sha256']:raise ValueError('native artifact changed')
    if plan['metric'] not in ROUTES or plan['route']!=ROUTES[plan['metric']] or plan['native_roots']!=[r['path'] for r in plan['files'] if r['role']=='executable']:
        raise ValueError('native route identity changed')
    if plan['runtime_admission'] is not False or plan['scope_qualified'] is not False:raise ValueError('source proposal cannot admit acquisition')
    return True
