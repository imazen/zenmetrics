"""Freeze a source-bound build-only plan; never launches a binary or admits scoring."""
import argparse,hashlib,json
from pathlib import Path

ADAPTER='62aaf7e94acaee2ccbe9b4f6c16f0ec3416a3559'
BASE='73c04e413fddc0e95e66b4147785910253772192'
BLOBS={'check_libjxl_scope.py':'82ba34a601ac20493def9d5ae3deab6760fc544c','libjxl_compute.cpp':'215f023d9c095bf3f218f3cbc69f07eb674ac1ea','native-rgb-scope.md':'985c0dfd3d86a7d80a51430557699f7ed01e3777'}
def plan(receipts,target):
    names=('adapter-binding.json','locked-check.json','inputs.json','rust-overlay.json','sibling-exports.json','dispatch-source-hashes.json')
    data={n:json.loads((receipts/n).read_text()) for n in names}
    adapter=data['adapter-binding.json']
    if adapter['commit']!=ADAPTER:raise ValueError('adapter commit')
    rows={Path(r['path']).name:r for r in adapter['files']}
    if any(rows[n]['git_blob']!=v for n,v in BLOBS.items()):raise ValueError('reviewed adapter blob')
    if data['locked-check.json']['passed'] is not True:raise ValueError('locked metadata failed')
    pair=data['inputs.json']
    if {r['role'] for r in pair}!={'reference','distorted'} or len(pair)!=2:raise ValueError('paired input binding')
    if any((r['width'],r['height'],r['stride_bytes'],r['layout'])!=(512,512,1536,'RGB8 packed sRGB') for r in pair):raise ValueError('input layout')
    hashes={r['path']:r['sha256'] for r in data['dispatch-source-hashes.json']}
    if hashes['work/zenmetrics/Cargo.lock']!=data['locked-check.json']['cargo_lock_sha256']:raise ValueError('lock identity')
    harness='work/zenmetrics/benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete.rs'
    if hashes[harness]!=data['rust-overlay.json']['after_sha256']:raise ValueError('benchmark overlay identity')
    return dict(schema=1,kind='native-build-plan-only',metric='butteraugli',selected_scalar='butteraugli_max',cell=dict(width=512,height=512,threads=1,mode='score'),rust_revision=BASE,adapter_revision=ADAPTER,
        receipts={n:hashlib.sha256((receipts/n).read_bytes()).hexdigest() for n in names},sources=hashes,inputs=pair,siblings=data['sibling-exports.json'],
        rust_features=['compete','metrics-parallel'],nr_or_pyiqa_dependency=False,scope_id='rgb-input',scope_qualified=False,
        next_action=dict(kind='fresh compilation only',target_root=str(target),argv=['python3',str(target/'native_build.py'),'build',str(target)],required_admission='existing two locks in established order plus run-heavy; resolve live identity before use',
            preconditions=['fresh build output paths absent','source/overlay/input hashes revalidated before compile','locked metadata still passes without lock changes','build helper source bound and build plan independently reviewable']),
        required_after_build=['actual two executable hashes','compiler/linker/header/static library/feature/overlay binding','transitive ELF and loader configuration closure','used controller interpreter/modules/bytecode binding','independent cohort/protocol/scope review'],
        not_executed=['metric compilation','scorer','warmup','parity','timing'],acquisition_authorized=False,attribution_ready=False,
        remaining_gates=['generic tied-fixture approval and full controls','unchanged numerical parity bounds on exact input','actual scorer gate/ownership','order/selection/dependence model and strict precision semantics'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipts',type=Path);p.add_argument('target',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    value=plan(a.receipts,a.target)
    with a.output.open('x') as f:json.dump(value,f,indent=2)
