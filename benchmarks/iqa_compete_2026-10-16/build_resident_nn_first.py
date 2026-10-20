#!/usr/bin/env python3
"""Link first-evaluation probes to frozen NN libraries with resident model caches."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from compare_canonical import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference_plan',type=Path);p.add_argument('target',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();plan=json.loads(a.reference_plan.read_text());target=a.target.resolve(strict=True)
    worker=target/'release/iqa-compete'
    if plan['bindings'].get(str(worker))!=sha(worker):raise ValueError('NN worker differs from the frozen reference plan')
    for asset,expected in plan['bindings'].items():
        if sha(asset)!=expected:raise ValueError('reference binding changed: '+asset)
    a.output.mkdir();out=a.output.resolve();deps=target/'release/deps';libraries={}
    for name in ['zenmetrics_cli','zenmetrics_abstractions','zenmetrics_nn','zenmetrics_pyiqa_ports','rayon']:
        paths=list(deps.glob('lib'+name+'-*.rlib'))
        if len(paths)!=1:raise ValueError('one frozen library required: '+name)
        libraries[name]=paths[0]
    sources={}
    declaration='    static MODELS: OnceLock<Models> = OnceLock::new();\n'
    for module,source in [('original_nn','nn'),('pyiqa_nn','pyiqa_ports')]:
        relative='crates/zenmetrics-cli/src/metrics/'+source+'.rs'
        original=subprocess.check_output(['git','show',plan['rust_source']+':'+relative])
        text=original.decode()
        if text.count(declaration)!=1 or 'benchmark_prepare' in text:raise ValueError('review changed CLI cache declaration')
        # A function-local static already has process lifetime. Hoisting this
        # one declaration lets construction prime the same cache without a call.
        adapted=text.replace(declaration,'',1)
        helpers='''
// Benchmark-only access to the existing cache; score() and its lock/evaluation body stay unchanged.
static MODELS: OnceLock<Models> = OnceLock::new();
pub(super) fn benchmark_prepare(kind: MetricKind) -> Result<(), Box<dyn std::error::Error>> {
    let column = kind.column_names()[0];
    let mut models = MODELS.get_or_init(Default::default).lock().map_err(|_| "model cache poisoned")?;
    if !models.contains_key(column) { models.insert(column, load(column)?); }
    Ok(())
}
'''
        path=out/(module+'.rs');path.write_text(adapted+helpers)
        sources[relative]={'original_sha256':hashlib.sha256(original).hexdigest(),'probe_module_sha256':sha(path),
            'scope':'Only hoist the original static cache and append a cache-priming entry; original score locking, image ingress, numerical evaluation and result extraction remain unchanged'}
    here=Path(__file__).resolve().parent
    source=out/'resident_nn_first.rs';shutil.copy2(here/'resident_nn_first.rs',source)
    timing=out/'timing.rs';timing.write_bytes(subprocess.check_output(['git','show',plan['rust_source']+':benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete_timing/mod.rs']))
    binary=out/'resident-nn-first'
    command=['rustc','--edition=2024','-C','opt-level=3','-L','dependency='+str(deps),'--cfg','feature="nn-ort"',
        *[part for name,path in libraries.items() for part in ['--extern',name+'='+str(path)]],str(source),'-o',str(binary)]
    subprocess.run(command,check=True)
    value={'source_revision':plan['rust_source'],'reference_plan_sha256':sha(a.reference_plan),'worker_sha256':sha(worker),
        'library_sha256':{name:sha(path) for name,path in libraries.items()},'source_modules':sources,
        'probe_source_sha256':sha(source),'cold_timer_source_sha256':sha(timing),'binary_sha256':sha(binary),
        'builder_sha256':sha(__file__),'command':command,
        'scope':'First evaluation after model cache priming, with RGB8 inputs resident; uses the stock CLI NN scoring body and existing cold_call timer, excluding model loading and outer CLI dispatch; a second evaluation checks scalar bit stability outside the clock'}
    (out/'build-receipt.json').write_text(json.dumps(value,indent=2)+'\n')
    print('BUILT_RESIDENT_FIRST',value['binary_sha256'],flush=True)


if __name__=='__main__':main()
