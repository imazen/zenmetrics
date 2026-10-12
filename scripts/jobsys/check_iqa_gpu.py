#!/usr/bin/env python3
"""GPU functional controls; preserve every failure and certify no performance."""
import json
import os
from pathlib import Path
import subprocess
import sys

binaries=Path(sys.argv[1]).resolve();backend=sys.argv[2];root=Path(sys.argv[3]).resolve();adapter=sys.argv[4]
assert backend in ('cuda','wgpu')
root.mkdir(parents=True,exist_ok=False)
xdg=root/'xdg';xdg.mkdir(mode=0o700)
w,h=256,256
reference=bytes((x*17+y*13+c*41)%256 for y in range(h) for x in range(w) for c in range(3))
distorted=bytes(max(0,value-7) for value in reference)
for name,data in [('reference',reference),('distorted',distorted)]:
    for layout,padding in [('packed',b''),('padded',bytes([99]*7))]:
        (root/f'{name}-{layout}').write_bytes(b''.join(data[y*w*3:(y+1)*w*3]+padding for y in range(h)))
env=dict(os.environ,RAYON_NUM_THREADS='1',ZENBENCH_OPERATIONS='2',TMPDIR=str(Path.home()/'tmp'),XDG_RUNTIME_DIR=str(xdg))
env.pop('ZENBENCH_RESULT_FILE',None);env.pop('ZENMETRICS_PAIR_TIMING',None)
controls=[];failures=[]
for metric in ['butteraugli-gpu','cvvdp-gpu','dssim-gpu','iwssim-gpu','ssim2-gpu','zensim-gpu']:
    for reuse in ['cold','warm_instance']:
        previous=None
        for layout,padding in [('packed',0),('padded',7)]:
            label=f'{metric}-{reuse}-{layout}'
            case=dict(schema=4,metric=metric,width=w,height=h,stride_bytes=w*3+padding,
                      reference=f'reference-{layout}',distorted=f'distorted-{layout}',pixel_format='rgb8_srgb',
                      reuse=reuse,threads=1,products='all',warmup_operations=0,backend=backend,expected_adapter=adapter,
                      memory_mode='full',parameter_preset='api-defaults-standard-fhd')
            path=root/f'{label}.json';path.write_text(json.dumps(case));outputs=[]
            for variant in ['latency','memory']:
                try:
                    result=subprocess.run([str(binaries/variant),'fleet',str(path)],env=env,capture_output=True,timeout=120)
                    (root/f'{label}-{variant}.stdout').write_bytes(result.stdout)
                    (root/f'{label}-{variant}.stderr').write_bytes(result.stderr)
                    if result.returncode or b'panicked at' in result.stderr or b'cubecl-wgpu:' in result.stderr:
                        raise ValueError('execution failed or a device-service error was observed')
                    value=json.loads(result.stdout)
                    if value['timing_usable']!=(variant=='latency'): raise ValueError('instrumentation boundary mismatch')
                    products=value['driver']['correctness']['products']
                    if len(products)!=2 or not products[0] or products[0]!=products[1]:
                        raise ValueError('scalar products differ across repeated operations')
                    outputs.append(products)
                except (ValueError,KeyError,subprocess.TimeoutExpired) as error:
                    failures.append(dict(case=label,variant=variant,reason=str(error)))
            if len(outputs)==2:
                if outputs[0]!=outputs[1]: failures.append(dict(case=label,reason='instrumentation changed scalar bits'))
                if previous is not None and previous!=outputs[0]: failures.append(dict(case=label,reason='padding changed scalar bits'))
                previous=outputs[0]
                controls.append(dict(case=label,products=outputs[0]))
(root/'controls.json').write_text(json.dumps(dict(backend=backend,expected_adapter=adapter,controls=controls,failures=failures,
    qualification='functional determinism only; actual adapter identity and metric accuracy require independent gates'),indent=2)+'\n')
print(json.dumps(dict(backend=backend,expected_adapter=adapter,control_cases=len(controls),failures=failures)),flush=True)
sys.exit(1 if failures else 0)
