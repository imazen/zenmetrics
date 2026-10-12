#!/usr/bin/env python3
"""Real IQA parity controls across instrumentation, reuse, threads and stride."""
import json
import os
from pathlib import Path
import subprocess
import sys

binaries, root = map(lambda s: Path(s).resolve(), sys.argv[1:])
root.mkdir(parents=True, exist_ok=False)
w, h = 65, 67
packed = bytes(((x*17+y*13+c*41)%241)+7 for y in range(h) for x in range(w) for c in range(3))
distorted = bytes(max(0,b-7) for b in packed)
for name, data in [('reference',packed),('distorted',distorted)]:
    (root/f'{name}-packed').write_bytes(data)
    (root/f'{name}-padded').write_bytes(b''.join(data[y*w*3:(y+1)*w*3]+bytes([99]*7) for y in range(h)))
controls=[]
for metric in ['ssim2','butteraugli','dssim','zensim']:
    for reuse in ['cold','warm_reference']:
        for threads in [1,2]:
            previous=None
            for layout,padding in [('packed',0),('padded',7)]:
                label=f'{metric}-{reuse}-{threads}-{layout}'
                case={'schema':1,'metric':metric,'width':w,'height':h,'stride_bytes':w*3+padding,
                      'reference':f'reference-{layout}','distorted':f'distorted-{layout}',
                      'pixel_format':'rgb8_srgb','reuse':reuse,'threads':threads}
                casepath=root/f'{label}.json';casepath.write_text(json.dumps(case))
                observed=[]
                for variant in ['latency','memory']:
                    env=dict(os.environ,RAYON_NUM_THREADS=str(threads),ZENBENCH_OPERATIONS='1')
                    env.pop('ZENBENCH_RESULT_FILE',None)
                    result=subprocess.run([str(binaries/variant),'fleet',str(casepath)],env=env,
                                          capture_output=True,timeout=120)
                    (root/f'{label}-{variant}.stdout').write_bytes(result.stdout)
                    (root/f'{label}-{variant}.stderr').write_bytes(result.stderr)
                    assert result.returncode == 0, (label,variant,result.stderr.decode())
                    value=json.loads(result.stdout)
                    assert value['timing_usable'] == (variant=='latency')
                    rows=value['driver']['memory']
                    if variant=='latency': assert rows == []
                    else:
                        assert rows
                        scopes={r['scope'] for r in rows}
                        assert 'score' in scopes
                        if reuse=='warm_reference': assert {'reference_preparation','score_warmup'} <= scopes
                        for scope in scopes:
                            data={r['quantity']:r['bytes'] for r in rows if r['scope']==scope}
                            assert data['requested_live_heap_peak'] >= max(data['requested_live_heap_entry'],data['requested_live_heap_exit'])
                    observed.append(value['driver']['correctness'])
                assert observed[0]==observed[1], (label,'instrumentation changed output',observed)
                if previous is not None: assert previous==observed[0], (label,'stride changed pixels')
                previous=observed[0]
                controls.append({'case':label,'correctness':observed[0]})
(root/'controls.json').write_text(json.dumps(controls,indent=2)+'\n')
print(json.dumps({'iqa_control_cases':len(controls),'instrumentation_parity':'exact','stride_parity':'exact'}))
