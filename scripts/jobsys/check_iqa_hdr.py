#!/usr/bin/env python3
"""Native absolute-nits workload controls; not performance measurements."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('binaries', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--size', type=int, nargs=2, default=[128, 96], metavar=('WIDTH', 'HEIGHT'))
args = parser.parse_args()
binaries, root = args.binaries.resolve(), args.output.resolve()
w, h = args.size
if w < 1 or h < 1:
    parser.error('dimensions must be positive')
root.mkdir(parents=True, exist_ok=False)
for name, factor in [('reference', 1.0), ('distorted', 0.9)]:
    rows = [b''.join(struct.pack('<f', factor*(50+(x*17+y*13+c*41)%701))
                    for x in range(w) for c in range(3)) for y in range(h)]
    for layout, padding in [('packed', b''), ('padded', bytes([255]*7))]:
        (root/f'{name}-{layout}').write_bytes(b''.join(row+padding for row in rows))
env = dict(os.environ, RAYON_NUM_THREADS='1', ZENBENCH_OPERATIONS='2')
env.pop('ZENBENCH_RESULT_FILE', None)
controls=[]
for metric, model in [('hdrvdp', 'hdrvdp2-defaults-bt709'),
                      ('hdrvdp3', 'hdrvdp3-quality-bt709-age24-nosurround-reference')]:
    previous=None
    for layout, padding in [('packed', 0), ('padded', 7)]:
        case=dict(schema=3, metric=metric, width=w, height=h, stride_bytes=w*12+padding,
                  reference=f'reference-{layout}', distorted=f'distorted-{layout}',
                  pixel_format='rgb_f32le_bt709_linear_nits', reuse='cold', threads=1,
                  products='all', warmup_operations=0,
                  viewing=dict(model=model, pixels_per_degree=45.0))
        label=f'{metric}-{layout}'
        path=root/f'{label}.json'; path.write_text(json.dumps(case))
        outputs=[]
        for variant in ['latency','memory']:
            result=subprocess.run([str(binaries/variant),'fleet',str(path)],env=env,
                                  capture_output=True,timeout=120)
            (root/f'{label}-{variant}.stdout').write_bytes(result.stdout)
            (root/f'{label}-{variant}.stderr').write_bytes(result.stderr)
            assert result.returncode==0, (label,variant,result.stderr.decode())
            value=json.loads(result.stdout)
            assert value['timing_usable']==(variant=='latency')
            products=value['driver']['correctness']['products']
            assert len(products)==2 and products[0]==products[1] and products[0]
            outputs.append(products)
        assert outputs[0]==outputs[1], (label,'instrumentation changed products')
        if previous is not None: assert previous==outputs[0], (label,'stride changed products')
        previous=outputs[0];controls.append(dict(case=label,products=outputs[0]))
    for change in [dict(pixel_format='rgb8_srgb'),dict(reuse='warm_reference'),
                   dict(viewing=dict(model=model,pixels_per_degree=0)),dict(unknown=True)]:
        invalid=dict(case,**change);path=root/f'{metric}-invalid.json';path.write_text(json.dumps(invalid))
        result=subprocess.run([str(binaries/'latency'),'fleet',str(path)],env=env,
                              capture_output=True,timeout=120)
        assert result.returncode!=0, change
(root/'controls.json').write_text(json.dumps(controls,indent=2)+'\n')
print(json.dumps(dict(native_hdr_metrics=2,width=w,height=h,instrumentation_and_stride_parity='exact scalar products',
                     qualification='functional routing only; no performance/admission certification')))
