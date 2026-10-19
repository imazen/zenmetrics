#!/usr/bin/env python3
"""Time unchanged primary MATLAB functions through a resident-RGB Octave cell.

The explicit source manifest provides paths, function name and ingress. Only
the call-site adapter is generated; author source is never modified here.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess


def quote(value):
    return "'"+str(value).replace("'", "''")+"'"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sources', type=Path)
    p.add_argument('metric')
    p.add_argument('width', type=int)
    p.add_argument('height', type=int)
    p.add_argument('reference', type=Path)
    p.add_argument('distorted', type=Path)
    p.add_argument('threads', type=int)
    p.add_argument('samples', type=int)
    a = p.parse_args()
    spec = json.loads(a.sources.read_text())[a.metric]
    setup = ['more off;', 'pkg load image;']
    for path in spec['paths']:
        setup.append('addpath('+quote(path)+",'-begin');")
    function = spec['function']
    setup.append(f"assert(strcmp(which({quote(function)}),{quote(spec['source'])}));")
    for role, name in [('reference', 'r'), ('distorted', 'd')]:
        path = getattr(a, role).resolve(strict=True)
        setup.append(f"f=fopen({quote(path)},'rb');assert(f>=0);[{name},n]=fread(f,Inf,'*uint8');assert(n=={a.width*a.height*3});assert(fclose(f)==0);")
    ingress = 'reference=permute(reshape(r,[3,%d,%d]),[3,2,1]);distorted=permute(reshape(d,[3,%d,%d]),[3,2,1]);' % (a.width,a.height,a.width,a.height)
    conversion = spec.get('ingress','rgb8')
    if conversion in ('luma601', 'luma601-round', 'msssim-luma'):
        weights = ['0.2989','0.5870','0.1140'] if conversion=='msssim-luma' else ['0.299','0.587','0.114']
        for name in ('reference','distorted'):
            ingress += name+'='+'+'.join(f'{c}*double({name}(:,:,{i+1}))' for i,c in enumerate(weights))+';'
            if conversion=='luma601-round':
                ingress += f'{name}=round({name});'
    call = f'score={function}(reference,distorted);'
    if a.metric == 'msssim':
        level=min(5,int(math.floor(math.log2(min(a.width,a.height)/11)))+1)
        setup.append(f"level={level};weights=[0.0448 0.2856 0.3001 0.2363 0.1333];")
        call="score=real(msssim(reference,distorted,[0.01 0.03],fspecial('gaussian',11,1.5),level,weights(1:level),'product'));"
    if a.metric.startswith('psnrhvs'):
        call=f'[score,unused]={function}(reference,distorted);'
    script='\n'.join(setup)+f"""
for call_index=0:{a.samples}
 clock=tic();
 {ingress}
 {call}
 clear reference distorted;
 elapsed=toc(clock)*1e9;
 assert(isreal(score)&&isfinite(score));
 label='SAMPLE';if call_index==0;label='COLD';end
 fprintf('%s %.0f %.17g\\n',label,elapsed,score);fflush(stdout);
end
"""
    path=Path(os.environ['TMPDIR'])/'canonical-entry.m'
    with path.open('x') as f:
        f.write(script)
    env=dict(os.environ)
    for key in ('DISPLAY','WAYLAND_DISPLAY','XAUTHORITY'):
        env.pop(key,None)
    result=subprocess.run(['octave-cli','--no-gui','--no-window-system','--no-init-file','--no-history','--quiet',str(path)],env=env)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
