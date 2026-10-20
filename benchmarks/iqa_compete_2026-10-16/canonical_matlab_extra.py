#!/usr/bin/env python3
"""Timed call-site glue for original CW-SSIM, PSNR-HVS, FSIM and MAD sources."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from canonical_octave_cell import quote


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sources',type=Path);p.add_argument('metric')
    p.add_argument('width',type=int);p.add_argument('height',type=int)
    p.add_argument('reference',type=Path);p.add_argument('distorted',type=Path)
    p.add_argument('threads',type=int);p.add_argument('samples',type=int)
    a=p.parse_args();spec=json.loads(a.sources.read_text())[a.metric]
    setup=['more off;pkg load image;',"warning('off','Octave:possible-matlab-short-circuit-operator');",f"fftw('threads',{a.threads});"]
    setup += [f'addpath({quote(path)},"-begin");' for path in spec['paths']]
    setup += [f'assert(strcmp(which({quote(name)}),{quote(path)}));' for name,path in spec['functions'].items()]
    for name,path in [('r',a.reference),('d',a.distorted)]:
        setup.append(f'f=fopen({quote(path.resolve(strict=True))},"rb");assert(f>=0);[{name},n]=fread(f,Inf,"*uint8");assert(n=={a.width*a.height*3});assert(fclose(f)==0);')
    ingress=''
    for name,raw in [('reference','r'),('distorted','d')]:
        ingress+=f'{name}=double(permute(reshape({raw},[3,{a.width},{a.height}]),[3,2,1]));'
        if a.metric in ['psnrhvs-y','cw-ssim','mad']:
            c='0.2989' if a.metric=='mad' else '0.299'
            ingress+=f'{name}={c}*{name}(:,:,1)+0.587*{name}(:,:,2)+0.114*{name}(:,:,3);'
            if a.metric=='psnrhvs-y':ingress+=f'{name}=round({name});'
            if a.metric=='cw-ssim':ingress+=f'{name}=roundb({name});'
    if a.metric in ['psnrhvs','psnrhvs-y']:
        call='values=zeros(size(reference,3),1);for c=1:size(reference,3);[masked,unmasked]=psnrhvsm(reference(:,:,c),distorted(:,:,c));values(c)=unmasked;end;score=mean(values);clear values masked unmasked;'
    elif a.metric=='cw-ssim':call='score=cwssim_index(reference,distorted,4,8,0,0);'
    elif a.metric=='fsim-y':call='score=FeatureSIM(reference,distorted);'
    elif a.metric=='mad':call='hi=hi_index(reference,distorted);lo=lo_index(reference,distorted);sig=1/(1+exp(-2.55/3.35)*hi^(1/(log(10)*3.35)));score=hi^sig*lo^(1-sig);clear hi lo sig;'
    else:raise ValueError('unknown original MATLAB reference')
    script='\n'.join(setup)+f"""
for index=0:{a.samples}
  clock=tic();{ingress}{call}
  clear reference distorted;
  elapsed=toc(clock)*1e9;assert(isreal(score)&&isfinite(score));
  label='SAMPLE';if index==0;label='COLD';end
  fprintf('%s %.0f %.17g\\n',label,elapsed,score);fflush(stdout);
end
"""
    path=Path(os.environ['TMPDIR'])/'original-entry.m'
    with path.open('x') as f:f.write(script)
    subprocess.run(['octave-cli','--no-gui','--no-init-file','--no-history','--quiet',str(path)],check=True)


if __name__=='__main__':main()
