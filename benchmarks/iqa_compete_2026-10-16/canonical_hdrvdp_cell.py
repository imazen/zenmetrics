#!/usr/bin/env python3
"""Resident RGB8 comparison to explicit official HDR-VDP Octave trees.

Only the adapter supplies the CLI's declared SDR units. It does not identify
the unpublished invocation behind an AIC dataset's precomputed columns.
"""
import argparse
import os
from pathlib import Path
import subprocess


def quote(value):
    return "'"+str(value).replace("'", "''")+"'"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('tree',type=Path)
    p.add_argument('metric',choices=['hdrvdp-aic2026','hdrvdp3-aic2026'])
    p.add_argument('width',type=int);p.add_argument('height',type=int)
    p.add_argument('reference',type=Path);p.add_argument('distorted',type=Path)
    p.add_argument('threads',type=int);p.add_argument('samples',type=int)
    p.add_argument('--statistics',type=Path,help='Directory of an explicitly bound upstream geomean.m')
    a=p.parse_args();v3=a.metric=='hdrvdp3-aic2026'
    setup=['more off;pkg load image;',f'addpath({quote(a.tree)});']
    pyr=a.tree/('matlabPyrTools' if v3 else 'matlabPyrTools_1.4_fixed')
    setup.append(f'addpath({quote(pyr)});')
    if v3:
        setup.extend([f'addpath({quote(a.statistics)});' if a.statistics else 'pkg load statistics;',f'addpath({quote(a.tree/"utils")});',f'addpath({quote(a.tree/"data")});'])
    for name,path in [('r',a.reference),('d',a.distorted)]:
        setup.append(f'f=fopen({quote(path.resolve(strict=True))},"rb");assert(f>=0);[{name},n]=fread(f,Inf,"*uint8");assert(n=={a.width*a.height*3});assert(fclose(f)==0);')
    conversion=[]
    for name,raw in [('reference','r'),('distorted','d')]:
        conversion.append(f'{name}=double(permute(reshape({raw},[3,{a.width},{a.height}]),[3,2,1]))/255;')
        if v3:conversion.append(f'{name}=round({name}*1023)/1023;')
        conversion.append(f'lo={name}<=0.04045;{name}(lo)={name}(lo)/12.92;{name}(~lo)=(({name}(~lo)+0.055)/1.055).^2.4;{name}=200*{name};')
        if v3:
            conversion.append(f'x={name};{name}(:,:,1)=0.4124564*x(:,:,1)+0.3575761*x(:,:,2)+0.1804375*x(:,:,3);{name}(:,:,2)=0.2126729*x(:,:,1)+0.7151522*x(:,:,2)+0.0721750*x(:,:,3);{name}(:,:,3)=0.0193339*x(:,:,1)+0.1191920*x(:,:,2)+0.9503041*x(:,:,3);clear x;')
    call=("result=hdrvdp3('quality',distorted,reference,'XYZ',37.842505866,{'use_gpu',false,'quiet',true,'age',24});"
          if v3 else "result=hdrvdp(distorted,reference,'rgb-bt.709',37.84,{});")
    script='\n'.join(setup)+f"""
for index=0:{a.samples}
  clock=tic();
  {''.join(conversion)}
  {call}
  score=result.Q;
  clear reference distorted result lo;
  elapsed=toc(clock)*1e9;
  assert(isreal(score)&&isfinite(score));
  label='SAMPLE';if index==0;label='COLD';end
  fprintf('%s %.0f %.17g\\n',label,elapsed,score);fflush(stdout);
end
"""
    path=Path(os.environ['TMPDIR'])/'hdrvdp-entry.m'
    with path.open('x') as f:f.write(script)
    subprocess.run(['octave-cli','--no-gui','--no-init-file','--no-history','--quiet',str(path)],check=True)


if __name__=='__main__':main()
