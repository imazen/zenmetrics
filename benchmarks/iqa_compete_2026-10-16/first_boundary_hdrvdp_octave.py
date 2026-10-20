#!/usr/bin/env python3
"""Prepare an official sRGB-display Octave entry; never execute it.

The input config binds a frozen official tree, runtime, packages and raw RGB
pair. No AIC2026 EOTF/XYZ conversion is applied. Upstream bodies stay unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def verify(config):
    for name, expected in config['bindings'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('frozen reference binding changed: ' + name)
    if config['metric'] not in ('hdrvdp', 'hdrvdp3'):
        raise ValueError('base sRGB-display routes only')
    if min(config['width'], config['height'], config['threads'], config['samples']) <= 0:
        raise ValueError('positive geometry, thread budget and samples required')
    if config['phase'] not in ('first', 'warm') or (config['phase'] == 'first' and config['samples'] != 1):
        raise ValueError('first requires exactly one call; warm requires positive samples')
    for role in ('reference', 'distorted'):
        path = Path(config[role]).resolve(strict=True)
        if str(path) not in config['bindings'] or path.stat().st_size != config['width'] * config['height'] * 3:
            raise ValueError('raw RGB identity/geometry not bound')
    tree = Path(config['tree']).resolve(strict=True)
    for path in tree.rglob('*'):
        if path.is_file() and str(path.resolve()) not in config['bindings']:
            raise ValueError('unbound original reference file: ' + str(path))
    for name in ('octave', 'geomean'):
        path = Path(config[name]).resolve(strict=True)
        if str(path) not in config['bindings']:
            raise ValueError('reference runtime/helper is unbound: ' + name)


def script(config, direct_control=False):
    verify(config)
    v3 = config['metric'] == 'hdrvdp3'
    tree = Path(config['tree']).resolve(strict=True)
    entry = tree / ('hdrvdp3.m' if v3 else 'hdrvdp.m')
    pyramid = tree / ('matlabPyrTools' if v3 else 'matlabPyrTools_1.4_fixed')
    setup = [
        'more off; pkg load image;',
        f'assert(strcmp(version(),{quote(config["octave_version"])}));',
        f'packages=pkg("list"); image_ok=false; for k=1:numel(packages); if strcmp(packages{{k}}.name,"image"); image_ok=strcmp(packages{{k}}.version,{quote(config["image_version"])}); end; end; assert(image_ok);',
        "fprintf('RUNTIME octave=%s image=%s\\n',version()," + quote(config['image_version']) + ');',
        f'addpath({quote(tree)},"-begin"); addpath({quote(pyramid)},"-begin");',
    ]
    if v3:
        setup += [f'addpath({quote(tree / "utils")},"-begin");',
                  f'addpath({quote(tree / "data")},"-begin");',
                  f'addpath({quote(Path(config["geomean"]).parent)},"-begin");',
                  f'assert(strcmp(which("geomean"),{quote(config["geomean"])}));']
    setup += [f'assert(strcmp(which({quote(entry.stem)}),{quote(entry)}));',
        'for name={"imresize","imfilter","ifft2","interp1","corrDn","upConv"}; fprintf("RUNTIME_BIND %s %s\\n",name{1},which(name{1})); end']
    for name, role in [('r', 'reference'), ('d', 'distorted')]:
        shape = f'[3,{config["width"]*config["height"]}]' if direct_control else 'Inf'
        setup += [f'f=fopen({quote(config[role])},"rb");assert(f>=0);[{name},n]=fread(f,{shape},"*uint8");assert(n=={config["width"]*config["height"]*3});assert(fclose(f)==0);']
    # Conversion, option parsing, original parameter-file I/O, evaluation and
    # output destruction all remain in the timed body. Imports and input reads
    # are excluded. There is no manual 200-nit EOTF or XYZ transform here.
    if direct_control:
        # Independent interleaved-row reshape: transpose Nx3 pixels, then
        # reshape channel columns. Controls run in their own fresh processes.
        convert = '\n'.join(f'{name}=permute(reshape(double({raw}\'),[{config["width"]},{config["height"]},3]),[2,1,3])/255;'
                            for name, raw in [('reference', 'r'), ('distorted', 'd')])
    else:
        convert = '\n'.join(f'{name}=double(permute(reshape({raw},[3,{config["width"]},{config["height"]}]),[3,2,1]))/255;'
                            for name, raw in [('reference', 'r'), ('distorted', 'd')])
    call = ("result=hdrvdp3('quality',distorted,reference,'sRGB-display',30,{'use_gpu',false,'quiet',true,'age',24,'surround','none'});score=result.Q_JOD;"
            if v3 else "result=hdrvdp(distorted,reference,'sRGB-display',30,{});score=result.Q;")
    body = convert + '\n' + call + '\nclear reference distorted result;\n'
    if config['phase'] == 'warm':
        setup += [body, 'clear score;']
    setup += [f'for iteration=1:{config["samples"]}', 'clock=tic();', body,
        'elapsed=toc(clock)*1e9; assert(isreal(score)&&isfinite(score));',
        f"fprintf('{('COLD' if config['phase'] == 'first' else 'SAMPLE')} %.0f %.17g\\n',elapsed,score); fflush(stdout);", 'end']
    return '\n'.join(setup) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--direct-control', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    text = script(config, args.direct_control)
    with args.output.open('x') as stream:
        stream.write(text)
    print(json.dumps(dict(script=str(args.output.resolve()),
        sha256=hashlib.sha256(text.encode()).hexdigest(), executed=False,
        qualification='Evaluator-first or warm with original parameter I/O included; not OS-cache-cold or process startup')))


if __name__ == '__main__':
    main()
