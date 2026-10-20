#!/usr/bin/env python3
"""Prepare all 24 official sRGB golden controls with identical raw inputs.

Writes source-bound system-image and unchanged resize-overlay entries. Never
runs Octave, compiles, installs packages, or changes existing plans/helpers.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

ARCHIVE_SHA = 'b4b6865e7f65ee5b9c254a69128bf10e93b38bc27f5bc5b973fef300217fc725'
GENERATOR_SHA = '907fab1cd50619ce930df4d63c9fa5fcb765f7b36899da78c4476de6dacdb071'
GOLDENS_SHA = '323aa664d1227344d8ab6a0b1773dd51bb9792516730f4618324619c4dd74599'
OVERLAY = {'imresize.m': 'c358d5381f325c5ab58c7eb0c3ccc8acc2141bc70634c66cc516541b823a15c8',
           'private/interp_method.m': 'bcd3156239d6d4dccb626108a642afead1b4c60b400fed8b11d73ef5edcc6beb'}
NAMES = ('identical', 'noise', 'blur', 'dark', 'bright', 'chroma')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def reference_codes(width, height):
    return bytes((31*x + 17*y + 59*c + x*y//13 + (x+2*c)**2//29) % 256
                 for y in range(height) for x in range(width) for c in range(3))


def distort(name, reference, width, height):
    if name == 'identical':
        return reference
    if name == 'dark':
        return bytes(v*55//100 for v in reference)
    if name == 'bright':
        return bytes(min(255, v*130//100+8) for v in reference)
    if name == 'chroma':
        return bytes(min(255, v+20) if i % 3 == 0 else v for i, v in enumerate(reference))
    if name == 'noise':
        output = bytearray()
        state = 12345
        for value in reference:
            state = state * 48271 % 2147483647
            output.append(min(255, max(0, value + state % 21 - 10)))
        return bytes(output)
    if name == 'blur':
        output = bytearray()
        for y in range(height):
            for x in range(width):
                for c in range(3):
                    total = sum(reference[(min(height-1, max(0, y+dy))*width
                                          + min(width-1, max(0, x+dx)))*3+c]
                                for dy in (-1, 0, 1) for dx in (-1, 0, 1))
                    output.append(total//9)
        return bytes(output)
    raise ValueError('unknown official distortion')


def golden_rows(path):
    if sha(path) != GOLDENS_SHA:
        raise ValueError('official sRGB golden TSV changed')
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    keys = {(r['case'], int(r['w']), int(r['h']), int(r['ppd'])) for r in rows}
    expected = {(name, width, height, ppd) for name in NAMES
                for width, height in ((128, 128), (160, 96)) for ppd in (30, 60)}
    if len(rows) != 24 or keys != expected or any(r['enc'] != 'srgb-display' for r in rows):
        raise ValueError('exact complete 24-case official panel required')
    return rows


def entry(root, official, overlay, rows, variant):
    lines = ['more off;pkg load image;', 'assert(strcmp(version(),"6.4.0"));',
        f'addpath({quote(official)},"-begin");',
        f'addpath({quote(official/"matlabPyrTools_1.4_fixed")},"-begin");']
    if variant == 'resize2201':
        lines += [f'addpath({quote(overlay)},"-begin");', 'clear imresize;']
    lines += [f'assert(strcmp(which("hdrvdp"),{quote(official/"hdrvdp.m")}));',
        f'assert(strcmp(which("imresize"),{quote(overlay/"imresize.m") if variant=="resize2201" else quote("/usr/share/octave/packages/image-2.12.0/imresize.m")}));',
        'fprintf("RUNTIME octave=%s variant=' + variant + '\\n",version());',
        'for name={"hdrvdp","imresize","fspecial","imfilter","padarray","interp1","ifft2"};fprintf("BIND %s %s\\n",name{1},which(name{1}));end;',
        'failed=0;count=0;']
    for row in rows:
        width, height, ppd = [int(row[k]) for k in ('w', 'h', 'ppd')]
        for variable, filename in [('reference', f'ref-{width}x{height}.rgb'),
                                   ('distorted', f'{row["case"]}-{width}x{height}.rgb')]:
            lines += [f'f=fopen({quote(root/"inputs"/filename)},"rb");assert(f>=0);[{variable},n]=fread(f,Inf,"*uint8");assert(n=={width*height*3});assert(fclose(f)==0);',
                      f'{variable}=double(permute(reshape({variable},[3,{width},{height}]),[3,2,1]))/255;']
        # Preserve existing official golden values and existing 5e-3 bound.
        lines += [f"result=hdrvdp(distorted,reference,'sRGB-display',{ppd},{{}});",
            f'expected={row["q"]};delta=abs(result.Q-expected);passed=isreal(result.Q)&&isfinite(result.Q)&&delta<=5e-3;',
            'failed=failed+!passed;count=count+1;',
            f"fprintf('SRGB_GOLDEN {row['case']} {width} {height} {ppd} %.17g %.17g %.17g %d %s\\n',result.Q,expected,delta,passed,num2hex(result.Q));fflush(stdout);",
            'clear reference distorted result;']
    lines += ['assert(count==24);fprintf("SRGB_COMPLETE cases=%d failed=%d\\n",count,failed);fflush(stdout);',
              'assert(failed==0,"official sRGB golden mismatch; retain all observations");']
    return '\n'.join(lines) + '\n'


def prepare(validation, official_source, resize_source, octave, output, archive):
    validation, official_source, resize_source, octave = [p.resolve(strict=True)
        for p in (validation, official_source, resize_source, octave)]
    archive=archive.resolve(strict=True)
    if sha(archive)!=ARCHIVE_SHA:
        raise ValueError('official retained image archive changed')
    generator, goldens = validation/'gen_srgb_goldens.m', validation/'goldens_srgb/goldens_srgb.tsv'
    if sha(generator) != GENERATOR_SHA:
        raise ValueError('official integer fixture generator changed')
    rows = golden_rows(goldens)
    for name, expected in OVERLAY.items():
        if sha(resize_source/name) != expected:
            raise ValueError('unchanged upstream resize source required: ' + name)
    output.mkdir()
    root = output.resolve()
    official, overlay, inputs = root/'official-v2', root/'resize2201', root/'inputs'
    shutil.copytree(official_source, official)
    for name in OVERLAY:
        dst=overlay/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(resize_source/name, dst)
    inputs.mkdir()
    pairs = []
    for width, height in ((128, 128), (160, 96)):
        reference = reference_codes(width, height)
        (inputs/f'ref-{width}x{height}.rgb').write_bytes(reference)
        for name in NAMES:
            distorted = distort(name, reference, width, height)
            path=inputs/f'{name}-{width}x{height}.rgb';path.write_bytes(distorted)
            pairs.append(dict(case=name,width=width,height=height,reference=str(inputs/f'ref-{width}x{height}.rgb'),
                distorted=str(path),reference_sha256=hashlib.sha256(reference).hexdigest(),distorted_sha256=sha(path)))
    shutil.copy2(generator,root/'original-gen_srgb_goldens.m')
    shutil.copy2(goldens,root/'original-goldens_srgb.tsv')
    shutil.copy2(Path(__file__),root/Path(__file__).name)
    controls=[]
    for variant in ('system212','resize2201'):
        path=root/(variant+'.m')
        with path.open('x') as stream:stream.write(entry(root,official,overlay,rows,variant))
        controls.append(dict(variant=variant,command=[str(octave),'--no-gui','--no-init-file','--no-history','--quiet',str(path)],
            script_sha256=sha(path),environment={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS')},
            expected_cases=24,absolute_bound=5e-3,executed=False))
    bindings={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
    bindings[str(octave)]=sha(octave)
    bindings[str(archive)]=ARCHIVE_SHA
    result=dict(schema=1,controls=controls,bindings=bindings,pairs=pairs,
        original_generator_sha256=GENERATOR_SHA,original_goldens_sha256=GOLDENS_SHA,
        expected_cases_per_variant=24,existing_absolute_bound=5e-3,
        executed=False,qualified_timing_cells=0,qualification='Diagnostic until original controls pass; unchanged v2 metric, same raw bytes, distinct runtime dependency',
        required_before_execution=['Root code review and heavy admission with both locks/run-heavy12G/disk TMPDIR',
            'Bind and verify system core/image/shared-library closure before and after each process',
            'Confirm generated raw bytes against original integer generator in an independent admitted control'],
        retention='Run both variants independently; a failing base process must not suppress overlay execution. Preserve every output and original baseline.')
    write(root/'control-plan.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('validation','official','resize','octave','output'):parser.add_argument(name,type=Path)
    parser.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args();result=prepare(args.validation,args.official,args.resize,args.octave,args.output,args.archive)
    print(json.dumps(dict(variants=len(result['controls']),cases_per_variant=24,pairs=len(result['pairs']),executed=False)))


if __name__=='__main__':main()
