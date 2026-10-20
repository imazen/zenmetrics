#!/usr/bin/env python3
"""Emit additive first-call benchmark adapters from exact reviewed harness bytes.

No original metric body is edited. Emission is preparation, not qualification:
run independent fresh-process original-call controls before using any timings.
"""
import argparse
import hashlib
import json
from pathlib import Path


SOURCES = {
    'cvvdp': ('canonical_cvvdp_cell.py', 'ae1e30afad06bfad962b6d7a57712dfb04a4d9a15922e6622a74cb20bfb13cc7'),
    'iwssim': ('canonical_iwssim_cell.py', 'b254069ca56fc288eee586160f402a9860ee310ba5c4614022c5e74ffd72b307'),
    'iwssim-piq': ('canonical_iwssim_cell.py', 'b254069ca56fc288eee586160f402a9860ee310ba5c4614022c5e74ffd72b307'),
    'nlpd': ('canonical_classical_cell.py', 'c0b449abcb678a89604cad138138a4b70816c6d1ca5c6e924a8b509ef38ce357'),
    'nlpd-iqa': ('canonical_classical_cell.py', 'c0b449abcb678a89604cad138138a4b70816c6d1ca5c6e924a8b509ef38ce357'),
    'niqe': ('canonical_niqe_resident_cell.py', 'ea91b5cfe85bd4eb7bac4d9ac6193a62284a9c6c5a6a74e926f86742b4628977'),
    'dssim': ('dssim_reference.c', '55fbb678aa2c2885e753c010365ced392c0a7dc3b2997e28de98c19356a8df76'),
    'libvmaf': ('libvmaf_rgb_compare.cpp', 'd08698529ee31bc7861b568329ad884e57e08c9775189e535a7eb40a450aa326'),
    'libjxl': ('libjxl_compute.cpp', '1d5733e90c7488205bdb59669c1da48b99af6e57e96a0e70be34600ba66f0f74'),
    'ssim2': ('ssimulacra2_primary_compute.cpp', '4142b2f1e7ef265d03f8ffdbba23ce656cd386ad357f1ae99a24e3a8e4427ddf'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('reviewed source anchor must occur exactly once: ' + before)
    return text.replace(before, after, 1)


def python_source(kind, text):
    if kind == 'cvvdp':
        constructor = "    model=pycvvdp.cvvdp(display_name='standard_fhd',device=torch.device('cpu'),quiet=True)\n"
        text = replace_once(text, constructor, '    model=None\n')
        # Native cold_call keeps the Cvvdp closure alive through elapsed capture.
        text = replace_once(text, '    def score():\n',
                            '    def score():\n        nonlocal model\n' + '    ' + constructor)
    elif kind in ('iwssim', 'iwssim-piq'):
        constructor = '    model=module.IW_SSIM(use_cuda=False,use_double=False)\n'
        text = replace_once(text, constructor, '')
        text = replace_once(text,
            '    def score():return float(model.test(gray(r),gray(d)).item())',
            '    def score():\n    ' + constructor +
            '        return float(model.test(gray(r),gray(d)).item())')
        if kind == 'iwssim-piq':
            text = replace_once(text, "    p.add_argument('source',type=Path);",
                                "    p.add_argument('piq_extra',type=Path);p.add_argument('source',type=Path);")
            text = replace_once(text, "    gray=load(a.source/'utils.py','primary_gray').rgb2gray",
                "    import sys\n    sys.path.insert(0,str(a.piq_extra.resolve(strict=True)))\n"
                "    from piq.functional import rgb2yiq\n"
                "    def gray(rgb):\n"
                "        tensor=torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).float()\n"
                "        return rgb2yiq(tensor)[0,0].numpy()")
    elif kind in ('nlpd', 'nlpd-iqa'):
        constructor = "        model=module.LaplacianPyramid(6,dims=3) if a.metric=='nlpd' else module.NLPD(channels=1)\n"
        text = replace_once(text, constructor, '')
        text = replace_once(text, '        def score():\n            x,y=tensor(r),tensor(d)',
                            '        def score():\n    ' + constructor + '            x,y=tensor(r),tensor(d)')
        text = replace_once(text, '    a=p.parse_args()\n',
                            f"    a=p.parse_args()\n    if a.metric!={kind!r}:raise ValueError('wrong first-boundary route')\n")
    elif kind == 'niqe':
        start = text.index('    original = score()\n')
        end = text.index('    for index in range(a.samples+1):\n', start)
        # The old same-process qualification primes the scorer. The control
        # driver must establish the exact-bit gate in separate fresh processes.
        text = text[:start] + (
            "    module.np = resident_numpy\n"
            "    print('BOUNDARY NIQE first evaluation; resident parameters; separate original-call control required',flush=True)\n"
        ) + text[end:]
    return text


def compiled_source(kind, text):
    if kind == 'dssim':
        setup = '    Dssim *metric = dssim_new();\n    if (!metric) return 2;\n'
        text = replace_once(text, setup, '')
        text = replace_once(text, '        long long start = now_ns();\n',
                            '        long long start = now_ns();\n' + setup)
        text = replace_once(text, '        long long elapsed = now_ns()-start;',
                            '        dssim_free(metric);\n        long long elapsed = now_ns()-start;')
        text = replace_once(text, '    dssim_free(metric); free(r); free(d); return 0;',
                            '    free(r); free(d); return 0;')
    elif kind == 'libvmaf':
        start = text.index('    VmafModel* model=nullptr;\n')
        end = text.index('    const std::string feature=', start)
        setup = text[start:end]
        text = text[:start] + text[end:]
        text = replace_once(text, '            const bool studio=convention!="709";',
                            setup + '            const bool studio=convention!="709";')
        text = replace_once(text, '            checked(vmaf_close(ctx),"close");',
                            '            checked(vmaf_close(ctx),"close");\n            if(model) vmaf_model_destroy(model);')
        text = replace_once(text, '    if(model) vmaf_model_destroy(model);\n    return 0;', '    return 0;')
    elif kind == 'libjxl':
        # Retain the first-created runner beyond elapsed capture, like native
        # process-global runtime workers. Score arithmetic stays unchanged.
        text = replace_once(text, '#include <chrono>', '#include <chrono>\n#include <memory>')
        text = replace_once(text, '    jpegxl::tools::ThreadPoolInternal pool(threads==1?0:threads);',
            '    if(scope!="rgb-input" || mode!="score" || metric=="ssim2")throw std::runtime_error("first-boundary requires Butteraugli rgb-input score");\n'
            '    std::unique_ptr<jpegxl::tools::ThreadPoolInternal> pool;')
        text = text.replace('pool.get()', 'pool->get()')
        text = replace_once(text, '            {   // Timed RAII scope: every per-call object releases before the clock stops.',
            '            {   // Construction-inclusive first-call scope.\n'
            '                if(!pool)pool=std::make_unique<jpegxl::tools::ThreadPoolInternal>(threads==1?0:threads);')
        # Native-buffer is unreachable; retain its body without arithmetic edits.
        text = replace_once(text,
            '            if(i)std::cout<<"SAMPLE "<<elapsed<<\' \'<<score<<\'\\n\'<<std::flush;\n        }\n    }\n',
            '            std::cout<<(i?"SAMPLE ":"COLD ")<<elapsed<<\' \'<<score<<\'\\n\'<<std::flush;\n        }\n    }\n')
    elif kind == 'ssim2':
        text = replace_once(text, '#include <chrono>', '#include <chrono>\n#include <memory>')
        setup = ('    auto runner = JxlThreadParallelRunnerMake(nullptr, threads == 1 ? 0 : threads);\n'
                 '    if (!runner) throw std::runtime_error("thread runner allocation");\n'
                 '    jxl::ThreadPool pool(JxlThreadParallelRunner, runner.get());\n')
        text = replace_once(text, setup,
            '    decltype(JxlThreadParallelRunnerMake(nullptr, 0)) runner;\n'
            '    std::unique_ptr<jxl::ThreadPool> pool;\n')
        text = replace_once(text, '        jxl::ImageMetadata ref_metadata, dist_metadata;',
            '        if (!pool) {\n'
            '          runner=JxlThreadParallelRunnerMake(nullptr, threads == 1 ? 0 : threads);\n'
            '          if (!runner) throw std::runtime_error("thread runner allocation");\n'
            '          pool=std::make_unique<jxl::ThreadPool>(JxlThreadParallelRunner, runner.get());\n'
            '        }\n'
            '        jxl::ImageMetadata ref_metadata, dist_metadata;')
        if text.count('&pool,') != 2:
            raise ValueError('two original ingress runner arguments required')
        text = text.replace('&pool,', 'pool.get(),')
    return text


def prepare(kind, source, output):
    name, expected = SOURCES[kind]
    if source.name != name:
        raise ValueError('wrong source filename')
    data = source.read_bytes()
    if digest(data) != expected:
        raise ValueError('source differs from frozen boundary review: ' + str(source))
    text = data.decode()
    if source.suffix == '.py':
        text = python_source(kind, text)
        # Support the original harness imports when emitting to a new directory.
        text = text.replace('import argparse\n',
            f'import argparse\nimport sys\nsys.path.insert(0, {str(source.parent.resolve())!r})\n', 1)
        compile(text, str(output), 'exec')
    else:
        text = compiled_source(kind, text)
    output_data = text.encode()
    receipt = dict(schema=1, family=kind, source=str(source.resolve()),
        source_sha256=expected, output=str(output.resolve()),
        output_sha256=digest(output_data), preparer_sha256=digest(Path(__file__).read_bytes()),
        qualified=False, original_call_bit_controls='required in separate fresh processes',
        scope='Resident RGB inputs; imports/runtime configured before clock; scorer/runner construction and evaluation included; no OS-cache-cold claim')
    # Validate first; never overwrite an earlier artifact or its receipt.
    receipt_path = output.with_name(output.name + '.json')
    if output.exists() or receipt_path.exists():
        raise FileExistsError('new adapter and receipt paths required')
    with output.open('xb') as stream:
        stream.write(output_data)
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=sorted(SOURCES))
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.family, args.source, args.output)))


if __name__ == '__main__':
    main()
