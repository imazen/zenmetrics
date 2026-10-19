#!/usr/bin/env python3
"""Resident primary Python references with RGB8 conversion inside each call.

Optional source roots and parameter files are explicit command arguments. Model
parameters are immutable resident state; owned temporary results remain timed.
"""
import argparse
import ast
import importlib
import json
from pathlib import Path
import sys
import time
import types


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('metric')
    p.add_argument('width', type=int)
    p.add_argument('height', type=int)
    p.add_argument('reference', type=Path)
    p.add_argument('distorted', type=Path)
    p.add_argument('threads', type=int)
    p.add_argument('samples', type=int)
    p.add_argument('--weights', type=Path)
    p.add_argument('--extra-path', type=Path)
    p.add_argument('--fixed-manifest', type=Path)
    a = p.parse_args()
    if a.extra_path:
        sys.path.insert(0, str(a.extra_path))
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    r, d = [np.fromfile(path, dtype=np.uint8).reshape(a.height, a.width, 3)
            for path in (a.reference, a.distorted)]

    if a.metric == 'brisque':
        import piq
        module = importlib.import_module('piq.brisque')
        weights = torch.load(a.weights, map_location='cpu', weights_only=True)
        # PIQ mutates its owned support-vector tensor in place. Returning clones
        # preserves that ownership while excluding immutable parameter I/O.
        module.load_url = lambda *args, **kwargs: tuple(v.clone() for v in weights)
        def score_one(rgb):
            x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).contiguous().float()
            return float(piq.brisque(x, data_range=255., reduction='none').item())
        def score():
            reference = score_one(r)
            return score_one(d), reference
    elif a.metric == 'niqe':
        # Avoid BasicSR's unrelated dataset/model imports. The actual NIQE,
        # MATLAB resize, color conversion and registry files stay unchanged.
        root = a.extra_path/'basicsr'
        for name, directory in [('basicsr', root), ('basicsr.metrics', root/'metrics'),
                                ('basicsr.utils', root/'utils')]:
            package = types.ModuleType(name)
            package.__path__ = [str(directory)]
            sys.modules[name] = package
        from basicsr.utils.color_util import bgr2ycbcr
        sys.modules['basicsr.utils'].bgr2ycbcr = bgr2ycbcr
        from basicsr.metrics.niqe import calculate_niqe
        def score():
            reference = float(calculate_niqe(r[:, :, ::-1].copy(), crop_border=0))
            return float(calculate_niqe(d[:, :, ::-1].copy(), crop_border=0)), reference
    elif a.metric == 'piqe':
        from pypiqe import piqe
        def score():
            reference = float(piqe(r[:, :, ::-1].copy())[0])
            return float(piqe(d[:, :, ::-1].copy())[0]), reference
    elif a.metric in ('flip', 'flip-v17'):
        import flip_evaluator
        def score():
            x, y = r.astype(np.float32)/255, d.astype(np.float32)/255
            value = flip_evaluator.evaluate(x, y, 'LDR', inputsRGB=True,
                                            applyMagma=False, computeMeanError=True)
            return float(value[1]), None
    elif 'ssim' in a.metric and 'pyiqa' in a.metric:
        import pyiqa.archs.ssim_arch as primary
        if a.fixed_manifest:
            from check_fixed_factor_oracle import check_manifest
            manifest = check_manifest(a.fixed_manifest)
            fixed = Path(manifest['output']['path'])
            tree = ast.parse(fixed.read_text(), filename=str(fixed))
            definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ssim']
            if len(definitions) != 1:
                raise ValueError('one checked SSIM definition required')
            exec(compile(ast.Module(body=definitions, type_ignores=[]), str(fixed), 'exec'), primary.__dict__)
        def score():
            x, y = [torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).contiguous().double()
                    for rgb in (r, d)]
            multiscale = a.metric.startswith('ms-')
            if a.metric in ('ssim-pyiqa', 'ms-ssim-pyiqa'):
                model = primary.MS_SSIM() if multiscale else primary.SSIM()
                return float(model(x/255, y/255).item()), None
            if not a.metric.startswith('ssimc-'):
                x = .299*x[:, :1]+.587*x[:, 1:2]+.114*x[:, 2:3]
                y = .299*y[:, :1]+.587*y[:, 1:2]+.114*y[:, 2:3]
            function = primary.ms_ssim if multiscale else primary.ssim
            return float(function(x, y, data_range=255, downsample=bool(a.fixed_manifest)).item()), None
    else:
        raise ValueError('no primary adapter for '+a.metric)
    print('CANONICAL '+json.dumps(dict(metric=a.metric, threads=torch.get_num_threads(),
                                      torch=torch.__version__, numpy=np.__version__)), flush=True)
    with torch.inference_mode():
        for index in range(a.samples+1):
            start = time.perf_counter_ns()
            value, reference = score()
            elapsed = time.perf_counter_ns()-start
            if reference is not None:
                print('REFERENCE_SCORE '+repr(reference), flush=True)
            print(('COLD' if index == 0 else 'SAMPLE')+f' {elapsed} {value:.17g}', flush=True)


if __name__ == '__main__':
    main()
