#!/usr/bin/env python3
"""Score published crop rows with a named CPU reference, retaining refusals.

References are loaded once. The output is evaluation evidence, never training
data. Package/source identities and exact options are supplied by the run manifest.
"""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resident_adapter(spec, metric):
    """Reuse the established call-site adapter's setup and score definitions.

    Strip only its argument parser, raw-file loading and timing loop. Functions,
    model construction, preprocessing and numerical operators stay unchanged.
    This transformation applies to our adapters, never to third-party sources.
    """
    path = Path(spec['path'])
    sys.path.insert(0, str(path.parent))
    module = load(path, 'aic_resident_adapter')
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    nodes = []
    for node in main.body:
        text = ast.unparse(node)
        if isinstance(node, (ast.For, ast.With)):
            break
        if (text.startswith(('p = argparse.', 'parser = argparse.', 'p.add_argument(', 'parser.add_argument(',
                             'a = p.parse_args()', 'args = parser.parse_args()', 'r, d = [np.fromfile(',
                             'reference, distorted = [np.fromfile(', 'if a.setup_only:', 'if args.setup_only:'))
                or text.startswith(('torch.set_num_threads(', 'torch.set_num_interop_threads('))):
            continue
        nodes.append(node)
    namespace = module.__dict__
    args = types.SimpleNamespace(metric=metric, width=840, height=944, threads=1,
                                 setup_only=False, samples=0, **spec.get('args', {}))
    for key, value in vars(args).items():
        if key in ('source', 'extra_path', 'weights', 'fixed_manifest', 'config', 'deepdc_convention') and value:
            setattr(args, key, Path(value))
    namespace.update(a=args, args=args)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    if metric in ('arniqa', 'maniqa', 'liqe', 'hyperiqa', 'paq2piq') and 'single' in namespace:
        return lambda r, d: float(namespace['single'](d))
    if 'score' in namespace:
        fn = namespace['score']
        def score(r, d):
            namespace.update(r=r, d=d)
            if spec.get('pil_arguments'):
                from PIL import Image
                value = fn(Image.fromarray(r), Image.fromarray(d))
            else:
                value = fn()
            return float(value[0] if isinstance(value, tuple) else value)
        return score
    if 'single' in namespace:
        return lambda r, d: float(namespace['single'](d))
    raise ValueError('adapter did not expose its score or single callable')


def scorer(metric, config):
    import numpy as np
    if metric.startswith('wadiqam-'):
        spec = config['resident_adapters'][metric]
        path = Path(spec['path'])
        module = load(path, 'aic_original_wadiqam')
        module.chainer.global_config.enable_backprop = False
        module.chainer.global_config.train = False
        model, oracle = module.load_model(json.loads(Path(spec['args']['config']).read_text()), metric)
        return lambda r, d: float(oracle.score_image(model, d, r) if '-fr-' in metric
                                 else oracle.score_image(model, d))
    import torch
    if metric in config.get('resident_adapters', {}):
        return resident_adapter(config['resident_adapters'][metric], metric)
    if metric in ('pyiqa-fsim', 'pyiqa-fsimc', 'pyiqa-vsi'):
        import pyiqa
        options = {'chromatic': False} if metric == 'pyiqa-fsim' else {}
        model = pyiqa.create_metric('vsi' if metric == 'pyiqa-vsi' else 'fsim', device='cpu', **options)
        def score(r, d):
            x, y = [torch.from_numpy(a.copy()).permute(2, 0, 1)[None].float()/255 for a in (d, r)]
            return float(model(x, y).item())
        return score
    if metric in ('ssim', 'psnr-y', 'psnr-y601', 'psnr-y-libvmaf'):
        from skimage.metrics import structural_similarity, peak_signal_noise_ratio
        def score(r, d):
            r, d = r.astype(np.float64), d.astype(np.float64)
            if metric == 'ssim':
                _, values = structural_similarity(r, d, data_range=255,
                    channel_axis=2, gaussian_weights=True, sigma=1.5,
                    use_sample_covariance=False, full=True)
                return float(values.mean())
            weights = (.299, .587, .114) if metric == 'psnr-y601' else (.2126, .7152, .0722)
            def plane(x):
                y = sum(w*x[:, :, c] for c, w in enumerate(weights))
                if metric == 'psnr-y-libvmaf':
                    y = np.floor(16 + 219*y/255 + .5).clip(0, 255)
                return y
            return float(peak_signal_noise_ratio(plane(r), plane(d), data_range=255))
        return score
    if 'pyiqa-f32' in metric:
        import pyiqa.archs.ssim_arch as primary
        source = Path(primary.__file__).read_text()
        target = 'f = max(1, round(min(X.size()[-2:]) / 256))'
        if source.count(target) != 1:
            raise ValueError('expected exactly one PyIQA factor expression')
        k = int(metric.rsplit('-k', 1)[1]) if '-k' in metric else 1
        patched = types.ModuleType('aic_pyiqa_fixed')
        patched.__package__ = 'pyiqa.archs'
        exec(compile(source.replace(target, f'f = {k}').replace('@ARCH_REGISTRY.register()', ''),
                     '<single-factor-patched-pyiqa>', 'exec'), patched.__dict__)
        def score(r, d):
            x, y = [torch.from_numpy(a.astype(np.float64)).permute(2, 0, 1)[None]
                    for a in (r, d)]
            if not metric.startswith('ssimc-'):
                x, y = [.299*a[:, :1]+.587*a[:, 1:2]+.114*a[:, 2:3] for a in (x, y)]
            fn = patched.ms_ssim if metric.startswith('ms-') else patched.ssim
            return float(fn(x, y, data_range=255, downsample=True).mean())
        return score
    if metric == 'nlpd':
        root = Path(config['nlpd_source'])
        sys.path.insert(0, str(root))
        model = load(root/'nlp.py', 'aic_author_nlpd').LaplacianPyramid(6, dims=3)
        def score(r, d):
            x, y = [torch.from_numpy(a.copy()).permute(2, 0, 1)[None].contiguous().float()/255
                    for a in (r, d)]
            return float(model.compare(x, y).item())
        return score
    if metric == 'iwssim':
        root = Path(config['iwssim_source'])
        def legacy_eig(matrix, eigenvectors=False):
            if eigenvectors:
                values, vectors = torch.linalg.eig(matrix)
                if torch.count_nonzero(values.imag) or torch.count_nonzero(vectors.imag):
                    raise ValueError('complex covariance requires actual legacy torch runtime')
                return torch.view_as_real(values), vectors.real
            return torch.view_as_real(torch.linalg.eigvals(matrix)), torch.empty(0, dtype=matrix.dtype)
        torch.eig = legacy_eig
        module = load(root/'IW_SSIM_PyTorch.py', 'aic_author_iwssim')
        gray = load(root/'utils.py', 'aic_author_gray').rgb2gray
        model = module.IW_SSIM(use_cuda=False, use_double=False)
        return lambda r, d: float(model.test(gray(r), gray(d)).item())
    if metric in ('ms-gmsd', 'ms-gmsdc'):
        import piq
        def score(r, d):
            x, y = [torch.from_numpy(a.copy()).permute(2, 0, 1)[None].float()/255 for a in (r, d)]
            return float(piq.multi_scale_gmsd(x, y, data_range=1., chromatic=metric.endswith('c')).item())
        return score
    raise ValueError('no registered reference: '+metric)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('metric')
    p.add_argument('pairs', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--limit', type=int)
    p.add_argument('--threads', type=int, default=1)
    a = p.parse_args()
    import numpy as np
    torch = None
    if a.metric != 'musiq' and not a.metric.startswith('wadiqam-'):
        import torch
    from PIL import Image
    if torch is not None:
        torch.set_num_threads(a.threads)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
    config = json.loads(a.config.read_text())
    if a.metric == 'musiq':
        score = resident_adapter(config['resident_adapters'][a.metric], a.metric)
    else:
        score = scorer(a.metric, config)
    pairs = list(csv.DictReader(a.pairs.open(), delimiter='\t'))
    if a.limit is not None:
        pairs = pairs[:a.limit]
    prior = set()
    if a.output.exists():
        prior = {json.loads(line)['distorted'] for line in a.output.read_text().splitlines()}
    ref_path, reference = None, None
    with a.output.open('a') as output:
        for index, row in enumerate(pairs):
            if row['distorted'] in prior:
                continue
            if ref_path != row['ref_path']:
                ref_path = row['ref_path']
                with Image.open(ref_path) as image:
                    reference = np.asarray(image.convert('RGB')).copy()
            with Image.open(row['dist_path']) as image:
                distorted = np.asarray(image.convert('RGB')).copy()
            if reference.shape != (944, 840, 3) or distorted.shape != reference.shape:
                raise ValueError('crop shape does not match July contract')
            record = dict(distorted=row['distorted'], metric=a.metric)
            try:
                value = score(reference, distorted)
                record['value'] = value
                if not np.isfinite(value):
                    record['error'] = 'reference returned nonfinite'
            except Exception as error:
                record['error'] = f'{type(error).__name__}: {error}'
            output.write(json.dumps(record, allow_nan=True)+'\n')
            output.flush()
            if index % 100 == 0:
                print(f'{a.metric}: {index+1}/{len(pairs)}', flush=True)
    print(f'{a.metric}: finished {len(pairs)} requested rows', flush=True)


if __name__ == '__main__':
    main()
