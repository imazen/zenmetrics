#!/usr/bin/env python3
"""Original WaDIQaM models with a local CPU overlay and no gradient graph."""
import argparse
import ast
import importlib
import json
import numbers
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

# Python startup may already have selected setuptools' distutils shim.
if os.environ.get('SETUPTOOLS_USE_DISTUTILS') != 'stdlib':
    os.execve(sys.executable, [sys.executable, '-B', *sys.argv],
              dict(os.environ, SETUPTOOLS_USE_DISTUTILS='stdlib'))

import chainer
import numpy as np
from chainer import serializers


def load_model(config, metric):
    """Keep Chainer's own CUDA backend intact; overlay author device aliases."""
    sys.path.insert(0, config['upstream'])
    mode, dataset = metric.split('-')[1:]
    available = chainer.cuda.check_cuda_available
    try:
        chainer.cuda.check_cuda_available = lambda: None
        module = importlib.import_module('fr_model' if mode == 'fr' else 'nr_model')
    finally:
        chainer.cuda.check_cuda_available = available
    module.cuda = SimpleNamespace(cupy=np, check_cuda_available=lambda: None)
    if mode == 'nr':
        module.xp = np
    model = (module.FRModel if mode == 'fr' else module.Model)(top='weighted')
    serializers.load_hdf5(str(Path(config['weights']) / f'{mode}_{dataset}_weighted.model'), model)
    tree = ast.parse(Path(config['oracle']).read_text())
    names = {'extract_patches', 'score_image'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError('original CPU oracle functions missing')
    namespace = dict(np=np, numbers=numbers, as_strided=np.lib.stride_tricks.as_strided)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), config['oracle'], 'exec'), namespace)
    return model, SimpleNamespace(score_image=namespace['score_image'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('metric', choices=['wadiqam-fr-live', 'wadiqam-fr-tid', 'wadiqam-nr-live', 'wadiqam-nr-tid'])
    parser.add_argument('width', type=int);parser.add_argument('height', type=int)
    parser.add_argument('reference', type=Path);parser.add_argument('distorted', type=Path)
    parser.add_argument('threads', type=int);parser.add_argument('samples', type=int)
    parser.add_argument('--setup-only', action='store_true')
    args = parser.parse_args()
    chainer.global_config.enable_backprop = False
    chainer.global_config.train = False
    chainer.global_config.cudnn_deterministic = True
    print('REFERENCE_INFERENCE enable_backprop=false', flush=True)
    model, oracle = load_model(json.loads(args.config.read_text()), args.metric)
    if args.setup_only:
        print(json.dumps(dict(metric=args.metric, setup='ready', chainer=chainer.__version__, numpy=np.__version__)))
        return
    reference, distorted = [np.fromfile(path, np.uint8).reshape(args.height, args.width, 3)
                            for path in (args.reference, args.distorted)]
    for index in range(args.samples + 1):
        start = time.perf_counter_ns()
        if '-fr-' in args.metric:
            value = oracle.score_image(model, distorted, reference)
            reference_score = None
        else:
            value = oracle.score_image(model, distorted)
            reference_score = oracle.score_image(model, reference)
        elapsed = time.perf_counter_ns() - start
        print(('COLD' if index == 0 else 'SAMPLE') + f' {elapsed} {value:.17g}', flush=True)
        if reference_score is not None:
            print(f'PAIR_REFERENCE {reference_score:.17g}', flush=True)


if __name__ == '__main__':
    main()
