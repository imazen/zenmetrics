#!/usr/bin/env python3
"""Observe studio-709 rounding conventions without changing either score route.

The integer-numerator definition is an independent mathematical control, not
an upstream software reference or an acceptance gate. Caller supplies resource
admission; per-image observations remain outside git.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pairs', type=Path)
    parser.add_argument('native', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--limit', type=int, default=1)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    from skimage.metrics import peak_signal_noise_ratio
    control = Path(__file__).parent.parent/'iqa_compete_2026-10-16/check_psnr_definitions.py'
    sys.path.insert(0, str(control.parent))
    spec = importlib.util.spec_from_file_location('aic_exact_psnr_control', control)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = {row['distorted']: row for row in csv.DictReader(args.native.open(), delimiter='\t')}
    observations = []
    for row in list(csv.DictReader(args.pairs.open(), delimiter='\t'))[:args.limit]:
        planes = []
        sources = {}
        differences = []
        for role in ('ref_path', 'dist_path'):
            source = Path(row[role])
            with Image.open(source) as image:
                rgb = np.asarray(image.convert('RGB'))
            x = rgb.astype(np.float64)
            # The frozen reference adapter weights code values before /255.
            code_first = np.floor(16+219*sum(w*x[..., c] for c, w in enumerate((.2126, .7152, .0722)))/255+.5)
            # Frozen native 4d833195 divides each channel before weighting.
            normalized_first = np.floor(16+219*sum(w*(x[..., c]/255) for c, w in enumerate((.2126, .7152, .0722)))+.5)
            exact = module.exact_planes(rgb)['studio709']
            planes.append((code_first, normalized_first, exact))
            sources[role] = dict(path=str(source), sha256=sha(source))
            differences.append(dict(role=role,
                code_vs_normalized=int(np.count_nonzero(code_first != normalized_first)),
                code_vs_integer=int(np.count_nonzero(code_first != exact)),
                normalized_vs_integer=int(np.count_nonzero(normalized_first != exact))))
        values = [float(peak_signal_noise_ratio(planes[0][i], planes[1][i], data_range=255)) for i in range(3)]
        own = native[row['distorted']]
        if 'psnr_y_libvmaf' not in own or not all(math.isfinite(value) for value in values):
            raise ValueError('diagnostic requires one native score and three finite controls')
        observations.append(dict(distorted=row['distorted'], native_export=own['psnr_y_libvmaf'],
            code_first_psnr=values[0], normalized_first_psnr=values[1], integer_definition_psnr=values[2],
            differing_pixels=differences, sources=sources))
    report = dict(status='diagnostic; no acceptance gate', observations=observations,
        inputs={str(path):sha(path) for path in (args.pairs, args.native, Path(__file__), control)},
        native_source='4d833195:crates/zenmetrics-cli/src/metrics/classical.rs::Kind::PsnrYLibvmaf',
        numpy=np.__version__)
    with args.output.open('x') as target:
        target.write(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
