#!/usr/bin/env python3
"""Caller-parameter precision controls; unchanged defaults remain the primary.

These are diagnostic counterfactuals, not replacement canonical scores or a
new acceptance bound. No upstream source file or global default dtype changes.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import struct


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joined', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = []
    for path in a.joined.glob('*.tsv'):
        rows += list(csv.DictReader(path.open(), delimiter='\t'))
    names = ['ssim-pyiqa', 'ssimc-pyiqa-f32', 'ms-ssim-pyiqa-f32']
    selected = [max((r for r in rows if r['metric'] == name and r['threads'] == '1' and r['candidate_abs_delta']),
                    key=lambda r: float(r['candidate_abs_delta'])) for name in names]
    inputs = {(r['pair'], r['size_name']): r for r in json.loads(a.inputs.read_text())['pairs']}
    a.output.mkdir()
    root = a.output.resolve()
    import numpy as np
    import torch
    import pyiqa.archs.ssim_arch as primary
    from pyiqa.utils.color_util import rgb2yiq
    assert importlib.metadata.version('pyiqa') == '0.1.16'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    files = {str(Path(inspect.getsourcefile(f)).resolve()): sha(Path(inspect.getsourcefile(f)))
             for f in [primary.ssim, primary.ms_ssim, primary.fspecial, primary.filter2, primary.preprocess_rgb, rgb2yiq]}
    identity = dict(files=files, torch_version=torch.__version__, numpy_version=np.__version__,
                    torch_default_dtype=str(torch.get_default_dtype()), inputs_sha256=sha(a.inputs), selected=selected,
                    scope='diagnostic caller-provided f64 Gaussian and f64 rounded luma; original defaults retained',
                    qualified_timing_cells=0)
    (root / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    results = []
    for record in selected:
        row = inputs[(record['pair'], record['size'])]
        tensors = []
        for role in ['reference', 'distorted']:
            path = (a.inputs.parent / row[role]).resolve(strict=True)
            assert sha(path) == row[role + '_sha256']
            data = np.frombuffer(path.read_bytes(), dtype=np.uint8).copy().reshape(row['height'], row['width'], 3)
            tensors.append(torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).to(torch.float64))
        x, y = tensors
        def luma(t):
            return 0.299*t[:, 0:1] + 0.587*t[:, 1:2] + 0.114*t[:, 2:3]
        lx, ly = luma(x), luma(y)
        if record['metric'] == 'ssimc-pyiqa-f32':
            px, py = x, y
            function = primary.ssim
        else:
            px, py = lx, ly
            function = primary.ms_ssim if record['metric'].startswith('ms-') else primary.ssim
        if record['metric'] == 'ssim-pyiqa':
            canonical = lambda: primary.SSIM(downsample=False)(x/255, y/255)
            px, py = torch.round(lx), torch.round(ly)
            original_planes = [primary.preprocess_rgb(t/255, True, 255, 'yiq').to(torch.float64) for t in [x, y]]
            different = [int((v != orig).sum().item()) for v, orig in zip([px, py], original_planes)]
        else:
            canonical = lambda: function(px, py, data_range=255, downsample=False)
            different = None
        yy, xx = np.ogrid[-5:6, -5:6]
        window = np.exp(-(xx*xx+yy*yy)/(2*1.5*1.5)); window /= window.sum()
        win = torch.from_numpy(window).repeat(px.shape[1], 1, 1, 1)
        with torch.inference_mode():
            original = [float(canonical().item()) for _ in range(2)]
            input_control = float(function(px, py, data_range=255, downsample=False).item())
            precision_control = float(function(px, py, win=win, data_range=255, downsample=False).item())
        assert struct.pack('>d', original[0]) == struct.pack('>d', original[1])
        assert original[0] == float(record['canonical']), 'original primary result did not reproduce'
        ours = float(record['candidate'])
        result = dict(metric=record['metric'], pair=record['pair'], size=record['size'], ours=ours,
                      original_primary=original[0], original_abs_delta=abs(ours-original[0]),
                      f64_luma_control=input_control, f64_luma_abs_delta=abs(ours-input_control),
                      f64_window_control=precision_control, f64_window_abs_delta=abs(ours-precision_control),
                      rounded_luma_pixels_differing=different, default_window_dtype=str(primary.fspecial(11,1.5,1).dtype))
        results.append(result)
        print(json.dumps(result), flush=True)
        (root / (record['metric'] + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    assert all(sha(Path(path)) == value for path, value in files.items())
    (root / 'results.json').write_text(json.dumps(dict(rows=results, qualified_timing_cells=0), indent=2) + '\n')


if __name__ == '__main__':
    main()
