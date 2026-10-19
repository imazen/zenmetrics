#!/usr/bin/env python3
"""Execute the verified fixed-factor PyIQA function on explicit RGB8 inputs.

Only the checked ssim function definition is loaded from its scratch copy;
the installed module's other functions, imports and classes stay unchanged.
Unrounded direct-function and original rounded-class conventions are labeled
separately. This acquires no timing ratio or undocumented numerical gate.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import struct
import traceback

from check_fixed_factor_oracle import check_manifest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['manifest', 'inputs', 'output']:
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--threads', type=int, required=True)
    p.add_argument('--convention', choices=['ssim-unrounded-y', 'ssim-unrounded-rgb', 'ms-ssim-unrounded-y',
                   'ssim-original-rounded-class', 'ms-ssim-original-rounded-class'])
    a = p.parse_args()
    if a.threads <= 0:
        p.error('positive CPU thread budget required')
    manifest = check_manifest(a.manifest)
    if manifest['source']['name'] != 'ssim-pyiqa':
        p.error('primary PyIQA SSIM source required')
    source = Path(manifest['source']['path']).resolve(strict=True)
    fixed = Path(manifest['output']['path']).resolve(strict=True)
    a.inputs = a.inputs.resolve(strict=True)
    inputs = json.loads(a.inputs.read_text())['pairs']
    a.output.mkdir()
    root = a.output.resolve()
    import numpy as np
    import torch
    import pyiqa.archs.ssim_arch as primary
    assert importlib.metadata.version('pyiqa') == '0.1.16'
    assert Path(primary.__file__).resolve() == source
    assert sha(source) == manifest['patch']['before_sha256']
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    tree = ast.parse(fixed.read_text(), filename=str(fixed))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'ssim']
    assert len(selected) == 1 and not selected[0].decorator_list
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(fixed), 'exec'), primary.__dict__)
    assert primary.ssim.__code__.co_filename == str(fixed)
    bindings = {}
    for name, function in [('ssim', primary.ssim), ('ms_ssim', primary.ms_ssim),
                           ('fspecial', primary.fspecial), ('filter2', primary.filter2),
                           ('preprocess_rgb', primary.preprocess_rgb)]:
        path = Path(inspect.getsourcefile(function)).resolve(strict=True)
        bindings[name] = {'path': str(path), 'sha256': sha(path)}
    rows = []
    (root / 'identity.json').write_text(json.dumps(dict(manifest_sha256=sha(a.manifest),
        input_manifest_sha256=sha(a.inputs), bindings=bindings, torch_version=torch.__version__,
        numpy_version=np.__version__, threads=a.threads), indent=2) + '\n')
    for row in inputs:
        tensors = []
        for role in ['reference', 'distorted']:
            path = (a.inputs.parent / row[role]).resolve(strict=True)
            assert sha(path) == row[role + '_sha256']
            data = path.read_bytes()
            assert len(data) == row['width'] * row['height'] * 3
            array = np.frombuffer(data, dtype=np.uint8).copy().reshape(row['height'], row['width'], 3)
            tensors.append(torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float64, device='cpu'))
        x, y = tensors
        def luma(t):
            return 0.299 * t[:, 0:1] + 0.587 * t[:, 1:2] + 0.114 * t[:, 2:3]
        lx, ly = luma(x), luma(y)
        calls = {
            'ssim-unrounded-y': lambda: primary.ssim(lx, ly, data_range=255, downsample=True),
            'ssim-unrounded-rgb': lambda: primary.ssim(x, y, data_range=255, downsample=True),
            'ms-ssim-unrounded-y': lambda: primary.ms_ssim(lx, ly, data_range=255, downsample=True),
            'ssim-original-rounded-class': lambda: primary.SSIM(downsample=True)(x / 255, y / 255),
            'ms-ssim-original-rounded-class': lambda: primary.MS_SSIM(downsample=True)(x / 255, y / 255),
        }
        for name, call in calls.items():
            if a.convention and a.convention != name:
                continue
            record = {'pair': row['pair'], 'size': row['size_name'], 'width': row['width'], 'height': row['height'],
                      'convention': name, 'factor': manifest['patch']['factor'], 'threads': a.threads,
                      'reference_sha256': row['reference_sha256'], 'distorted_sha256': row['distorted_sha256']}
            try:
                with torch.inference_mode():
                    values = []
                    for index in range(2):
                        value = float(call().item())
                        values.append(value)
                        with (root / 'calls.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(record, call=index, value=value,
                                score_bits=struct.pack('>d', value).hex())) + '\n')
                bits = [struct.pack('>d', value).hex() for value in values]
                assert bits[0] == bits[1], 'canonical score changed on repetition'
                record.update(status='scored', values=values, score_bits=bits)
            except Exception as error:
                record.update(status='canonical-error', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            rows.append(record)
            with (root / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print('CASE_DONE', row['pair'], row['size_name'], name, record['status'], flush=True)
    check_manifest(a.manifest)
    assert sha(source) == manifest['patch']['before_sha256']
    result = {'manifest_sha256': sha(a.manifest), 'input_manifest_sha256': sha(a.inputs),
              'bindings': bindings, 'torch_version': torch.__version__, 'numpy_version': np.__version__,
              'threads': a.threads, 'rows': rows, 'qualified_timing_cells': 0,
              'scope': 'direct-function unrounded conventions and unchanged original rounded classes are separate observations; no undocumented gate'}
    (root / 'results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
