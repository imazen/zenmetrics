#!/usr/bin/env python3
"""Original BasicSR NIQE with immutable parameter I/O outside pair timing."""
import argparse
import importlib
import json
from pathlib import Path
import struct
import sys
import time
import types


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('metric', choices=['niqe'])
    p.add_argument('width', type=int)
    p.add_argument('height', type=int)
    p.add_argument('reference', type=Path)
    p.add_argument('distorted', type=Path)
    p.add_argument('threads', type=int)
    p.add_argument('samples', type=int)
    p.add_argument('--extra-path', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    sys.path.insert(0, str(a.extra_path))
    root = a.extra_path/'basicsr'
    for name, directory in [('basicsr', root), ('basicsr.metrics', root/'metrics'),
                            ('basicsr.utils', root/'utils')]:
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
    from basicsr.utils.color_util import bgr2ycbcr
    sys.modules['basicsr.utils'].bgr2ycbcr = bgr2ycbcr
    module = importlib.import_module('basicsr.metrics.niqe')
    images = [np.fromfile(path, dtype=np.uint8).reshape(a.height, a.width, 3)
              for path in (a.reference, a.distorted)]
    parameters = (Path(module.__file__).resolve().parent/'niqe_pris_params.npz').resolve(strict=True)
    with np.load(parameters) as archive:
        resident = {name: archive[name] for name in archive.files}
    assert set(resident) == {'mu_pris_param', 'cov_pris_param', 'gaussian_window'}
    for value in resident.values():
        value.flags.writeable = False

    class ResidentArchive:
        def __getitem__(self, name):
            # npz member reads originally return owned arrays. Preserve that
            # ownership and its allocation/release inside each timed call.
            return resident[name].copy(order='K')

    def resident_load(path, *args, **kwargs):
        if Path(path).resolve() != parameters or args or kwargs:
            raise ValueError('Unexpected parameter load in NIQE reference')
        return ResidentArchive()

    resident_numpy = types.ModuleType('numpy')
    resident_numpy.__dict__.update(vars(np))
    resident_numpy.load = resident_load

    def score():
        reference, distorted = [float(module.calculate_niqe(rgb[:, :, ::-1].copy(), crop_border=0))
                                for rgb in images]
        return distorted, reference

    original = score()
    module.np = resident_numpy
    cached = score()
    bits = lambda values: [struct.pack('>d', value).hex() for value in values]
    if bits(original) != bits(cached):
        raise ValueError('Resident parameters changed original NIQE score bits')
    print('CANONICAL '+json.dumps(dict(metric='niqe', torch=torch.__version__, numpy=np.__version__,
        parameter_path=str(parameters), original_score_bits=bits(original), resident_score_bits=bits(cached),
        scope='Resident RGB8 pair conversion, unchanged BasicSR callable, owned parameter-array copies and result release; immutable archive I/O excluded')), flush=True)
    for index in range(a.samples+1):
        start = time.perf_counter_ns()
        value, reference = score()
        elapsed = time.perf_counter_ns()-start
        print('REFERENCE_SCORE '+repr(reference), flush=True)
        print(('COLD' if index == 0 else 'SAMPLE')+f' {elapsed} {value:.17g}', flush=True)


if __name__ == '__main__':
    main()
