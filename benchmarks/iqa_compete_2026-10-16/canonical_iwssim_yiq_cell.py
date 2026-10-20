#!/usr/bin/env python3
"""Original IW-SSIM body with original PIQ rgb2yiq unrounded Y ingress."""
from pathlib import Path
import sys
import types

import torch

import canonical_iwssim_cell as author


def main():
    # Load the explicitly supplied PIQ package after choosing the torch runtime.
    sys.path.insert(0, str(Path(sys.argv.pop(1)).resolve(strict=True)))
    from piq.functional import rgb2yiq
    original = author.load

    def gray(rgb):
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float()
        return rgb2yiq(tensor)[0, 0].numpy()

    def load(path, name):
        return types.SimpleNamespace(rgb2gray=gray) if path.name == 'utils.py' else original(path, name)

    author.load = load
    author.main()


if __name__ == '__main__':
    main()
