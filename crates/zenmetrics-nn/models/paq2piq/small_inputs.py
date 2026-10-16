#!/usr/bin/env python3
"""Make the two eval-only small golden pairs by cropping the 1415 64px rung.

Usage: small_inputs.py LADDER_DIR NONSQUARE_DIR
Existing images are verified pixel-for-pixel and retained.
"""
import sys
from pathlib import Path
from PIL import Image

source, destination = map(Path, sys.argv[1:])
destination.mkdir(parents=True, exist_ok=True)
for size, width, height in [('32', 32, 32), ('33x47', 33, 47)]:
    for kind in ('ref', 'dist'):
        with Image.open(source / f'1415_64_{kind}.png') as image:
            crop = image.convert('RGB').crop((0, 0, width, height))
        target = destination / f'1415_{size}_{kind}.png'
        if target.exists():
            with Image.open(target) as existing:
                assert existing.size == crop.size and existing.convert('RGB').tobytes() == crop.tobytes(), target
        else:
            crop.save(target)
        print(target.name)
