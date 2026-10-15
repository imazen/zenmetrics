"""Shared export qualification: required image pairs and the canonical JND gate."""
import csv
import math
from pathlib import Path


def score_tolerance(model):
    with Path(__file__).with_name('jnd_scale.tsv').open() as stream:
        scales = dict(csv.reader((line for line in stream if line.strip() and not line.startswith('#')), delimiter='\t'))
    scale = float(scales[model])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f'{model}: invalid JND scale {scale}')
    return scale * 1e-3


def assert_parity(model, got, want):
    tolerance = score_tolerance(model)
    delta = abs(float(got) - float(want))
    if not math.isfinite(delta) or delta > tolerance:
        raise AssertionError(f'{model}: |d|={delta} exceeds {tolerance} score units (1e-3 JND)')
    return delta


def golden_rows(path, image_dir, model):
    with Path(path).open() as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if reader.fieldnames != ['image', 'size', model, model + '_ref']:
            raise ValueError(f'{model}: invalid golden columns {reader.fieldnames}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{model}: goldens must contain image pairs')
    for row in rows:
        for kind in ('ref', 'dist'):
            image = Path(image_dir) / f"{row['image']}_{row['size']}_{kind}.png"
            if not image.is_file():
                raise FileNotFoundError(image)
        for key in (model, model + '_ref'):
            if not math.isfinite(float(row[key])):
                raise ValueError(f'{model}: nonfinite golden {row[key]}')
    return rows
