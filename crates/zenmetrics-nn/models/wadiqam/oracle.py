#!/usr/bin/env python3
"""WaDIQaM oracle — dmaniry/deepIQA @51dece994d73f43dbd2ea44d3f2ef581014194ad
run on CPU. Upstream evaluate.py drives the GPU; this harness keeps the
upstream model classes (fr_model.FRModel / nr_model.Model) and
extract_patches verbatim, patches out only the cupy device calls, and
batch-scores a ladder into the zenmetrics-nn goldens TSV shape.

usage: oracle.py <deepIQA-dir> <model-file> <top> <fr|nr> <col> <out.tsv> <ladder-dir>...
FR rows:  image\tsize\t<col>           (pair score)
NR rows:  image\tsize\t<col>\t<col>_ref (dist and ref scores)
"""
import numbers
import sys

import numpy as np
from numpy.lib.stride_tricks import as_strided

import chainer

# CPU shim: upstream calls cuda.check_cuda_available() at nr_model import
# and reads xp = cuda.cupy inside forward() for ones_like/repeat/zeros.
chainer.cuda.check_cuda_available = lambda: None

DEEP = sys.argv[1]
sys.path.insert(0, DEEP)

from chainer import serializers  # noqa: E402

import imageio.v2 as imageio  # noqa: E402


def extract_patches(arr, patch_shape=(32, 32, 3), extraction_step=32):
    # verbatim from deepIQA evaluate.py
    arr_ndim = arr.ndim
    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)
    patch_strides = arr.strides
    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides
    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1
    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))
    return as_strided(arr, shape=shape, strides=strides)


def score_image(model, img, fr_ref=None):
    """evaluate.py's loop: patches -> per-batch forward -> y, a -> sum(y*a)/sum(a)."""
    xp = np
    patches = extract_patches(img)
    X = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))
    X_ref = None
    if fr_ref is not None:
        patches_r = extract_patches(fr_ref)
        X_ref = np.transpose(patches_r.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))
    t = xp.zeros((1, 1), np.float32)
    y, w = [], []
    batchsize = min(2000, X.shape[0])
    for i in range(0, X.shape[0], batchsize):
        X_batch = xp.array(X[i:i + batchsize].astype(np.float32))
        if fr_ref is not None:
            X_ref_batch = xp.array(X_ref[i:i + batchsize].astype(np.float32))
            model.forward(X_batch, X_ref_batch, t, False,
                          n_patches_per_image=X_batch.shape[0])
        else:
            model.forward(X_batch, t, False, X_batch.shape[0])
        y.append(np.asarray(model.y[0].data).reshape((-1,)))
        w.append(np.asarray(model.a[0].data).reshape((-1,)))
    y = np.concatenate(y)
    w = np.concatenate(w)
    return float(np.sum(y * w) / np.sum(w))


def main():
    model_file, top, mode, col, out_path = sys.argv[2:7]
    ladders = sys.argv[7:]

    chainer.global_config.train = False
    chainer.global_config.cudnn_deterministic = True

    if mode == "fr":
        from fr_model import FRModel
        model = FRModel(top=top)
    else:
        from nr_model import Model
        model = Model(top=top)
    serializers.load_hdf5(model_file, model)
    # now that weights are bound, forward() may read cuda.cupy as np
    chainer.cuda.cupy = np

    rows = {}
    for ladder in ladders:
        import os
        for f in sorted(os.listdir(ladder)):
            if not f.endswith("_dist.png"):
                continue
            stem = f[: -len("_dist.png")]
            image, size = stem.split("_", 1)
            dist = imageio.imread(os.path.join(ladder, f))
            ref = imageio.imread(os.path.join(ladder, stem + "_ref.png"))
            if mode == "fr":
                rows.setdefault((image, size), score_image(model, dist, ref))
            else:
                rows.setdefault(
                    (image, size), (score_image(model, dist), score_image(model, ref))
                )

    with open(out_path, "w") as fh:
        if mode == "fr":
            fh.write(f"image\tsize\t{col}\n")
            for (image, size), s in sorted(rows.items()):
                fh.write(f"{image}\t{size}\t{s:.10g}\n")
        else:
            fh.write(f"image\tsize\t{col}\t{col}_ref\n")
            for (image, size), (sd, sr) in sorted(rows.items()):
                fh.write(f"{image}\t{size}\t{sd:.10g}\t{sr:.10g}\n")
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
