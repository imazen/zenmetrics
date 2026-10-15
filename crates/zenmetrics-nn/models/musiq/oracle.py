"""MUSIQ goldens via the upstream google-research JAX oracle.

usage: oracle.py LADDER_DIR UPSTREAM_DIR CKPT_NPZ OUT_TSV [GOLDEN_SIZES]

UPSTREAM_DIR is a clone of github.com/google-research/google-research (the
licence-audited pin is 3ef5211) — the `musiq` package must import from it.
CKPT_NPZ is the official KonIQ checkpoint `koniq_ckpt.npz` from the project's
GCS bucket (gs://gresearch/musiq/). The runtime is the legacy stack upstream
requires: Python 3.9, jax/jaxlib 0.3.25, flax 0.3.3 (legacy `flax.nn`),
tensorflow CPU — a venv recipe is in card.toml.

Runs `run_predict_image.py`'s path verbatim per image: `decode_jpeg` ->
`get_multiscale_patches` -> `Model.call`, CPU fp32. Writes
`image, size, musiq, musiq_ref` — the parity gate's upstream golden.

TF eager execution is required: upstream's `_pad_or_cut_to_max_seq_len`
calls `len(tf.shape(x))`, which only works eagerly.
"""
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

LADDER, UPSTREAM, CKPT, OUT = sys.argv[1:5]
SIZES = {int(s) for s in (sys.argv[5] if len(sys.argv) > 5
                         else '64,128,256,512,1024').split(',')}

sys.path.insert(0, UPSTREAM)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow.compat.v2 as tf  # noqa: E402

import musiq.model.multiscale_transformer as model_mod  # noqa: E402
import musiq.model.preprocessing as pp_lib  # noqa: E402
from musiq.run_predict_image import get_params_and_config  # noqa: E402

model_config, pp_config, params = get_params_and_config(CKPT)
model = model_mod.Model.partial(num_classes=1, train=False, **model_config)
pp_fn = pp_lib.get_preprocess_fn(**pp_config)


def oracle(path):
    data = pp_fn(dict(image=tf.constant(open(path, 'rb').read())))
    img = np.asarray(tf.expand_dims(data['image'], axis=0).numpy())
    logits = model.call(params, jnp.asarray(img))
    return float(np.asarray(logits[0, 0]))


with open(OUT, 'w') as g:
    g.write('image\tsize\tmusiq\tmusiq_ref\n')
    for name in sorted(os.listdir(LADDER)):
        if not name.endswith('_ref.png'):
            continue
        stem, size = name.split('_')[0], name.split('_')[1]
        if not size.isdigit() or int(size) not in SIZES:
            continue
        sr = oracle(os.path.join(LADDER, name))
        sd = oracle(os.path.join(LADDER, name.replace('_ref', '_dist')))
        g.write(f'{stem}\t{size}\t{sd:.9g}\t{sr:.9g}\n')
        g.flush()
        print('musiq', stem, size, sd, sr, flush=True)
print('done')
