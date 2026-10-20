#!/usr/bin/env python3
"""Original Google MUSIQ JAX model and preprocessing on resident RGB8."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path);p.add_argument('metric',choices=['musiq'])
    p.add_argument('width',type=int);p.add_argument('height',type=int)
    p.add_argument('reference',type=Path);p.add_argument('distorted',type=Path)
    p.add_argument('threads',type=int);p.add_argument('samples',type=int)
    p.add_argument('--setup-only',action='store_true')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3';os.environ['JAX_PLATFORM_NAME']='cpu'
    eigen='true' if a.threads>1 else 'false'
    os.environ['XLA_FLAGS']=f'--xla_cpu_multi_thread_eigen={eigen} intra_op_parallelism_threads={a.threads}'
    os.environ['TF_NUM_INTRAOP_THREADS']=str(a.threads);os.environ['TF_NUM_INTEROP_THREADS']='1'
    sys.path.insert(0,cfg['upstream'])
    import numpy as np
    import jax
    import jax.numpy as jnp
    import tensorflow.compat.v2 as tf
    import musiq.model.multiscale_transformer as model_mod
    import musiq.model.preprocessing as preprocessing
    from musiq.run_predict_image import get_params_and_config
    tf.config.threading.set_intra_op_parallelism_threads(a.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    model_config,pp_config,params=get_params_and_config(cfg['checkpoint'])
    model=model_mod.Model.partial(num_classes=1,train=False,**model_config)
    # Both arms begin at RGB8. Replace file decoding only; normalization,
    # TensorFlow resampling, patches and the original model body stay unchanged.
    preprocessing.decode_image=lambda image:tf.convert_to_tensor(image,dtype=tf.uint8)
    preprocess=preprocessing.get_preprocess_fn(**pp_config)
    if a.setup_only:
        print(json.dumps(dict(metric=a.metric,setup='ready',jax=jax.__version__,tensorflow=tf.__version__)))
        return
    r,d=[np.fromfile(path,np.uint8).reshape(a.height,a.width,3) for path in [a.reference,a.distorted]]
    def single(rgb):
        data=preprocess(dict(image=rgb))
        image=np.asarray(tf.expand_dims(data['image'],axis=0).numpy())
        logits=model.call(params,jnp.asarray(image))
        return float(np.asarray(logits[0,0]))
    for i in range(a.samples+1):
        start=time.perf_counter_ns();value=single(d);reference=single(r);elapsed=time.perf_counter_ns()-start
        print(('COLD' if i==0 else 'SAMPLE')+f' {elapsed} {value:.17g}',flush=True)
        print(f'PAIR_REFERENCE {reference:.17g}',flush=True)


if __name__=='__main__':main()
