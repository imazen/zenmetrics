"""LPIPS-vgg via ONNX Runtime (CUDA EP, TF32 off by default) on the exported trunk;
head in numpy float64-free float32 like upstream. Parity vs goldens + timing."""
import sys, time, statistics, os, ctypes, glob
import numpy as np
# make the pip-installed CUDA/cuDNN libs visible before onnxruntime loads them
import torch  # noqa: F401  (loads the nvidia libs torch ships with)
import onnxruntime as ort
from PIL import Image

model_dir, ladder, provider = sys.argv[1], sys.argv[2], sys.argv[3]
sizes = [int(s) for s in sys.argv[4].split(',')]
reps = int(sys.argv[5])
opts = {}
prov = [('CUDAExecutionProvider', {'use_tf32': '0', 'cudnn_conv_algo_search': 'EXHAUSTIVE'})] if provider == 'cuda' else ['CPUExecutionProvider']
so = ort.SessionOptions()
sess = ort.InferenceSession(os.path.join(model_dir, 'lpips_vgg_trunk.onnx'), so, providers=prov)
lin_all = np.fromfile(os.path.join(model_dir, 'lpips_vgg_lin.f32'), dtype='<f4')
lins, at = [], 0
for c in (64, 128, 256, 512, 512):
    lins.append(lin_all[at:at + c].reshape(1, c, 1, 1)); at += c

def load(p):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / np.float32(255.0)
    return (a.transpose(2, 0, 1)[None] * np.float32(2) - np.float32(1)).astype(np.float32)

def score(r, d):
    tr = sess.run(None, {'x': r}); td = sess.run(None, {'x': d})
    tot = 0.0
    for a, b, l in zip(tr, td, lins):
        na = a / (np.sqrt((a * a).sum(1, keepdims=True)) + np.float32(1e-10))
        nb = b / (np.sqrt((b * b).sum(1, keepdims=True)) + np.float32(1e-10))
        tot += float(((na - nb) ** 2 * l).sum(1).mean())
    return tot

def trunk_only(r, d):
    sess.run(None, {'x': r}); sess.run(None, {'x': d})

gold = [l.split('\t') for l in open(os.path.join(model_dir, 'goldens.tsv')).read().splitlines()[1:]]
for stem, n, want, *_ in gold:
    n = int(n)
    if n not in sizes:
        continue
    r = load(os.path.join(ladder, f'{stem}_{n}_ref.png')); d = load(os.path.join(ladder, f'{stem}_{n}_dist.png'))
    s = score(r, d)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); trunk_only(r, d); ts.append((time.perf_counter() - t) * 1e3)
    med = statistics.median(ts) if ts else float('nan')
    print(f'ort-{provider}\t{stem}\t{n}\tgot {s:.9f}\twant {float(want):.9f}\t|d| {abs(s - float(want)):.3e}\ttrunk x2 median {med:.3f} ms', flush=True)
