import sys, time, statistics, os
import numpy as np, torch  # noqa: F401
import onnxruntime as ort
from PIL import Image
model_dir, ladder = sys.argv[1], sys.argv[2]
sizes = [int(s) for s in sys.argv[3].split(',')]; reps = int(sys.argv[4])
sess = ort.InferenceSession(os.path.join(model_dir, 'lpips_vgg_full.onnx'),
    providers=[('CUDAExecutionProvider', {'use_tf32': '0', 'cudnn_conv_algo_search': 'EXHAUSTIVE'})])
def load(p):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / np.float32(255.0)
    return (a.transpose(2, 0, 1)[None] * np.float32(2) - np.float32(1)).astype(np.float32)
gold = [l.split('\t') for l in open(os.path.join(model_dir, 'goldens.tsv')).read().splitlines()[1:]]
for stem, n, want, *_ in gold:
    n = int(n)
    if n not in sizes: continue
    r = load(os.path.join(ladder, f'{stem}_{n}_ref.png')); d = load(os.path.join(ladder, f'{stem}_{n}_dist.png'))
    s = float(sess.run(None, {'r': r, 'd': d})[0])
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); sess.run(None, {'r': r, 'd': d}); ts.append((time.perf_counter() - t) * 1e3)
    print(f'ort-cuda-full\t{stem}\t{n}\tgot {s:.9f}\twant {float(want):.9f}\t|d| {abs(s-float(want)):.3e}\tcompute median {statistics.median(ts):.3f} ms', flush=True)
