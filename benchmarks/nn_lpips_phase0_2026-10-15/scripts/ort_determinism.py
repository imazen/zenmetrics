import sys, time, statistics, os, numpy as np, torch  # noqa: F401
import onnxruntime as ort
from PIL import Image
md, ladder = sys.argv[1], sys.argv[2]
def load(p):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / np.float32(255.0)
    return (a.transpose(2, 0, 1)[None] * np.float32(2) - np.float32(1)).astype(np.float32)
for search in ('EXHAUSTIVE', 'HEURISTIC', 'DEFAULT'):
    for det in (False, True):
        so = ort.SessionOptions(); so.use_deterministic_compute = det
        sess = ort.InferenceSession(os.path.join(md, 'lpips_vgg_full.onnx'), so,
            providers=[('CUDAExecutionProvider', {'use_tf32': '0', 'cudnn_conv_algo_search': search})])
        for n in (256, 1024):
            r = load(f'{ladder}/1415_{n}_ref.png'); d = load(f'{ladder}/1415_{n}_dist.png')
            vals, ts = [], []
            for i in range(8):
                t = time.perf_counter(); v = float(sess.run(None, {'r': r, 'd': d})[0]); ts.append((time.perf_counter() - t) * 1e3)
                vals.append(np.float32(v).view(np.uint32))
            distinct = sorted(set(int(x) for x in vals))
            print(f'{search}\tdet={det}\t{n}\tcalls 1..8 distinct bit patterns {len(distinct)}\tfirst==rest {all(v == vals[1] for v in vals[1:])}\tfirst {vals[0]} later {vals[-1]}\tmedian(2..8) {statistics.median(ts[1:]):.3f} ms', flush=True)
