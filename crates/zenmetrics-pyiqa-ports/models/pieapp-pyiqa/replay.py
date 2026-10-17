"""Check that zenmetrics-nn's exported pieapp.onnx, fed pyiqa's patch grid, reproduces pyiqa `pieapp`.

usage: replay.py PIEAPP_ONNX PAIRS_TSV GOLDENS_TSV [N]

pyiqa 0.1.16 PieAPP (archs/pieapp_arch.py): 64x64 patches at stride 27 (unfold, no padding), so
origins 0, 27, ... while <= d - 64; per patch (error, weight + 1e-6) from the graph;
score = sum(w * e) / sum(w). The graph takes RGB in [0,1] and does the BGR x255 itself.
Prints per pair pyiqa golden, replay, |d|, and the worst |d| over the first N pairs (default all).
"""
import csv, sys
import numpy as np, onnxruntime as ort
from PIL import Image

onnx_path, pairs, goldens = sys.argv[1:4]
n_max = int(sys.argv[4]) if len(sys.argv) > 4 else None
g = {r['distorted']: float(list(r.values())[1]) for r in csv.DictReader(open(goldens), delimiter='\t')}
s = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])


def planes(p):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8).astype(np.float32) / np.float32(255)
    return a.transpose(2, 0, 1)


worst = 0.0
for k, r in enumerate(csv.DictReader(open(pairs), delimiter='\t')):
    if n_max is not None and k >= n_max:
        break
    xr, xd = planes(r['ref_path']), planes(r['dist_path'])
    h, w = xr.shape[1:]
    pos = [(y, x) for y in range(0, h - 63, 27) for x in range(0, w - 63, 27)]
    num = den = 0.0
    for i in range(0, len(pos), 64):
        chunk = pos[i:i + 64]
        pr = np.stack([xr[:, y:y + 64, x:x + 64] for y, x in chunk])
        pd = np.stack([xd[:, y:y + 64, x:x + 64] for y, x in chunk])
        out = s.run(None, {'reference': pr, 'distorted': pd})[0].astype(np.float64)
        num += float((out[:, 0] * out[:, 1]).sum())
        den += float(out[:, 1].sum())
    v = num / den
    d = abs(v - g[r['distorted']])
    worst = max(worst, d)
    print(r['distorted'], len(pos), g[r['distorted']], v, d, flush=True)
print('worst |replay - pyiqa|:', worst)
