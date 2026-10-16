"""Bar A / Bar B timing for upstream LPIPS and DISTS (PyTorch reference).

e2e:     per pair, PNG decode (PIL) + tensor build + upload + forward + .item()
compute: tensors already on the device; forward + sync only
Writes one TSV row per (metric, dtype, mode, size).
"""
import argparse, os, platform, socket, statistics, sys, time
import numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--ladder', required=True)
ap.add_argument('--device', default='cpu')
ap.add_argument('--metrics', default='lpips-vgg,lpips-alex,dists')
ap.add_argument('--dtypes', default='fp32')
ap.add_argument('--sizes', default='64,128,256,512,1024,2048,4096')
ap.add_argument('--image', default='1415')
ap.add_argument('--budget-s', type=float, default=6.0, help='time budget per cell')
ap.add_argument('--out', required=True)
a = ap.parse_args()
dev = torch.device(a.device)

def sync():
    if dev.type == 'cuda':
        torch.cuda.synchronize()

def load(path):
    return np.asarray(Image.open(path).convert('RGB'))

def to_t(arr, metric, dtype):
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev).float() / 255.0
    if metric.startswith('lpips'):
        t = t * 2 - 1
    return t.to(dtype)

def build(metric):
    if metric.startswith('lpips'):
        import lpips
        return lpips.LPIPS(net=metric.split('-')[1], verbose=False)
    import DISTS_pytorch
    return DISTS_pytorch.DISTS()

def forward(m, metric, x, y):
    if metric.startswith('lpips'):
        return m(x, y)
    return m(x, y, batch_average=True)

def timed(fn, budget):
    fn(); sync()  # warmup
    ts = []
    t_end = time.perf_counter() + budget
    while len(ts) < 3 or (time.perf_counter() < t_end and len(ts) < 50):
        t0 = time.perf_counter(); r = fn(); sync(); ts.append(time.perf_counter() - t0)
    return ts, r

new = not os.path.exists(a.out)
out = open(a.out, 'a')
if new:
    out.write('host\tdevice\tdevice_name\ttorch\tthreads\ttf32_conv\tmetric\tdtype\tmode\tsize\tn\tmedian_ms\tmin_ms\tscore\tnote\n')
devname = torch.cuda.get_device_name(0) if dev.type == 'cuda' else platform.processor() or platform.machine()
for metric in a.metrics.split(','):
    base = build(metric).to(dev).eval()
    for dt in a.dtypes.split(','):
        dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'fp32strict': torch.float32}[dt]
        torch.backends.cudnn.allow_tf32 = dt != 'fp32strict'
        torch.backends.cuda.matmul.allow_tf32 = False
        m = base.to(dtype)
        for n in map(int, a.sizes.split(',')):
            rp = os.path.join(a.ladder, f'{a.image}_{n}_ref.png'); dp = rp.replace('_ref', '_dist')
            if not os.path.exists(rp):
                continue
            for mode in ('compute', 'e2e'):
                note = ''
                try:
                    with torch.inference_mode():
                        if mode == 'compute':
                            x = to_t(load(rp), metric, dtype); y = to_t(load(dp), metric, dtype)
                            fn = lambda: forward(m, metric, x, y)
                        else:
                            fn = lambda: forward(m, metric, to_t(load(rp), metric, dtype), to_t(load(dp), metric, dtype)).item()
                        ts, r = timed(fn, a.budget_s)
                    score = float(r) if mode == 'e2e' else float(r.float().item())
                    med, mn = statistics.median(ts) * 1e3, min(ts) * 1e3
                except torch.OutOfMemoryError as e:
                    ts, score, med, mn, note = [], float('nan'), float('nan'), float('nan'), 'OOM'
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    ts, score, med, mn, note = [], float('nan'), float('nan'), float('nan'), ('OOM' if 'memory' in str(e).lower() else 'ERR:' + str(e)[:60])
                row = [socket.gethostname(), a.device, devname, torch.__version__, torch.get_num_threads(), torch.backends.cudnn.allow_tf32,
                       metric, dt, mode, n, len(ts), f'{med:.3f}', f'{mn:.3f}', f'{score:.7f}', note]
                out.write('\t'.join(map(str, row)) + '\n'); out.flush()
                print('\t'.join(map(str, row[6:])), flush=True)
                if dev.type == 'cuda':
                    torch.cuda.empty_cache()
