"""pyiqa goldens for a pairs TSV: `distorted <col>` from `pyiqa.create_metric(METRIC)` defaults.

usage: goldens.py PAIRS_TSV OUT_TSV METRIC COLUMN

PAIRS_TSV has `ref_path dist_path distorted`; scores are `metric(dist_path, ref_path)` on CPU
fp32 (pyiqa's own image loading), the way an AIC2026-style scorer calls pyiqa. Run with the
oracle venv and TORCH_HOME at its weight cache. Rows already in OUT_TSV are kept (resumable).
"""
import csv, os, sys
import torch, pyiqa

pairs_tsv, out, name, col = sys.argv[1:5]
torch.set_grad_enabled(False)
rows = list(csv.DictReader(open(pairs_tsv), delimiter='\t'))
done = set()
if os.path.exists(out):
    done = {r['distorted'] for r in csv.DictReader(open(out), delimiter='\t')}
metric = pyiqa.create_metric(name, device='cpu', as_loss=False)
# PIQA_FEATURE_CHUNK=N: run `net.compute_features` (PieAPP) N patches at a time. pyiqa runs every
# patch of an image in one batch, which at 2048x1536 (4,070 patches) needs over 14 GiB; the
# per-patch network is independent per patch, so only the batch size changes.
chunk = int(os.environ.get('PIQA_FEATURE_CHUNK', '0'))
if chunk and hasattr(metric.net, 'compute_features'):
    inner = metric.net.compute_features

    def chunked(x):
        outs = [inner(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)]
        return tuple(torch.cat(parts, 0) for parts in zip(*outs))

    metric.net.compute_features = chunked
new = not os.path.exists(out)
with open(out, 'a') as f:
    if new:
        f.write(f'distorted\t{col}\n')
    for r in rows:
        if r['distorted'] in done:
            continue
        try:
            s = float(metric(r['dist_path'], r['ref_path']).flatten()[0])
        except Exception as e:  # pyiqa refuses the pair: record it, the port must refuse too
            print('REFUSED', r['distorted'], str(e).splitlines()[0], flush=True)
            s = float('nan')
        f.write(f"{r['distorted']}\t{s!r}\n")
        f.flush()
        print(r['distorted'], s, flush=True)
