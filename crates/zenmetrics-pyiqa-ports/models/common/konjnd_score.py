"""Score KonJND-1k pairs with pyiqa metrics for one JND per column (one row per pair).

usage: konjnd_score.py PAIRS_TSV OUT_TSV START END METRIC [METRIC...]

PAIRS_TSV: `ref_path dist_path src codec level` (KonJND-1k at each image's mean PJND,
round-half-up; zenmetrics benchmarks/nn_konjnd_scale_2026-10-15.md). Appends rows
`src codec level <metric>... <metric>_same...` for pairs [START, END): metric(dist, ref) and
metric(ref, ref) (WaDIQaM's identical-pair score is image-dependent). Rows already in OUT_TSV
are skipped, so it can run in chunks. With NOSAME=1 the identical-pair scores are not computed
(written as nan): for metrics whose identical-pair score is exactly 0, like DMM.
"""
import csv, os, sys
import torch, pyiqa

pairs_tsv, out, start, end, metrics = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:]
torch.set_grad_enabled(False)
rows = list(csv.DictReader(open(pairs_tsv), delimiter='\t'))[start:end]
done = set()
if os.path.exists(out):
    done = {r['src'] for r in csv.DictReader(open(out), delimiter='\t')}
nets = {m: pyiqa.create_metric(m, device='cpu', as_loss=False) for m in metrics}
new = not os.path.exists(out)
with open(out, 'a') as f:
    if new:
        f.write('\t'.join(['src', 'codec', 'level'] + metrics + [m + '_same' for m in metrics]) + '\n')
    for r in rows:
        if r['src'] in done:
            continue
        s = [float(nets[m](r['dist_path'], r['ref_path']).flatten()[0]) for m in metrics]
        same = [float('nan') if os.environ.get('NOSAME') == '1' else
                float(nets[m](r['ref_path'], r['ref_path']).flatten()[0]) for m in metrics]
        f.write('\t'.join([r['src'], r['codec'], r['level']] + [repr(v) for v in s + same]) + '\n')
        f.flush()
print('done', start, end, flush=True)
