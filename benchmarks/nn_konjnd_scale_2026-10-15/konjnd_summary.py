#!/usr/bin/env python3
import csv, os
import numpy as np
OUT = os.path.expanduser("~/tmp/konjnd-nn")
rows = list(csv.DictReader(open(os.path.join(OUT, "pairs_scores.tsv")), delimiter="\t"))
metrics = ["lpips_alex", "lpips_vgg", "lpips_squeeze", "dists", "stlpips_alex", "stlpips_vgg", "deepdc"]
lines = ["metric\tcodec\tn\tp25\tmedian\tp75\tmean\tp75_over_p25"]
for m in metrics:
    for codec in ("JPEG", "BPG", "pooled"):
        v = np.array([float(r[m]) for r in rows if codec == "pooled" or r["codec"] == codec])
        p25, med, p75 = np.percentile(v, [25, 50, 75])
        lines.append(f"{m}\t{codec}\t{len(v)}\t{p25:.6f}\t{med:.6f}\t{p75:.6f}\t{v.mean():.6f}\t{p75/p25:.3f}")
open(os.path.join(OUT, "jnd_scale.tsv"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
# extra: Spearman of metric vs PJND level within codec (how much the metric value at JND tracks the level)
from scipy.stats import spearmanr
for m in metrics:
    for codec in ("JPEG", "BPG"):
        sel = [r for r in rows if r["codec"] == codec]
        rho = spearmanr([float(r["level"]) for r in sel], [float(r[m]) for r in sel]).statistic
        print(f"spearman(level,{m}) {codec}: {rho:.3f}")
