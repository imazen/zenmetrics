#!/usr/bin/env python3
"""CROSS-CODEC ACHIEVED-QUALITY COVERAGE GATE — the quality-axis analogue of
check_mandatory_coverage (which gates modes/knobs, not the rate axis).

WHY: the sweep samples a generic `q` grid that each codec resolves to its own native param
(quality_to_quantizer / resolve_distance_for_quality), so equal `q` != equal achieved zensim.
A picker/oracle works in ACHIEVED-zensim space; if codec X has no samples in a zensim band that
codec Y covers, the oracle (min-bytes at that zensim) silently excludes X -> biased labels +
'valid-looking' eval on invalid ground truth. This gate measures, per lossy codec, the fraction
of variants reaching each achieved-zensim band, and FAILS on a high-band coverage asymmetry."""
import sys, collections, pandas as pd
BASE = '/mnt/v/output/canonical-picker-2026-06-27'
LOSSY = [('jpeg', 'zenjpeg_lossy'), ('webp', 'zenwebp_lossy'), ('jxl', 'zenjxl_lossy'), ('avif', 'zenavif_lossy')]
BANDS = [70, 80, 85, 90, 93, 95, 97]
HIGH_BAND_FLOOR = 90   # bands >= this are picker-critical (near-lossless region)
MAX_SPREAD = 0.40      # if (max-min) coverage across codecs in a band exceeds this -> biased oracle

ceil = {}  # codec -> {variant: max achieved zensim}
for fam, d in LOSSY:
    df = pd.read_parquet(f'{BASE}/{d}/train.parquet', columns=['variant_name', 'score_zensim']).dropna()
    ceil[fam] = df.groupby('variant_name')['score_zensim'].max()
allv = set.intersection(*[set(c.index) for c in ceil.values()])
print(f"cross-codec coverage over {len(allv)} shared variants\n")
hdr = "band   " + "".join(f"{f:>8}" for f, _ in LOSSY) + "   spread"
print(hdr); print("-" * len(hdr))
fail = []
for b in BANDS:
    cov = {f: (ceil[f].loc[list(allv)] >= b).mean() for f, _ in LOSSY}
    spread = max(cov.values()) - min(cov.values())
    flag = " <== ASYMMETRY" if (b >= HIGH_BAND_FLOOR and spread > MAX_SPREAD) else ""
    print(f">=zq{b} " + "".join(f"{cov[f]*100:7.0f}%" for f, _ in LOSSY) + f"   {spread*100:5.0f}%{flag}")
    if flag:
        worst = min(cov, key=cov.get); best = max(cov, key=cov.get)
        fail.append(f"  zq>={b}: {worst} covers {cov[worst]*100:.0f}% but {best} covers {cov[best]*100:.0f}% "
                    f"-> oracle/labels biased toward {best} above ~zq{b}")
print()
if fail:
    print("GATE FAILED — under-sampled codec(s) in picker-critical band(s):")
    print("\n".join(fail))
    print("\nFix: re-sweep the under-covered codec(s) to a higher achieved-quality range (target zensim,\n"
          "not generic q), OR mask the affected zensim band from training/oracle + flag it. Equal-q is\n"
          "NOT equal-quality across codecs; sample to a common ACHIEVED-zensim grid.")
    sys.exit(1)
print("GATE PASSED — comparable high-band coverage across codecs.")
