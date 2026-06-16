# Hetzner cost/result — jpeg-encode throughput (256x256 q80, all cores, 5s)
# 2026-05-30, fsn1. Workload: zenfleet-worker bench_encode (image-crate JPEG). MEASURED, not extrapolated.
# cax31/cax41 (8/16-core ARM): UNAVAILABLE (Hetzner Ampere out of stock fsn1/nbg1/hel1).

type          arch            cores   EUR/hr    enc/s  enc/s/core  enc/s per EUR/hr  Mencodes/EUR
cax21         ARM                 4   0.0152     2958         740           194,612         700.6
cpx32         x86-shared          4   0.0264     1969         492            74,572         268.5
cpx42         x86-shared          8   0.0481     4081         510            84,844         305.4
ccx23         x86-dedicated       4   0.0593     1619         405            27,297          98.3

Ranking by cost-efficiency (enc/s per EUR/hr, higher=better):
  cax21  ARM             194,612   (1.00x vs best)
  cpx42  x86-shared       84,844   (2.29x vs best)
  cpx32  x86-shared       74,572   (2.61x vs best)
  ccx23  x86-dedicated    27,297   (7.13x vs best)
