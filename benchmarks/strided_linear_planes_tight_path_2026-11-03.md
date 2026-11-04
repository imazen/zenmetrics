# Tight-path cost of strided linear planes and VMAF frames (2026-11-03)

Snapshot, not a live claim. Host r7900x (Ryzen 9 7900X, Zen 4), **not measured on
Zen 5**. No `-C target-cpu=native`, `RAYON_NUM_THREADS=1`, release profile. Base
`d46ec235`, change `ef9f086f`. RSS not measured.

Harness: one process per side, interleaved, 21 rounds of 21 calls with the first
call of each process dropped (n = 420 per side). Input is 320x240 synthetic, tight
(stride equals width): `cvvdp::Cvvdp::score_from_linear_planes` and
`vmaf::score_v1_420` at `Standard1080p`, 8 bits. The harness was a throwaway example
and is not committed; the calls above are all it does.

## Default build

| call | median ms base | median ms change | ratio | instructions base | instructions change |
|---|---|---|---|---|---|
| vmaf, three repetitions | 2.121 / 2.093 / 2.104 | 2.412 / 2.389 / 2.397 | 1.137 / 1.141 / 1.139 | 17,351,502,607 (300 calls) | 17,351,717,185 |
| cvvdp, three repetitions | 1.949 / 1.947 / 1.943 | 1.990 / 1.995 / 1.996 | 1.022 / 1.024 / 1.027 | 61,264,405 (1 call) | 61,263,963 |

Instructions are flat: vmaf +0.001%, cvvdp -0.001%. vmaf's extra cycles (3.39 G to
3.83-3.89 G over 300 calls) are in `vmaf::speed::vif_filter1d`, 34% to 42% of samples
in `perf record`. This change does not touch that kernel, and branch misses stay near
1 M. That points at code layout, so the next table controls for it.

## Both sides with forced code alignment

`RUSTFLAGS="-C llvm-args=-align-all-functions=6 -C llvm-args=-align-all-nofallthru-blocks=5"`

| call | median ms base | median ms change | ratio | cycles base | cycles change | instructions base | instructions change |
|---|---|---|---|---|---|---|---|
| vmaf, two repetitions | 2.329 / 2.316 | 2.349 / 2.321 | 1.009 / 1.002 | 2,520,880,387 | 2,534,249,814 | 11,569,262,036 | 11,569,372,434 |
| cvvdp, two repetitions | 1.936 / 1.931 | 1.928 / 1.933 | 0.996 / 1.001 | 2,155,444,929 | 2,141,898,138 | 10,105,471,690 | 10,105,482,686 |

Counters here are over 200 calls. With layout pinned the two sides agree within 1%,
so the default-build gap is a layout effect and not the work this change adds. The
forced-alignment base is itself slower than the default base (2.32 ms against
2.10 ms), which is the same sensitivity seen from the other side: `vif_filter1d`'s
wall time on this CPU moves about 14% with its placement alone.

## Not measured

GPU paths: this host has no NVIDIA driver, and wgpu runs on llvmpipe. The tight path
through them adds a length check and a slice per plane.
