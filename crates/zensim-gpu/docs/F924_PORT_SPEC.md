# Porting the 924-feature `Folded720Append` regime to zensim-gpu

Verified against the CPU oracle on 2026-07-29 (Apple Silicon, wgpu/Metal
backend). Everything here was established by **measurement or by reading the
CPU source**, not inferred from constants — three of the findings contradict
what the headers say, and each would have cost real time.

Gate: `cargo run --release -p zensim-gpu --no-default-features \
  --features "wgpu,cubecl-types" --example f924_parity`

---

## 1. Target layout

CPU `Zensim::compute_folded720_append_features` emits 924 `f64` at 4 scales:

| range | block | on a 256x256 synthetic pair |
|---|---|---|
| `[0..156)` | v1-basic (13/ch/scale) | 136/156 nonzero |
| `[156..372)` | deprecated | **all zero** (verified) |
| `[372..720)` | v2-348 (29/ch/scale) | 347/348 nonzero |
| `[720..924)` | append-204 (17/ch/scale) | 161/204 nonzero |

`[156..372)` is v1's peak/masked/IW material, deliberately zeroed by the
folded layout. It is not computed and discarded — it is emitted as `0.0`.

## 2. The GPU's emitted array is BLOCK-structured, not interleaved

`lib.rs` defines `FEATURES_PER_CHANNEL = 19` (13 basic + 6 peak) and
`FEATURES_PER_SCALE = 19 * 3`, which reads as `[scale][channel][19]`. **The
emitted array is not in that order.** `compute_features` returns 228 laid out
as `[0..156)` basic then `[156..228)` peaks.

Proven at the boundary: indices 150..155 agree with the CPU folded block to
f32 precision, and index 156 is exactly where the CPU goes to `0.0` while the
GPU continues with nonzero peaks. An interleaved reading mismatches from index
13 onward; it does not.

Cost of getting this wrong: applying the 13-vs-19 "stride correction" reports
`max |diff| = 6.9e-1`, which looks exactly like a genuine parity failure and
would send a session off to fix a block that is already correct. The
index-aligned reading gives `1.38e-4` (f32 GPU vs f64 CPU).

**Consequence: `[0..156)` is already done.** The port is blocks 3 and 4.

## 3. v2 SSIM is a DIFFERENT per-pixel formula — the v1 map cannot be reused

The GPU's fused kernel already computes a per-pixel `sd` and accumulates
`Σd`, `Σd²`, `Σd⁴`. It is tempting to pool those into the v2 moments. **Do
not** — they are not the same quantity.

GPU / v1 (`kernels/fused.rs`), SSIMULACRA2-style, **no C1**:

```text
num_m   = 1 - (mu1 - mu2)^2
num_s   = 2*(s12 - mu1*mu2) + C2
denom_s = (ssq - mu1^2 - mu2^2) + C2
sd      = max(0, 1 - num_m*num_s/denom_s)
```

CPU / v2 (`feature_v2.rs::ssim_d_local`), standard SSIM, **with C1**:

```text
a     = 2*mu1*mu2 + C1_V2
b     = mu1^2 + mu2^2 + C1_V2
c     = 2*(s12 - mu1*mu2) + C2_V2
d     = ssq - mu1^2 - mu2^2 + C2_V2
d_v2  = max(0, 1 - (a*c)/(b*d))
```

The v2 block needs its own per-pixel evaluation from the same
`mu1/mu2/s12/ssq` planes the existing fused kernel already produces — the
blur work is shared, the SSIM evaluation is not.

## 4. A missing accumulator

The v2 moments finalize from **raw power sums** (`feature_v2.rs`, the
raw-to-central conversion):

```text
mean = Σd/n ; raw2 = Σd²/n ; raw3 = Σd³/n ; raw4 = Σd⁴/n
m2   = max(0, raw2 - mean²)
m4   = max(0, raw4 - 4*mean*raw3 + 6*mean²*raw2 - 3*mean⁴)
SSIM_MEAN = clamp(mean, 0, 2) ; SSIM_DEV2 = clamp(√m2, 0, 2)
SSIM_DEV4 = clamp(m4^0.25, 0, 2)
```

The GPU accumulates `Σd, Σd², Σd⁴, Σd⁸` — it has **no `Σd³`**. The v2 kernel
must add it. (Note the CPU deliberately uses raw sums rather than Terriberry
online moments, and documents the two as equal within 5e-4 relative — so the
GPU should match the raw-sum form, not re-derive an online one.)

## 5. v2 per-channel index map (29 slots)

```text
 0 SSIM_MEAN      1 SSIM_DEV2     2 SSIM_DEV4
 3 ART            4 DET           5 MSE
 6 HF_GAIN        7 HF_LOSS       8 HF_MAG_LOSS
 9 SSIM_SOFT_PEAK 10 ART_SOFT_PEAK 11 DET_SOFT_PEAK
12 MASKED_SSIM   13 MASKED_ART   14 MASKED_DET   15 MASKED_MSE
16 IW_SSIM       17 IW_ART       18 IW_DET       19 IW_MSE
20 PJND_TRANSDUCER 21 PJND_FRAGILITY
22 GMS           23 PJND_TRANSDUCER_LOW_K  24 PJND_TRANSDUCER_HIGH_K
25 BLOCKINESS    26 RINGING      27 BANDING      28 EDGE_WIDTH_CHANGE
```

Slots 9-19 use weighted pooling (`Σw·v/Σw`); 20-24 are transducers; 25-28 are
the phase-2 structural detectors. Slot 28 (`EDGE_WIDTH_CHANGE`) is the one
scale-level rather than per-pixel value in the set.

## 6. Suggested staging

Each stage gated against the harness before the next begins — a 552-feature
diff debugged at the end is not a debuggable thing.

1. v2 basic-9 (slots 0-8) — needs the new per-pixel v2 SSIM + `Σd³`.
2. v2 soft-peak 3 (9-11) — weighted pooling on the same map.
3. masked-4 + IW-4 (12-19) — same pooling, two weight polarities.
4. PJND 2 + bank 2 (20, 21, 23, 24) — transducers.
5. GMS (22) + structural 4 (25-28).
6. append-204.

## 7. Acceptance bar

The already-correct `[0..156)` block agrees at `1.38e-4` max absolute (f32 GPU
vs f64 CPU). That is the natural bar for the new blocks. Note the CPU itself
documents 5e-4 relative between its own SIMD and scalar paths, so demanding
bit-equality against f64 would be stricter than the CPU is with itself.
