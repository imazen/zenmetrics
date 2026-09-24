# Validation against official HDR-VDP-2.2.2

The reference implementation is the official `hdrvdp-2.2.2` MATLAB release
(SourceForge: `hdrvdp/files/hdrvdp/2.2.2/hdrvdp-2.2.2.zip`), run under
**GNU Octave 11.1** with the `image` package. Goldens were generated on
2026-09-24; the corpus lives in `validation/goldens/` and the generator in
`validation/gen_goldens.m`.

## Corpus

24 cases, all `'luminance'` encoding (absolute cd/m²): a deterministic
two-component sinusoid + uniform-noise texture (~1–400 cd/m²), and the six
distortions `identical`, `noise` (σ=4), `blur` (7×7 gauss σ=1.2), `dark`
(×0.55), `bright` (×1.5+8), `contrast` (×0.6 about the mean), at
{128×128, 192×160} × `pixels_per_degree` {30, 60}.

## Measured parity (worst case over all 24 cases)

| output | tolerance achieved |
|---|---:|
| `P_det` | 5.3e-14 absolute |
| `C_max` | 3.9e-12 relative |
| `P_map` | 3.9e-11 absolute |
| `res.Q` (`HdrVdpResult::q`) | 9.9e-7 absolute |
| per-plane `D` bands | ~1e-16 max abs |

The `res.Q` delta is dominated by the border pixels of the reconstructed
`S_map` (the documented `reflect1`-vs-`EXPAND` synthesis-edge gap, which
shifts `msre` slightly in the lowest-weight planes); the visibility maps
themselves agree to ~4e-11.

## Running the check

```sh
cargo run --release -p hdrvdp --example golden_check -- \
    crates/hdrvdp/validation/goldens
```

`--diag` additionally compares the `noise_128x128_p30` `D_bands` pyramid and
`S_map` stage-by-stage against `diag_*` dumps and reconstructs official
`res.Q` from the crate's per-plane `quality_terms`.

`tests/golden_2_2_2.rs` runs the same comparison in the test suite with the
tolerances above.

## Regenerating the goldens

1. `unzip hdrvdp-2.2.2.zip` (the official source is *not* vendored here —
   its licence does not permit redistribution inside this repo).
2. `pkg install image` in Octave (upstream calls `padarray`).
3. Apply the patches below — all are *Octave-compatibility or correctness*
   fixes to the upstream sources; none changes the intended math.
4. `HDRVDP_SRC=/path/to/hdrvdp-2.2.2 HDRVDP_GOLDENS=out \
      octave --no-gui validation/gen_goldens.m`

### Required patches to upstream 2.2.2 for Octave

- **`fast_conv_fft.m`, `fast_gauss.m`**: drop the `'symmetric'` flag from
  `ifft2` calls (MATLAB-only syntax; `real(ifft2(...))` is unchanged).
- **`reconSpyr.m`, `reconSpyrLevs.m` — the `is_mex` bug (load-bearing)**:
  upstream detects the compiled `upConv` via
  `strcmp(finfo.file((end-2):end), '.m')`, but the last **3** characters of
  a `.m` path can never equal the 2-char `'.m'` — so `is_mex` is *always*
  true. Under real MATLAB+MEX that is harmless (the in-place `upConv`
  really writes into `res`), but under Octave the `.m` fallback runs
  instead, the "in-place" call **silently discards its result**, and the
  residual high-pass band plus all 12 oriented bands are dropped from
  `S_map`. The effect is dramatic: σ=4 noise on a 128² field goes from
  `C_max = 211.6` (MATLAB-equivalent) to `C_max = 0.0002` (broken Octave).
  Fix: force `is_mex = false` so the `.m` accumulation path
  (`res = upConv(...)`) runs. Quality (`res.Q`) is unaffected either way —
  it is accumulated in the band loop and never calls `reconSpyr`.
- **`hdrvdp.m` instrumentation (required by `gen_goldens.m`)**: record
  `{msre, w_f, band, ori}` per plane into `res.qmsres` and dump
  `D_bands.pyr`/`pind`/`S_map` onto `res.dbg_*` — pure telemetry appended
  to `res`, no numeric change. Without it the generator's `*_planes.tsv`
  and `diag_*` artifact writes fail on the missing fields.

## What this does and does not prove

- **Does**: the full pathway → pyramid → masking → pooling pipeline is a
  numerically faithful port of official 2.2.2 on the cases above.
  `tests/bit_lock.rs` continues to fence the *optimized* code against a
  frozen verbatim copy of this port.
- **Does not**: luminance-encoding coverage only; display encodings
  (`sRGB-display`, `rgb-bt.709`, `XYZ`, `luma-display`) are exercised by
  unit tests but not by these goldens. And UPIQ SROCC vs the published
  0.812 (chunk 4) is a separate, still-open measurement — golden parity is
  implementation parity, not subjective-score validation.
