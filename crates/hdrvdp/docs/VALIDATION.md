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

Current pipeline — `f32` planes/intermediates with `f64` reductions
(fleet-throughput re-baseline, 2026-09-25):

| output | measured delta | test tolerance |
|---|---:|---:|
| `P_det` | 1.4e-5 absolute | 1e-4 |
| `C_max` | 1.0e-4 relative | 5e-4 |
| `P_map` | 1.7e-2 absolute | 5e-2 |
| `res.Q` (`HdrVdpResult::q`) | 7.8e-4 absolute | 5e-3 |

The pre-conversion `f64` pipeline (superseded; kept for provenance)
achieved `P_det` 5.3e-14, `C_max` 3.9e-12 rel, `P_map` 3.9e-11, `res.Q`
9.9e-7, per-band `D` ~1e-16. The f32 deltas are diffuse quantisation
noise — per-plane drift is 1e-9…1e-4 with no systematic bias — and are
orders of magnitude below the JND resolution the metric resolves.
Diagnostic stage-isolation (`--diag`, per-plane `D` comparison) remains
the tool for distinguishing float noise from an algorithmic regression.
The same test covers the `magetypes` SIMD kernels and the `parallel`
feature: `cargo test -p hdrvdp --release --features parallel` exercises
the batched-FFT / `pow_midp` / rayon paths against these goldens —
parallel partitions are structural, so results are bit-identical to the
sequential build (including `score() == hdrvdp().q` exactly), and SIMD
tier differences stay inside the tolerances above.

`res.Q` is far tighter than `P_map` because the border pixels of the
reconstructed `S_map` (the documented `reflect1`-vs-`EXPAND`
synthesis-edge gap) sit in the lowest-weight planes and `res.Q` is a
weighted reduction, while `P_map` exposes every pixel.

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
  numerically faithful port of official 2.2.2 on the cases above. This
  test is the crate's correctness gate — the earlier `bit_lock.rs`
  self-lock (optimized vs frozen f64 copy) was retired when the pipeline
  moved to f32; self-consistency against official output was always the
  property that mattered, and tolerance-vs-golden is the honest form of
  it once the gate itself is no longer bit-exact.
- **Does not**: luminance-encoding coverage only; display encodings
  (`sRGB-display`, `rgb-bt.709`, `XYZ`, `luma-display`) are exercised by
  unit tests but not by these goldens.
- **UPIQ (chunk 4) — closed 2026-09-24**: the real-corpus counterpart.
  All 380 HDR pairs, `luminance` encoding, full resolution, fixed
  `pix_per_deg=30` (the protocol that reproduces the released score
  column): ours vs official `HDRVDP2_2` SROCC **0.9962** / delta
  +0.046±0.363; ours vs JOD SROCC **0.8203** (official column: 0.8117,
  published 0.812). See
  [`benchmarks/hdrvdp_upiq_2026-09-24.md`](../../../benchmarks/hdrvdp_upiq_2026-09-24.md).

## HDR-VDP-3 (`v3` module) — separate validation record

`hdrvdp::v3` is a different metric (HDR-VDP-**3.0.7**, display-adaptive)
with its own gate, kept separate from this 2.2.2 record:

- **Golden corpus**: 21 cases in `tests/vdp3_goldens/` generated by the
  jpeg-ai-qaf `VDP3/` numpy port @ `0628a6b` (`gen_vdp3_goldens.py`),
  covering every `InputEncoding` (rgb-bt.709, rgb-bt.2020, rgb-native,
  xyz, luminance, srgb-display, luma-display, generic+custom emission),
  every `Task` (quality/side-by-side/flicker), `surround` none/mean/cd m²,
  `do_pixel_threshold`, `si_gauss`, `do_masking` off, odd dims, identity.
  `tests/reference_vdp3.rs` asserts worst |ΔQ_JOD| = **1.6e-13**, worst
  |ΔP_map| = 6.1e-13, plus staged intermediates (L_adapt ≤ 1.4e-12, JND
  LUTs, sp0 bands ≤ 1e-12) and Err-not-panic input validation.
- **MATLAB cross-check**: the numpy port itself was validated against the
  **official HDR-VDP-3.0.7 MATLAB release** (SourceForge zip, run under
  GNU Octave 11.1 with `pkg load image` + a local `geomean.m` shim) on
  three AIC-4 pairs under the jpeg-ai-qaf harness ingress: 9.696845 /
  8.517431 / 9.645288 — identical to this crate and the port at print
  precision. So: Rust = port = MATLAB 3.0.7.
- **Published AIC-4 `HDR_VDP_3` column**: NOT reproduced under any
  recovered configuration (ppd/task/EOTF/reflectance/swap/bsc probes all
  leave distortion-proportional residuals) — different upstream run
  config; recorded in `DIVERGENCES.md`, not fitted to.
