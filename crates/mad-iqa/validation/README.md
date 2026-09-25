# MAD validation

## Provenance

The oracle is the authors' MATLAB implementation of MAD
(E. C. Larson & D. M. Chandler, JEI 19(1), 2010):

- `hi_index.m`, `lo_index.m` — from the STMAD_2011 package, archived
  in the public `Netflix/vmaf` repository at
  `matlab/STMAD_2011_MatlabCode/` (the author's own site download was
  unreachable at port time). These files carry the authors' copyright
  and are **not committed** here — fetch them from the vmaf repo to
  reproduce.
- `ical_std.c`, `ical_stat.c` — C mex sources in the same directory.
  The environment lacked `mkoctfile`/Octave headers, so
  `ical_std.m` / `ical_stat.m` in this directory are **committed
  Octave shims ported verbatim from the C sources** (same loops,
  same denominators, same bounds checks). They are our own code and
  safe to redistribute.
- The single-image combine (`sig = 1/(1+b1·HI^b2)`,
  `MAD = HI^sig·LO^(1−sig)`, `b1 = exp(−2.55/3.35)`,
  `b2 = 1/(ln10·3.35)`) is the JEI 2010 paper formula — the original
  `MAD_index` bundle was unreachable, so `hi_index` and `lo_index`
  are golden-verified independently and the combine follows the paper
  (and every published port).

## Reproducing the goldens

```bash
mkdir /tmp/mad-ref && cd /tmp/mad-ref
# fetch hi_index.m / lo_index.m from Netflix/vmaf
#   matlab/STMAD_2011_MatlabCode/
cp /path/to/zenmetrics/crates/mad-iqa/validation/{gen.m,gen_goldens.m,ical_std.m,ical_stat.m} /tmp/mad-ref/
octave --no-gui --quiet gen_goldens.m
```

Prints `name w h hi=… lo=… mad=…` rows — these are the `GOLDENS`
table in `src/tests.rs`.

## Pitfall: MATLAB integer typing in `gen.m`

`gen(2,...)` uses `bitxor`, which returns **uint64** in Octave.
`hi_index.m` only checks `isinteger(ref_img)`: with a double
reference and a uint64 distorted image it takes the *non*-LUT branch
`dst = k .* dst_img .^ (2.2/3)`, where the power is evaluated in
**integer arithmetic** — the luminance collapses to 0/1 and `hi`
shifts by ~6%. The reference binary's own uint8 path (both images
integer) takes the LUT branch and produces proper double luminance,
matching what a typed f32 API computes. `gen.m` therefore ends with
`img = double(img)` so the goldens reflect the intended algorithm.
(If you delete that line, `hi` goldens for cases with a gen2
distorted image will regress to the uint64-corrupted values — do not
"fix" the Rust port to match them.)

## Notes on exactness

- `hi_index`'s `ifftshift(fftshift(X).*csf)` and `lo_index`'s
  `X.*fftshift(filter)` use *different* shifts on odd-sized axes:
  `floor(n/2)` vs `ceil(n/2)`. The port implements both.
- `ical_std`'s `std_1` is an 8×8 sample-std of the reference,
  min-pooled over `{0,5}`-offset positions with bounds checks —
  replicated, not a plain 8×8 grid.
- `imfilter 'same'` for the 16-tap box uses window `{i−7..i+8}`
  (verified against Octave), with whole-point symmetric padding.
- `lo_index` uses `abs(EO)` (complex magnitudes), not `EO.real`.
