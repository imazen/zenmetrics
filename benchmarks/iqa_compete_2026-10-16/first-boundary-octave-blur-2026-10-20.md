# HDR-VDP blur observation and a small image-package overlay

Missing: no identical-raw metric comparison or overlay control has executed in
this lane. The old observation does not establish that Octave 11.1 is required.
No metric, assertion or accepted numerical bound changes here. Retain all
runtime/overlay results as diagnostic until root's controls pass. No Docker
pull, compilation or package installation is needed for the proposed controls.

## What the retained old blur check actually did

The retained `lumcheck.m:8–14` seeds `rand`, regenerates a 128×128 luminance
reference from sinusoids plus uniform random values, builds a 7×7 Gaussian with
explicit `exp`/outer product/normalization, then calls:

```matlab
res = hdrvdp(conv2(ref,k,'same'),ref,'luminance',30);
```

It does not read the committed raw pixels. The committed
`crates/hdrvdp/validation/gen_goldens.m:26–32,40–55` uses the same custom Gaussian
and `conv2` pattern. Neither fixture path calls image-package `fspecial` or
`imfilter`. The new `gen_srgb_goldens.m:48–56` instead forms blur with integer
3×3 sums and floor division; its other distortions are integer-generated too.
It also does not depend on image-package filtering for fixture construction.

Consequently the documented 59.2606 versus 58.6232 observation does not isolate
metric arithmetic on identical raw pixels. The inspected reproduction regenerates
the reference; no matching raw-input SHA receipt accompanies that script. This
does not prove the regenerated pixels differ. It means their identity must be
checked instead of assuming either a fixture or a metric cause.

The frozen original luminance scalar is `58.623187869885932` for blur, and
`96.035911842528023` for dark. Exact committed 128² f64 little-endian inputs are:

| Input | SHA-256 |
| --- | --- |
| reference | `ea12cbe610ee9778be840d0a9267b887e764c1e8f26475f8b010a3ae8652e88a` |
| blur | `d07cb0ca82a50622e000836e93f5d079e47b8a71762a86f0d1d76cadefb1306f` |
| dark | `54ca34c12e618301d6bde925d1977c4274026544762fa6115e7d8d0be2dd293a` |

Reading these bytes removes RNG, fixture-kernel and regeneration differences
from the old luminance test. Root's separate identical-integer-RGB sRGB golden
controls test the actual new production ingress.

## Concrete dependency differences inside the metric

Unchanged official v2 `hdrvdp.m:311,356,359,391` resizes adaptation, neighboring
masking bands and the difference mask within scoring. Official v3 does the same
at `hdrvdp3.m:563,617,620,656` and also resizes adaptation at 802/805. Thus package
resize semantics affect the metric body even when input pixels are fixed.

System image 2.12 `imresize.m:134–135` makes endpoint-spanning coordinates with
`linspace` and calls `imremap`. Unchanged image 2.20.1 uses scaled kernels for
antialiasing (299–305), symmetric convolution interpolation (327–331), and weight
normalization (405–423). These are concrete different arithmetic paths, not a
package-version label alone. The prior VSI control already executed the newer
two-file resize implementation on Octave 6.4; it does not qualify HDR-VDP, but
shows that this selected overlay can run without installing the whole package.

Other inspected package changes provide less specific evidence for this HDR
path: `padarray` has only a trailing comment change; `imfilter` adds a `reflect`
option while its `replicate` filtering path is unchanged; `fspecial`'s Gaussian
branch replaces the deprecated `.+` spelling with `+`. The metric does use
`fspecial`/`imfilter` in some internal Gaussian paths, so an optional broader
M-file control remains useful after testing resize alone. No output equality
between those package versions is claimed from reading source.

## Minimal unchanged upstream source candidate

The already retained official image 2.20.1 archive has SHA-256
`b4b6865e7f65ee5b9c254a69128bf10e93b38bc27f5bc5b973fef300217fc725`.
The inspected files match their archive members byte-for-byte:

| Selected source | SHA-256 |
| --- | --- |
| `inst/imresize.m` | `c358d5381f325c5ab58c7eb0c3ccc8acc2141bc70634c66cc516541b823a15c8` |
| `inst/private/interp_method.m` | `bcd3156239d6d4dccb626108a642afead1b4c60b400fed8b11d73ef5edcc6beb` |

Archive provenance:
[official source archive](https://downloads.sourceforge.net/project/octave/Octave%20Forge%20Packages/Individual%20Package%20Releases/image-2.20.1.tar.gz),
[package catalogue](https://gnu-octave.github.io/packages/image/).
The package's `DESCRIPTION` requires Octave ≥8.1. This proposal selects unchanged
M-files with their private helper; it does not install the package, claim whole
package compatibility, or change the original HDR-VDP bodies.

Root's admission queue can run the same committed raw inputs against:

1. System Octave 6.4 / image 2.12.
2. The same runtime plus only the unchanged two-file resize overlay.
3. If needed, the same runtime plus the retained six-file overlay: resize/private helper, fspecial, imfilter, padarray and private isimage.

Prepared scripts read the exact committed little-endian f64 bytes, reject short
or trailing input, print original Q, hex bits, expected scalar and residual, and
record resolved runtime/helper paths. They assign no new acceptance tolerance.
Run each in a fresh process; retain the old attempts. Root then runs the real
integer-sRGB goldens on identical raw RGB pixels before selecting a runtime.
Keep the existing v2 5e-3 gate and the absent new official v3 bound unchanged.
A residual that persists with the overlay still needs attribution to core
runtime behavior, another dependency, source adaptation or input identity; it
is not automatic proof of a Rust metric defect.

The frozen private evidence bundle owns the original script, archive-member
hashes, raw inputs, two overlays, three new entry scripts and execution bindings.
It is preparation only. Existing running helpers, source bundles and plans were
not edited. No measurement is reported here.
