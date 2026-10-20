# HDR parameter I/O: source review and resident-loader proposal

Status: source review only. No inference, numerical controls or timing has run
for this proposal. Existing reference adapters include parameter loading inside
their clocks; preserve them as diagnostics. Keep `unmatched_scope` and
`matched_scope=false` on successor plans until the controls below pass and the
remaining evaluator boundaries are reviewed.

## Original APIs do not accept resident spectral arrays

The frozen v2 `hdrvdp.m` SHA256 is
`a7ba38438c3fcc5d0cc02b28fa18545640e0321bcaf84fe98016e3f1a4c5d9d6`;
v3 `hdrvdp3.m` is
`0f9f267f68b042c167ef03f41daaf07388ae192ab30f81b243766863f9a5402a`.

Both option parsers accept arbitrary field names. That assignment does not
create a matrix-input API: nonempty `spectral_emission` is consumed as a
filename by v2 lines 246–247 and v3 lines 438–439. The documented options also
describe a filename. V2 constructs a path by concatenation; v3 uses `fullfile`.
Neither has a numeric-array branch. Both overwrite the field with the loaded
matrix before invoking their visual pathways.

Observer loading cannot be bypassed through options either. V2
`hdrvdp_visual_pathway.m:57,61` and v3 `hdrvdp_visual_pathway.m:58,62`
unconditionally call `load_spectral_resp` for the cone and rod tables.
Adding an observer field through the permissive parser leaves those calls
unchanged. Calling the visual pathway directly would omit the original
top-level evaluation and pooling, so it cannot substitute for the official
score route.

Preserve distinct default emission tables:

| Original route | Emission table | Observer tables |
| --- | --- | --- |
| v2 sRGB-display, 30 ppd, `Q` | `emission_spectra_ccfl-lcd.csv` (`hdrvdp.m:242–249`) | `log_cone_smith_pokorny_1975.csv`, `cie_scotopic_lum.txt` |
| v3 quality, sRGB-display, 30 ppd, CPU, age 24, surround none, `Q_JOD` | `data/emission_spectra_led-lcd-srgb.csv` (`hdrvdp3.m:388,441`) | Same table basenames under `data/`, plus `data/ciexyz31.csv` for display-primary conversion |

The original v2 `load_spectral_resp.m` and v3
`utils/load_spectral_resp.m` have identical bytes, SHA256
`f51b19ae6a874bf5474a0bee367d9564fac84b091b9d343bc7c7a6e72e6ca560`.
Line 10 calls `dlmread`; lines 11–20 build the wavelength grid and interpolate
each column using the original `interp1(...,'pchip',0)`. V2 has one
emission-loader call and two observer-loader calls per image pathway (five
calls across three unique files). V3 additionally calls
`hdrvdp_iturgb2native('rgb-bt.709',IMG_E)` for sRGB-display
(`hdrvdp3.m:467–468`); that helper loads the **bare filename** `ciexyz31.csv`
at line 6 (six calls across four unique files). Its integration and matrix
solve must remain inside the original call. Those repeated loads are not
avoided by the separate original photoreceptor cache.

## Proposed separate read-only loader cache

Prepare a new frozen cohort after the runtime golden decision. Keep original
metric bodies, helpers and prior plans intact. The proposal changes dispatch
only at the `load_spectral_resp` boundary:

1. With the original helper selected, preload its **two outputs** for each of
   the three v2 or four v3 exact-bound files before any metric evaluation or timing. Keep
   original class, shape, ordering and every bit. Do not reproduce its
   wavelength grid, interpolation or arithmetic. Preloading file bytes alone
   would leave parameter parsing/resampling inside the evaluation clock.
2. Install a separate path shim named `load_spectral_resp.m`. Its only action
   is to retrieve the saved output pair by an exact request-string key. V2
   uses three bare filenames; v3 uses three absolute paths constructed
   from its own frozen base directory plus the exact bare `ciexyz31.csv` key.
   No basename fallback, path search,
   canonicalization, file open/stat/hash, lazy fill or cache-miss delegation
   occurs in the scoring body. Unknown requests fail explicitly.
3. A separate cache owner permits one bootstrap initialization and thereafter
   exposes reads only. No reset, replacement or insertion is permitted after
   sealing. It returns arrays with Octave's normal value semantics. Preserve
   the original subsequent cone exponentiation, rod insertion, spectral
   integration and every other operation inside the original metric bodies.
4. Before installation, explicitly configure original tree, pyramid and v3
   utils/data paths so the metric's automatic path setup is unnecessary.
   Install the shim last, keep the working directory outside original helper
   directories, and assert every selected function path. V3 adds utils when
   `reconSpyr` is missing (`hdrvdp3.m:311–317`); silently allowing that branch
   could replace the shim. Do not modify that original branch.
5. Hash exact source/data/runtime closure before and after each process,
   outside the measurement clock. Bind original helper, shim, cache owner,
   loaded table hashes and bootstrap provenance to every plan and result.
   Initialization, parameter I/O, imports, raw input reads and hashing stay
   outside first and warm clocks. RGB conversion, original top-level call and
   result destruction retain the existing adapter boundary. First means no
   earlier metric evaluation; warm performs its ordinary untimed warmup only
   after installation.

This removes only repeated immutable parameter acquisition from evaluation;
it does not precompute the image transform or move original pathway arithmetic.
The separate AIC 200-nit XYZ adapter retains its original convention.

## Required independent controls before scope promotion

The coordinator runs these controls under the existing resource admission:

- Compare every cached wavelength/response array bit-for-bit to fresh outputs
  from the original helper in a separate control. Mutating a returned array
  must leave a subsequent retrieval bit-identical; repeated initialization and
  unknown-key requests must fail. No metric precheck enters a timing process.
- Compare unchanged original-default and resident-loader scores in separate
  fresh processes on the same raw inputs, runtime and thread budgets. Cover
  the 512 photo and full original 12-pair grid at 1/28 threads, for first and
  warm phases. Require finite scores and identical `num2hex` strings across
  both arms and repeated resident evaluations. Keep complete output/logs and
  source/runtime identities. A scalar tolerance cannot replace this exact
  identity gate.
- In independent diagnostic processes, omit each cache key
  in turn and require the actual original metric call to fail with that exact
  missing key. This detects helper-resolution bypass and incomplete loader
  coverage without adding counters to the timed body. Unexpected file access
  or fallback must refuse the resident cohort.
- Retain the original v2 all24 `5e-3` golden gate for the selected runtime.
  This official v3 photo/grid cohort still has **no assigned external bound**;
  exact default-versus-resident identity alone does not supply one. Its older
  NumPy-golden bound belongs to that separate corpus. Missing numerical
  qualification and other scope differences continue to withhold ratios.

No resident shim is installed or active by this source-review proposal.
