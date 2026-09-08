# Static AV1 comparison executable

One executable links C libaom, C SVT-AV1, zenav1-aom, zenav1-svt and zenrav1e.
The C adapters use public encoder APIs and return owned OBU bytes through a
checked Rust boundary. This is benchmark tooling, with no change to the
production Rust codecs' zero-C dependency contract.

The initial `av1-api-i420-v1` protocol accepts one packed, limited-range,
8-bit I420 still, even dimensions 64..16384, and threads=1. For C SVT that
means **lp1**, not a promise of exactly one OS thread. Quantizers and speeds
are backend-native; equal numbers are not equal effort or achieved quality.
AOM uses ALL_INTRA with explicit SB64; C otherwise retains library tool
defaults, and Rust AOM enables restoration to match the C ALL_INTRA default.
SVT uses still-picture CQP (AQ0). Unsupported formats are errors.

C source is taken from the pinned sibling submodules, built separately under
Cargo OUT_DIR with multithreading enabled, native tuning off and FP contraction
off. The differential oracles and their caches are untouched. The small nested
workspace allows building this executor without unrelated GPU and HDR corpus
sibling repositories; the normal zenmetrics workspace currently requires those.

On Linux x86-64, build and test under the shared heavy-job wrapper:

```sh
RUSTFLAGS='-C target-feature=+crt-static' cargo test --release \
  --target x86_64-unknown-linux-gnu --manifest-path benchmarks/av1-compare/Cargo.toml
RUSTFLAGS='-C target-feature=+crt-static' cargo build --release \
  --target x86_64-unknown-linux-gnu --manifest-path benchmarks/av1-compare/Cargo.toml
```

Check `file` and `readelf -d` on the resulting executable: static C archives
alone do not prove the final executable has no shared-library requirements.
Run its smoke request in the deployment environment before distributing it.

Input is a JSON request on stdin, for example:

```json
{"config":{"backend":"libaom","width":64,"height":64,"quantizer":40,"speed":6,"threads":1},"input":"source.i420","output":"libaom.obu"}
```

Backend names are `libaom`, `c-svt-av1`, `zenav1-aom`, `zenav1-svt`, `zenrav1e`.
The output row records settings, source/output hashes, revision, byte count,
and fresh-lifecycle API elapsed nanoseconds, including setup, owned output
copying and teardown. Plane preparation is included. Input file I/O, hashing,
independent decode and output file I/O are outside the timer. This is not a
codec-kernel-only timing. Every output must pass libaom decode and dimensions,
bit-depth and chroma checks before a success row is emitted.

The unit-test checkerboard is solely a correctness witness. It is not a speed
or quality calibration corpus. Real comparisons must use the registered
CID22/CLIC/screen sources, repeated interleaved trials on the same worker,
independent scores, and matched-quality comparisons.

This executable does not schedule or partition fleet work. The zenmetrics
planner, zenfleet declarations/claims/ledger and the existing Nomad worker
lifecycle remain the owners. Its current stdin request is a local smoke API;
a DesiredJob adapter and timing/artifact result persistence are still needed
before declaring the distributed comparison campaign. Wider formats, explicit
color signaling controls, encoder-only timing and multithread arms remain
follow-up work, not advertised support.

## Local validation, 2026-09-08

- Release tests: both passed, including all five encoder arms decoded through
  libaom and invalid-input refusal before FFI.
- CLI smoke: all five arms emitted valid rows and independently decoded OBU
  files; repeated output paths were refused.
- Linux x86-64 executable: `file` reports static PIE; `readelf -d` lists no
  `NEEDED` entries.
- This crate's all-target Clippy passes with warnings denied. Existing warnings
  in the SVT/AOM dependencies remain; this is not a claim that those workspaces'
  broader gates pass.
- New files pass the repository's shared hygiene patterns. No CI or fleet
  campaign was started by these checks, and the smoke timings are not routing
  calibration data.
