# Margarine research

Margarine is an unpublished Butteraugli-lineage approximation. It targets
quality-rank loss of at most 0.01 and at most 1% statistically distinguishable
harmful encoder choices, with 4× scoring speed and one-quarter total-process
peak RAM on larger inputs. Full qualification is unfinished; spatial encoder
steering has not been evaluated end to end.

## Build and use

From the zenmetrics repository root:

```sh
just margarine-build
crates/margarine/target/release/margarine reference.png distorted.png
crates/margarine/target/release/margarine --all-scores --diffmap new.f32le reference.png distorted.png
```

The selected command remains `simd-row-malta`, with 128-row/512-column
geometry and primary max pooling. Lower scores mean less distortion.
`--all-scores` reports max/p1/p2/p3/p6; maps are native-size row-major
little-endian f32. Inputs retain encoded-sRGB RGB/RGBA 8/16-bit precision;
alpha must be opaque. Existing diffmap files are rejected.

This standalone package has its own lockfile and is excluded from the umbrella
workspace. Its minimal command excludes the optional `research` dependencies
zensim, zenstats and zenbench. No sibling repository checkout is required.
The eight shared Butteraugli modules are frozen under `vendor/butteraugli/`,
with their BSD license, source commit and SHA-256 hashes in `SOURCE.json`.
The teacher dependency pins the same source revision. No arithmetic changes
are intended by the repository migration.

## Qualification and experiments

[Data and evaluation index](DATA_PROVENANCE.md) records the selected candidate,
the newer native-UHF control, artifact locations, and missing acceptance gates.
[Measurement tables](../../benchmarks/margarine/) retain the historical build
identities. Migration does not turn old measurements into new-build results.

`just margarine-check` runs the research checks; `just margarine-candidate-check
native-uhf` exercises the newer control. `just margarine-report NEW_DIRECTORY
COMMIT` builds the offline report from the retained tables. Research drivers
keep their existing arguments; use `--help` for Python instruments.

The complete pre-migration [research guide](https://github.com/imazen/butteraugli/blob/13c49cbcab6e2b2bb65f60e3cb28344463bacbf7/experiments/margarine/README.md)
and [experiment history](https://github.com/imazen/butteraugli/blob/13c49cbcab6e2b2bb65f60e3cb28344463bacbf7/DATA_PROVENANCE.md)
remain pinned historical records, including their original hyperlinks. Their
`experiments/margarine/` commands now use `crates/margarine/`; measurement tables
now live in `benchmarks/margarine/`. Oversized historical metadata has a small
pointer carrying the original commit URL, file hash and local archive path.
No datasets, encoded images, diffmaps, or generated binaries are copied into git.
