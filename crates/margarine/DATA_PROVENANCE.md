# Margarine data and evaluation index

## Current implementations

The selected command is the streamed Butteraugli-lineage `simd-row-malta`
implementation, 128 rows × 512 columns, primary max pooling. Its frozen quality
build `6bb371fb` covers 43,506 pairs across nine corpora/releases, including
121 AIC3 estimated labels reported separately. All thirteen corpus/cohort
primary point panels meet the 0.01 rank-loss screen against Butteraugli max.
Native16 conversion build `b6c1e6b1` reproduces all 4,292 frozen CID22 maps
and five norms byte for byte. These are historical measurements, not a new
qualification run after relocation.

The newer `native-uhf` control (`14b1b68a`) retains both native UHF Malta banks
and interpolates only four smoother HF/MF banks. It remains unselected.
Its primary SROCC deltas are −0.000176446 on AIC4 and +0.000675806 on CID22.
Four LIVE1 sessions stay within the rank-loss screen and have zero supported
harmful choices across 328 observed budgets. The measured 1 MP/8.44 MP
photographic crop speedups are 4.019170×/4.171563×, with total-process RSS
fractions 0.189235/0.081399. Decode-inclusive speedups are only
3.295749×/3.393420×. These are measured means, not confidence-bound passes.

Full acceptance remains unproven. The newer control needs the remaining
corpora and independent content/resource checks. CID22 lacks participant
uncertainty; AIC4 lacks original bitrate information. Spatial steering inside
an encoder has not been evaluated end to end. Preserved maps and the older
peak-location diagnostics do not establish steering quality.

## Records and artifacts

All compact tables and artifact pointers live under
[benchmarks/margarine](../../benchmarks/margarine/). Their build commits refer
to the original [Butteraugli repository](https://github.com/imazen/butteraugli)
unless explicitly marked as zenmetrics commits.

- Mac artifact root: `/Users/lilith/work/codec-artifacts/margarine/`.
- Linux artifact root: `/home/lilith/work/codec-artifacts/margarine/`.
- NAS canonical root: `/mnt/user/coefficient/output/margarine/`.
- Migration archive: Mac artifact root `repo-migration-2026-09-27/originals/`.
- Complete migration inventory: `repo-migration-2026-09-27/inventory.json`.

Existing corpus images and generated maps stay in their current artifact
stores. Earlier selected-candidate runs have verified NAS mirrors. The latest
native-UHF maps are on the evaluation host, with metadata on the Mac; those
runs have not yet been mirrored to NAS. Host aliases and access details remain
in the private machine configuration and original artifact manifests.

The [complete historical index](https://github.com/imazen/butteraugli/blob/13c49cbcab6e2b2bb65f60e3cb28344463bacbf7/DATA_PROVENANCE.md)
preserves all earlier experiment decisions, source locations and hyperlinks.
The [original guide](https://github.com/imazen/butteraugli/blob/13c49cbcab6e2b2bb65f60e3cb28344463bacbf7/experiments/margarine/README.md)
documents each experimental switch and evaluation contract. Migration records
must not overwrite original manifests or relabel an old result with a new hash.

## Relocation verification

The [migration record](../../benchmarks/margarine/margarine_repository_migration_2026-09-27.json)
pins the source revision, complete file inventory, and pre/post binary hashes.
Both the selected command and native-UHF control reproduce all five scalar
fields and native map bytes on all 300 AIC4 pairs on the migration host.
This establishes relocation equivalence for that replay, not a new cross-arch
quality or performance qualification. The original artifact stores remain intact.
