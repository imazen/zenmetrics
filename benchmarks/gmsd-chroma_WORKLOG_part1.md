# gmsd-chroma worklog

Quarantined, lane started 2026-09-24. No push. Part B prerequisite is ready;
Part A qualification is still pending.

## Bootstrap and command recording

Initial read-only instruction/source discovery and workspace creation were
performed through the Codex tool transcript before the file-backed recorder
existed. Their exact wall times were not retained; they are not represented
as reproducible numeric evidence. Workspace creation printed UTC
2026-09-24T07:47:45Z. The workspace was created at master@origin b50cde1f.
That tree lacks gmsd; read-only jj history identified optimized tip 8fc65d91.

From 2026-09-24T07:49:05Z onward, recorded experiment commands live in
`/var/tmp/gmsd-chroma/commands.jsonl`: UTC start/end, cwd, argv, exit code,
full stdout+stderr log path and SHA-256. `run.py` is the recorder. Reading
source, tool-based patches and web searches also remain in the transcript.
The lane manifest is `/home/lilith/tmp/devin/rev4_gmsd-chroma_manifest.tsv`.
Each final record binds this append-only command index with a digest.

## Reference discovery

The author MDSI release downloaded successfully from MathWorks' public
versioned archive endpoint. `inspect_refs.log` contains archive contents,
complete source/licence and sha256sum output. No author file is in git.
MDSI's negative similarity values require complex principal-root pooling.

MS-GMSD primary sources inspected: the authors' paper
<https://www.cse.ust.hk/~psander/docs/gradsim.pdf>, Pedro Sander's author
page <https://www.cse.ust.hk/~psander/>, and HKUST's publication record.
No author software obtained yet; third-party PyTorch implementations are
not substitutes for author-reference parity.

Docker's existing cache has no gnuoctave/octave image. To comply with the
lane's /var/tmp-only cache rule, an isolated daemon is queued under the
heavy lock, with its data/exec roots and all container mounts under
`/var/tmp/gmsd-chroma`. No host packages are installed. The resolved
amd64 image manifest is
`sha256:185db7993e000d4f3f6e7bbbf7fb3f999f52e799ea52231ad8a15353381e0dcb`.
The version/package smoke run has not completed as of this entry.

## Local prerequisite merge

`prerequisite_merge.log` records `jj new master@origin 8fc65d91...`.
Both parents are preserved. CHANGELOG conflict combines the CVVDP and GMSD
additions without altering their text. No other lane's checkout was edited.
The shared zensim `.workongoing` marker is refreshed per LANE_PREAMBLE.

## Frozen TRAIN selection and unvalidated implementation (08:04Z)

`select_train.log`: `native_train_pairs 72`, selection SHA-256
`f1da89bd4e77a5971a5003dfd6a2cb62c588e11ea22c9c8039a5de361cb6c042`.
This is 40 KADID colour pairs (8 each of types 04–08), 16 CID22-train and
16 SafeSyn, selected from label-free authorities. Amendment 1 corrects the
initial KADID numeric type list using the dataset author's descriptions.
The decoder selected for new KADID pixel dumps is the existing canonical
GMSBANK decoder binary, SHA-256
`3306465d56d279a512b02c3a63de701e9b634d6aebda5e12fb164817ba491122`.
Its `dump` route checks both pixel hashes and pair-key construction.

New source implements the default MDSI sum model, f64 maps, deterministic
pooling, the additive RGB8 entry and CLI dispatch. It is **not compiled or
validated yet**. `mdsi_oracle` calls the public scorer and includes the same
private module for map export, keeping map access out of the public API.
The external Octave wrapper (kept in the external archive) compares against the
untouched reference scores; one Octave function absent from the base image was
shimmed, details in the external archive.

`script_syntax.log`: `parsed 4 Python scripts` (AST syntax only).
The Octave job remains queued on the common heavy lock; its log is empty.
No corpus pixel scoring, reference execution, tests, clippy or timing has
run. No label read has occurred. Part B remains gated by its missing marker.

## Speed harness prepared

`chroma_speed` uses the current local zenbench 0.1.9 source (README and API
read before use), with the required four sizes and 1/8-thread pools, 10–30
interleaved rounds per cell. It saves JSON samples and CSV externally.
The lib.rs version check returned HTTP 403; this is a path dependency, not
an inferred registry version. Build/lockfile update and timing are pending.

## Source review and pending checks (08:13Z)

A second preregistration correction restricts TID to its 12-reference TRAIN
view; the previous wording "all TRAIN" was wrong. DATA_SPLITS.md section 8
is authoritative. No human-label data has been read. The MDSI CLI column
entry was corrected to the existing borrowed-string return shape during
source review, before compilation.

`build_checks_v1` is queued through the shared heavy wrapper after the
Octave smoke job. The script copies dependency caches to /var/tmp before
Cargo can write, then formats only gmsd/zenmetrics-cli, tests all gmsd
features, checks no_std, builds release examples, and runs crate clippy.
Neither job has produced output yet. The workspace justfile has no clippy
or lint-scripts recipe, so explicit package checks are required. Its
package-scoped formatting rule prevents writes into sibling repositories.

## Octave smoke completed; reporting harnesses (08:19Z)

`octave_smoke.log` SHA-256
`bd758ccea8298f6e5ba576d1323a0b44339c3f0beef23c2cae06946f477cfdea`
records exit 0, Octave `11.3.0` and `no packages installed.`; smoke details are
in the external archive. The wrapper reports
31 seconds inside its resource scope; the 26-minute command interval also
includes waiting for the shared lock. The isolated daemon has exited.

The build/check job and the canonical-decode/input-stage job are queued.
The reference driver is queued after staging, using the same pinned image.
Each job checks the quota-stop marker before starting work.

`report_colour.py` reuses `zen_stats.panel`, its signed batch entry, and
`band_reliability.marginal_ci` with reference clusters; both source hashes
are pinned. It requires a frozen implementation/population and measured
parity pass before opening its report table. No table or labels have been
read. `report_speed.py` uses NumPy OLS on raw zenbench means and retains
residuals, with explicit contended/pre-round-clean labels. Neither report
has run. `harness_syntax_v2.log`: `parsed 6 Python scripts`.

Part B marker appeared by 08:19Z, naming corrected GMSBANK tip
`620a384e06e49422d501bdddc9c0439afc846a98`. Part B has not started: the
brief requires Part A first.

The TID restriction in Amendment 2 was itself mistaken: DATA_SPLITS.md
section 8.1 explicitly supersedes its historical table with the user's
all-3,000-row / 25-reference TRAIN-only ruling. Amendment 3 restores the
original admission before any population construction or label read.
The split-compliance owner and GMSBANK registry corroborate it. No other
corpus admission changes.

## Reporting population (08:29Z)

`freeze_colour_population_v2.log` records 1,539 pairs/stimuli, 200 per
KADID type 04–08, 125 per TID type 02/18/22/23, and 39 KonFiG colour
diffusion rows. Population and scoring TSV hashes are in the prereg.
The first path-derived population had 1,499 rows and is superseded, kept
under `colour/` with its producer retained as `prepare_colour_inputs_v1.py`.
`inspect_colour_metadata.log` found 120 full-KADID canonical-path type
mismatches (e.g. a named type-08 stimulus uses `I02_01_01.png` as its
identical-pixel representative). Therefore source stimulus metadata is
the type authority. The corrected colour subset itself has 1,539 unique
keys and 1,539 stimuli; no duplicates are dropped in it.

Only four metadata columns were projected from the three TRAIN label
containers: pair_key, source_row_id, codec, knob. No human_score value was
requested. Schema-only inspection of stored tables found no SSIMULACRA2
column in the ceiling human tables; the stored peer source is still to
be identified before the accuracy report can run.

## Admission checks and bitwise coverage

Pending heavy scripts now check quota, the /home floor, and every output/
cache environment at actual lock admission; `preflight_light.log` passed.
Docker daemon logs get unique filenames so the smoke log is retained.
`syntax_v3.log` parsed 9 Python and 3 shell scripts. The queued jobs have
not begun their payloads yet; script changes precede execution.

The oracle driver now compares exact f64 bits for CS, GCS and pooled scores
on every input across scalar and all available SIMD tiers, plus 1/8-thread
map runs. This replaces ordinary float equality in the tier unit test too.
No numeric tolerance is relaxed. Code remains uncompiled at this entry.

## First build and peer provenance (08:50Z)

`build_checks_v1.log` (SHA-256
`dcb084e5e1436c5a312db11aaa43c4102d93377698623dfb5a86e3d2fe87e0ba`)
was admitted at 08:47:34Z and failed compilation, rc 101, at 08:48:04Z.
Rust/Cargo are 1.98.1. `arcane` cannot parse a macro's opaque `ty` fragment;
the band macro now passes a bare token identifier. The AVX-512 token has an
f64x8 backend, not f64x4; maps now use eight independent f64 pixels on all
tiers (native AVX-512, split AVX2/NEON/WASM, scalar fallback). No reduction
or formula changes. Retry `build_checks_v2` is queued. No test pass claimed.

Cargo refreshed the lock to current path-dependency manifests (including
jxl-encoder 0.4.0 and zenavif 0.2.0); these sibling sources were not edited.
The initially selected local zenbench source differs from published 0.1.9.
Use registry 0.1.9, whose inspected source supports the same harness calls,
so Cargo.lock pins it rather than depending on mutable sibling source.

The speed harness retains resource checks with zero wait budget, recording
unclean rounds as CONTENDED. Its UTS namespace sets a neutral name solely
inside the benchmark process tree because zenbench stores host names in
lock metadata. `speed_namespace_smoke.log` rc 0, SHA-256
`b76f75d2140f84e996a54bac41989ce1b7eef0d5a76177a069ae4b626ce93955`,
prints `isolated_hostname_ok`. No timing was performed in that smoke.

Following the user's global-CLAUDE hint, schema-only reads located
`/mnt/v/zen/zensim-training/2026-05-18-ssim2/{kadid,tid}_ssim2_local.parquet`.
`peer_schema_v1.log` reports 10,125 / 3,000 rows and `ssim2_gpu`, consistent
with the directory manifest and DATA_INTEGRITY_root_cause_2026-05-25.md.
They are not relabelled fast-ssim2 CPU. `peer_schema_v2.log` finds no peer
column in the July KonFiG or TRAIN joint-core KonFiG parquet; canonical
KADID/TID TRAIN tables also carry `ssim2_gpu`. No payload values were read.
The requested stored CPU peer source remains unresolved.

### Frozen timing harness (08:53Z)

Further schema inspection found that the published zenbench 0.1.9 does not
retain individual rounds, so it cannot support the declared raw-sample
report. The final build recipe uses an external source snapshot of the
inspected local imazen harness, injected with Cargo's per-command
`patch.crates-io.zenbench.path` option. Its 40 files (source, manifest,
declared examples/benches, README and licences) are byte copies under
`/var/tmp/gmsd-chroma/zenbench-snapshot/`; inventory SHA-256
`394a18b62ba4a67b6782548d781ed9bf51c57663bb2e368d336291f3c1064da6`.
The build checks that digest and every member before compiling. This
supersedes the registry-only plan immediately above. No benchmark has run.

A pooling-order correction made at this checkpoint (details in the external
archive) precedes any oracle score read and does not change the declared
tolerance. C8's separate row-merge contract is unaffected. Retry remains queued
at this checkpoint.

## MS-GMSD paper fallback prepared (09:03Z)

Commit `2eb3987e` preregistered the paper variant's missing numerical
conventions before measurement. Rust source now implements four scales,
masked gradient similarity and I/Q colour RMSE with the published weights
and logistic fusion. Public RGB8 entries and CLI routes are additive.
The independent NumPy oracle uses array operations and the paper equations;
it does not call the Rust implementation. The shared oracle driver checks
all available tiers, 1/8 threads, stride, four maps and both final scores,
with c=17 as a deliberately wrong stabilizer. All remain uncompiled and
unmeasured here. Existing MDSI numeric gates are unchanged.

CLI checks now enable `sweep`, which owns `score-pairs`; the initial check
recipe only covered `score`. This also pulls the workspace's existing HDR
and codec dependencies. No decoder code was changed. The independent
NumPy job is queued through heavy as `ms_numpy_oracle`.

The TRAIN bank label schemas and converter were inspected without reading
human_score values. Their extraction audits also contain human_score and
mixed-role rows; those audit payloads were not opened to search for peers.
Only the manifests and producer source were read. Stored CPU peer columns
are still not located; GPU sidecars remain unqualified substitutes.

## Shared arithmetic and colour ingress (09:16Z)

Both f64 chroma scorers now expand the same private Prewitt arithmetic,
without changing its operation order. The colour scoring example consumes
canonical raw RGB8 and emits the three newly computed distance arms. Its
preparation script requires the already committed population hash, checks
all decoded dimensions and pixel hashes, and preserves every admitted pair.
CLI smoke coverage remains a separate requirement. No metric was measured.

`colour_rgb_prepare.log` rc 0 (UTC 09:16:06.774–09:16:06.802), SHA-256
`1fe7d66d68fff3fc454b06e8089108becd2310a80537986f07398e7232fd9af8`,
prints `colour_pairs 1539` and decoder-input SHA-256
`a0219eb89c74fc80edd1658dedcfc71519b707bd84cdf34641a06ee8128c13a4`.
Exact command/cwd is in commands.jsonl; output is
`/var/tmp/gmsd-chroma/colour_v2/decode_pairs.tsv`. Human values remain unread.
Python source syntax parsed successfully; Rust is pending queued build v2.

## Qualification wiring (09:26Z; no new metric result)

`validate_oracles.sh` runs the Rust driver and retains both numeric reports
if a comparison gate fails. `stage_colour.sh` only decodes/hash-checks the
frozen TRAIN population. Both are queued under heavy. The speed matrix
now includes MS-GMSDc alongside GMSD/MDSI (24 cells); its report retains
individual rounds, contamination flags and descriptive fit residuals.

`build_examples.py` binds every example binary to source and Cargo.lock
hashes, checks the source did not change while compiling, and saves a
versioned build record. Oracle/timing/prediction jobs reject stale builds.
The reporting join script enforces both parity passes, exact predictor
coverage and an explicitly documented CPU peer source before opening
TRAIN human scores. Those predictors and reports have not been run.

A schema-only scan (no values/statistics) examined 270 likely-corpus
parquets in 1.19 s. It included metadata from mixed/validation files but
read no human labels or row groups. `peer_schema_corpus.log` rc 0, SHA-256
`5eb78def69465b528e1c1ae3402ccb63d8f1fafa375443b59ea6ad3d9c223503`,
prints `schemas_only 270` and `with_peer_named_columns 130`. The 62
`score_ssim2` columns are in old model predictions_val tables; the
remaining SSIM2 sources use `ssim2_gpu`/`ssim2_log_norm`. KonFiG schemas
had no peer column. No qualifying CPU TRAIN source is established.
The user's clarification about whether the GPU columns were intended
is pending. This does not change the frozen metric or role definitions.

Python AST and bash -n checks passed for all lane scripts; no new Rust
check has completed. UTC/commands/rc/log digests are in commands.jsonl.

## Decoder freeze and static review (09:33Z)

Copied the verified canonical decoder into this lane's `bin/` so another
lane's future rebuild cannot invalidate queued preparation. Snapshot
`decoder_snapshot.json` records 10,177,384 bytes, SHA-256
`3306465d56d279a512b02c3a63de701e9b634d6aebda5e12fb164817ba491122`;
source and copied bytes were both hashed at 09:30:37.861Z. Both preparation
scripts now use that immutable copy. No pixel output was generated yet.

Static review replaced index-only loops with iterators and collapsed the
MDSI box bounds check, preserving arithmetic and traversal order. All
Python AST/bash syntax checks pass. The build recipe now ends with scoped
`cargo fmt -- --check`; zenmetrics' justfile forbids `fmt --all` because it
can rewrite sibling path dependencies, and defines neither `just clippy`
nor `just lint-scripts`. Package-scoped clippy and script syntax checks are
the corresponding checks for this workspace.

Footprint enumeration now includes root-owned throwaway Docker cache names
through a read-only sudo enumeration of this lane's scratch root. It does
not follow symlinks or read payloads. Enumeration itself remains pending.

## no_std scalar square root (09:38Z; awaiting compile)

Installed Rust 1.98.1 core source marks standalone f32 sqrt as unstable
`core_float_math`; the inherited no_std branch used the std-only inherent
method in its scalar gradient and f64 Newton seed. Route those scalar
calls through a private helper: std retains `v.sqrt()`, no_std uses
`libm::sqrtf(v)` from the already added dependency. The three f64 Newton
steps and all standard-build arithmetic remain as before. This is source
inspection, not a completed no_std check; build_checks_v2 includes that
check and the inherited GMSD tests.

## Author reference completed (2026-09-24 09:45Z)

`stage_train_inputs`: rc 0; queued 08:18:50.345Z, admitted
09:45:15.900Z, ended 09:45:16.383Z. Log SHA-256
`fabfc7c8608a8eafc535e7936dd55bea2c0e01e2b932c72280015117a490e010`.
Exact output: `oracle_rows 116`; input TSV SHA-256
`25c4702ea483b127e37faf1249c81f8623e652b7a4c3800690d7c7a45175eae1`,
input manifest `48443e24866b2cbce577bd82519dbb2b8d6c32d0e2b17941ca6d98ceefead526`.

`octave_reference_v1`: rc 0; queued 08:19:48.381Z, admitted
09:45:18.012Z, ended 09:45:21.963Z. Log SHA-256
`715ad6ee3225b2aeeb74784f1826b3bd5839be7ccbc5ab6fad99085090cd9c1a`.
The pinned container printed `Octave 11.3.0` and
`oracle 116/116 stress3 3.03186812161755082e-01`. Every instrumented return
matched the untouched reference scorer exactly; one compatibility shim was
used (details in the external archive). No Rust parity conclusion follows yet.

`octave_inventory` rc 0, UTC 09:47:09.522–09:47:09.549Z, log SHA-256
`00e3bc67351c2478216ee89c69490db66081f34331ae0691c05ba6022df0dfcc`:
116 scores / 116 unique IDs, 696 map files. Scores TSV SHA-256
`cb4bb01fc625e960f9dac169a3027f1bf31ab552e7c58d44fb8079295f1cbe83`.
Exact recomputation commands and outputs are in commands.jsonl / named
logs. Rust build v2, NumPy, comparison, colour decoding and timing remain
queued. Read-only C8 integration notes are external; Part B has not begun.

## Explicit branch coverage and CLI ingress checks (09:54Z)

The four complementary-colour stress fixtures do not themselves reach
negative GCS: `stress_map_inventory.log` rc 0, SHA-256
`e9c714548955fa5088199eed69ec006cbb11dd12abd8f94dbd777b856fdb938d`,
reports zero negative samples for all four. `compare_oracle.py` now counts
negative samples in every author GCS map and refuses a coverage pass if
none exists. This enforces the preregistered complex-root coverage intent;
all numeric tolerances and the input population are unchanged. A targeted
supplement will be required if the full corpus has no negative samples.
No Rust result has been read or tolerance adjusted.

Prepared `cli_smoke.py`: one already frozen TRAIN pair per set, canonical
raw RGB scoring versus both CLI `score` JSON and `score-pairs` parquet,
joined by pair key and compared bitwise for GMSD/MDSI/MS-GMSDc. It is queued
as `cli_smoke_v1`. Raw predictor output now uses 17 digits after the decimal
in scientific notation, preserving small f64 values on round-trip. This
was changed before any Rust score run/build success. Python syntax passes.

## Combined-table peer search (09:58–09:59Z)

The generic-name schema scan sampled 1,292 filename families representing
37,012 parquet filenames; 286 families had peer-like columns. It reads
field names only, never row groups or statistics, and does not establish
corpus coverage from filenames. `peer_schema_combined.log` rc 0, SHA-256
`0c30553c397bfe58bec1506da9fd1fb8192f5222f49437e93f182d28e6c27d57`.
No CPU KADID/TID/KonFiG TRAIN source was established. The promising
`im26anchor` konleg table was checked only through its dataset/codec
columns: 40,000 rows, four imazen codec legs of 10,000 each. No peer score
or human value was read. `peer_konleg_metadata.log` rc 0, SHA-256
`e9dd032fe28baf392fa3a0fbf38d96458050361b23178a5b317ca10b5b4cd8c5`.
Those codec legs do not supply the requested colour-distortion population.

## Post-run author-file integrity (10:23Z)

`author_artifact_recheck`: UTC 10:23:06.938–10:23:06.954Z, rc 0, log
SHA-256 `5db74fb0cc89fe42759deb9084abb4767b688e73c6e216e59206602d39fdb454`.
Output: `{"author_files_hash_match": 3, "map_files": 696,
"unique_scores": 116, "wrapper_body_unchanged": true}`. The three original
release files are unchanged; after normalizing line endings, the wrapper
changes only the function signature. The reference inventory now includes
an exact command replay for this read-only check. The queued Rust build
has not been admitted; no Rust-parity, test or speed pass is asserted.

## Rust build/check results and peer-rescore authorization (10:35Z)

`build_checks_v2`: queued 08:49:53.027Z, admitted 10:24:27.113Z,
ended 10:25:09.431Z, rc 101. Log SHA-256
`5bac35d49a4fd440a2cd132aaa10bdaa33ffbfff35e4f6529673de725c3c4ce6`.
Exact test result: `21 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`.
The gmsd no-default-features check, release examples and all-target/all-feature
clippy passed; CLI feature check passed. CLI clippy then failed on 13 inherited
unused/unreachable diagnostics in cvvdp-gpu and iwssim-gpu with no GPU backend.
The CLI release build and final fmt check did not execute. The changed-crate
CLI clippy retry uses `--no-deps`, retaining `-D warnings`; no warning or test
is suppressed in our source. That retry and release build now precede the
queued CLI smoke check. A redundant separately queued CLI retry was cancelled
before admission (only its verified childless flock received SIGTERM);
`finish_cli_checks_v1.log` is empty, rc -15, not a check result.

The successful example build manifest is
`build_examples_20260924T102441590276.json`, SHA-256
`9e438380b54c72ce64146910f202d8ae940390017bdc3a632d16389558bff2b2`.
Later Cargo invocations changed the zenbench lock resolution back to the
registry source, so this manifest fails the current lockfile-hash gate.
The lane now supplies the same frozen patch through its private CARGO_HOME
config as well as command arguments, covering clippy's child Cargo invocation.
The queued oracle-validation job rebuilds examples and records the actual
post-resolution lockfile hash before any comparison. Metric source hashes
must remain unchanged during that build. Rustfmt changes are retained.

The user authorizes a main-branch fast-ssim2 rescore on remote workers.
The prereg amendment pins `f011259a0cd2fb7f538e08398ba41c0ccd0dfcb4` and
the unchanged full TRAIN population. A small private RGB8 driver calls the
existing public CPU API; it is unbuilt. Both workers initially returned
No route to host. The existing private wake utility sent both wake requests;
`remote_wake_request.log` rc 0, SHA-256
`65082a53ced9605e8256ebe199e3e6cdf86e70e0bf4d7421c7b304323105e87c`.
The first post-wake probe still failed on both (`remote_probe_after_wake.log`,
SHA-256 `d03fc765c45540b8c51c8c1ec9dd84919ad2abebf71025ba7560d79c574c175c`).
Private addresses and household identifiers are omitted. No remote job or
human-label read has occurred. Python AST and shell syntax checks pass.
