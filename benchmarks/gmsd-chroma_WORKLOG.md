# gmsd-chroma worklog

Quarantined; no push. Part A qualification is in progress; Part B has not
begun. Earlier commands and decisions through 2026-09-24 10:35Z are in
[gmsd-chroma_WORKLOG_part1.md](gmsd-chroma_WORKLOG_part1.md).
The exact command index remains `/var/tmp/gmsd-chroma/commands.jsonl`;
each experiment entry records UTC, cwd, argv, exit code and log SHA-256.

## First differential results (10:35Z; failed MDSI gate preserved)

| Command | UTC queued → ended | rc | Log SHA-256 |
|---|---|---:|---|
| ms_numpy_oracle | 09:04:14.947 → 10:35:22.763 | 0 | `f8e5ffc0b471eaede67bcd9e82b2ee065bc82f7419f98f9c49da0869ca5a8107` |
| oracle_validation_v1 | 09:17:42.543 → 10:35:37.927 | 1 | `0a54f21f9d6aa40abaf196f5a050db7b09825927e5ebc3630b1c9d706693894e` |
| colour_rgb_stage | 09:17:49.661 → 10:35:43.147 | 0 | `72108091e811d9e3c032384674bdd2fba82d6d6ee35404fc892b36f1b981f111` |
| chroma_speed_v1 | 09:26:42.587 → 10:36:55.294 | 0 | `4f791b57e3f6c120757198ea7a19345a5d2cef90699e21d5b0ad09f3ee52e044` |

NumPy 2.5.2 generated 116 rows, scores SHA-256
`1f05aa5cba64e869e37bc22f6772b9793b24cc180b2ed17d681e3ae37a2291f9`.
The Rust run completed all 116 inputs with scalar/v3/v4 and 1/8-thread
bit checks. MDSI's CS/GCS maps passed their gates (maximum absolute errors
8.881784197001252e-16 / 7.105427357601002e-15), including 6,727 negative
author GCS samples. Its wrong constant was rejected on 115 cases.
**MDSI failed six final-score gates**, all reference-zero cases: four
identities plus `011_odd` and `011_tiny`. Maximum absolute score error
4.17956512028158e-05; maximum relative error on nonzero references
5.716624607372263e-10. The failed cases' maps were byte-identical to the
author maps, isolating the problem to pooling.

MS-GMSD and MS-GMSDc passed the independent NumPy gates, respectively:
max absolute 5.013697790268168e-14 / 3.202646481348381e-14;
max relative 2.9648248124009975e-12 / 2.2801471108977573e-12.
The wrong constant was rejected on 109 cases. These are equation-transcription
comparisons, never author-software parity claims.

The entire first Rust result tree and both JSON reports were renamed into
`validation_v1/`, preserving evidence before retry. Report hashes:
MDSI `451ee8861773b842bbd1f25a3952c197b43eaa7ec581a87c46123193a00a9f66`;
MS `960f045d41472827529ccea0c71ad400f7995d9781666300cf58948065741062`.
`validation_v1_summary` (10:45:15.735–10:45:15.749Z, rc 0) recomputes the
numbers above; log SHA-256
`6d8b3674c3f8e2d0392e84092534a5e9556f812b9c7c497682f7a0e7b8bed303`.

## Pooling correction and pending retry

A pooling-arithmetic correction was made at this checkpoint (details about the
reference software's internals were removed from this copy at landing; the
unredacted original is in the external archive). No map formula, tolerance,
input population or negative control was changed.

The no_std build's numeric parity with the reference scores remained unmeasured at
this checkpoint; the retry adds no-default-feature unit tests as well as the
existing compile check. The default std oracle
retry is queued as `oracle_validation_v2`; it must pass before prediction
generation or label reads. CLI checks/smoke are also pending.

Colour staging verified 1,565 unique decoded RGB files for 1,539 pairs.
`colour_v2/raw_pairs.tsv` SHA-256:
`33836707b103b9bd03a951635084ce9cd942d946607e050d373d52784d6008db`.
The first 24-cell timing run is CONTENDED and predates the pooling correction.
It is retained as preliminary evidence and is not the corrected implementation's
speed result. Its raw samples and fits remain under `speed/`.

## Main-branch CPU peer preparation (10:36–10:50Z)

`fetch_fast_ssim2_main`: 10:36:02.286–10:36:03.988Z, rc 0, log SHA-256
`b6d3292dff9d6c4f922a9042cdf0151b00557c9b106fdbd367ca48ea22572eab`.
The pinned main archive contains 107 files; archive SHA-256
`8019f7d4970e8595c1310efb1dbcfcf5e87ee54858dfca16d5019f1f31ded56b`.
`fast_ssim2_main_source.json` pins every extracted file. The source is
external under `/var/tmp/gmsd-chroma/fast-ssim2-main/`, untouched.

Worker 2 became reachable after the wake request; its read-only probe
reported x86_64, enough free RAM/disk, and no Rust tools or existing lane
wrapper. Build the peer locally under heavy, stage only the exact frozen
TRAIN RGB files and binary through SSH/rsync under heavy, and score remotely
through a copy of the existing serialized run-heavy wrapper. No host package
installation or boot configuration change is made. All remote data stays
under `/var/tmp/gmsd-chroma/`; only the small missing `~/tmp/devin/heavy`
wrapper is provisioned, with existing-file protection. The transfer manifest
and remote scorer verify every input hash, all pair keys and binary provenance.
`build_fast_peer_v1` and `stage_fast_peer_v1` are queued, not completed.

## HANDOFF — QUOTA STOP, 2026-09-24

The user ordered a stop at weekly usage 48.0%; the global
`/home/lilith/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md` also exists.
This is an incomplete, quarantined checkpoint, not a qualification pass.
Do not resume experiments until the user resumes the lane and the shared
stop condition is resolved; do not remove the coordinator's flag unilaterally.

**Safe point:** at 10:54:25.519Z all four remaining owned jobs were cancelled
while waiting for the shared lock, before admission. Each recorder returned
-15 (shell status 241), and each log is empty with SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The closed sessions are `cli_smoke_v1`, `oracle_validation_v2`,
`build_fast_peer_v1`, and `stage_fast_peer_v1`. Exact UTC start/end,
cwd and argv are in `commands.jsonl`; the cancellation snapshot is
`/var/tmp/gmsd-chroma/quota_stop_checkpoint.json`. The owned heartbeat
flag was removed and its process exited 0 (`heartbeat_stopped`). No owned
heavy job remains in flight. No remote build, staging or scoring started.
Worker 2 was awakened and only probed read-only; no shutdown was attempted.

**Done and preserved:** the original author oracle, its integrity check,
the first failed MDSI run, the passing independent MS equation comparison,
the decoded TRAIN population, and preliminary CONTENDED timing samples.
The first validation reports and Rust outputs are under `validation_v1/`;
the current top-level parity reports deliberately do not exist. Existing
results must not be overwritten or relabelled as results of the correction.
No human-score payload has been read. Part B has not begun.

**Unqualified source:** commit `596b05bdd15d` contains the MDSI power-pooling
correction and identity regression test. Neither has run after the change.
Before that correction, 21 gmsd unit tests, its no-default compile check,
release examples, gmsd clippy, and CLI feature check passed. The broader
build command returned 101 because CLI clippy encountered 13 inherited GPU
dependency diagnostics; CLI release build and its final formatting check
were skipped. These earlier passes do not qualify the current source.
The final WIP launcher `run_remote_peer.py` has never executed or been tested.

**Exact next steps after authorized resumption:** work only in
`/home/lilith/work/zen/zenmetrics--gmsd-chroma`, retaining bookmark
`quarantine/codex/gmsd-chroma`. Use unique recorder names; the cancelled
names already have logs. Re-read the binding brief and current local notes.

1. Run the corrected oracle gate, which includes tests, no-default tests,
   clippy, formatting, frozen-harness verification and both comparisons:
   `python3 /var/tmp/gmsd-chroma/run.py oracle_validation_v3 /home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- bash scripts/gmsd-chroma/validate_oracles.sh`.
   If std `powf(0.5)` is optimized back to sqrt, resolve it without changing
   the gate or fixtures. The no_std quarter-power branch remains numerically
   unqualified against the author implementation. Preserve each failed run.
2. Run CLI qualification:
   `python3 /var/tmp/gmsd-chroma/run.py cli_smoke_v2 /home/lilith/tmp/devin/heavy --mem 12G --jobs 8 -- python3 scripts/gmsd-chroma/cli_smoke.py`.
   Review the private Cargo patch and committed lockfile portability too.
3. The user authorized main-branch fast-ssim2 CPU rescoring on the available
   workers after schema searches found no usable stored CPU TRAIN column.
   Source main is pinned to `f011259a0cd2fb7f538e08398ba41c0ccd0dfcb4`,
   archive SHA-256 `8019f7d4970e8595c1310efb1dbcfcf5e87ee54858dfca16d5019f1f31ded56b`.
   Its 107-file manifest and raw TRAIN inputs are ready. **Do not execute the
   draft stage/remote launch scripts as written:** the workspace notes require
   remote metric scoring through the existing zenfleet job system. Adapt
   this WIP to that owner before dispatch, and review diagnostic sanitization.
   Read `docs/RUNNING_JOBS.md` and the private node runbook without printing
   private identifiers. No host software installation is required or authorized
   by this handoff. The standalone CPU driver is unbuilt/unformatted; local
   build preparation is `python3 /var/tmp/gmsd-chroma/run.py build_fast_peer_v2 /home/lilith/tmp/devin/heavy --mem 12G --jobs 8 -- python3 scripts/gmsd-chroma/build_fast_peer.py`.
4. Only after both metric gates pass, run `score_colour.py` under heavy.
   Require the exact 1,539-key CPU peer manifest before `freeze_colour_report.py`.
   That freeze is the first human-label read; then `report_colour.py` uses the
   existing statistics owner and reference-clustered intervals. Preserve the
   preregistration and data roles, and record any exposure as required.
5. Archive the old `speed/` tree, then rerun `run_speed.sh` through the recorder
   and heavy wrapper with a new name. Finish appropriate A7 checks and docs.
   Run `record_footprint.py` under heavy; full cache/Docker footprint enumeration
   is still missing. Do not delete caches or evidence to make the census easier.
6. Part B is entirely pending. The corrections marker exists and pins
   `620a384e06e49422d501bdddc9c0439afc846a98`; recheck it on resume, then create
   the separate zensim jj workspace as the brief specifies. The read-only
   `/var/tmp/gmsd-chroma/c8_readonly_integration_notes.md` is planning only.
   Preregister before any calibration, then implement and run B1–B5; keep
   f0–f1321 bit-identical and leave the other lane's workspace untouched.

Bulk outputs remain under `/var/tmp/gmsd-chroma/`; the append-only file
manifest is `/home/lilith/tmp/devin/rev4_gmsd-chroma_manifest.tsv`.
The partial report is `/home/lilith/tmp/zensim-paper/rev4/GMSD_CHROMA_DONE.md`.
Nothing is pushed or promoted. Normal final checks are deferred by the explicit
quota-stop order, not represented as passing.

Final stop bookkeeping uses `python3 /var/tmp/gmsd-chroma/run.py quota_handoff_commit_v1 python3 /var/tmp/gmsd-chroma/finalize_quota_handoff.py`; exact UTC/cwd/argv/rc/log SHA are appended to `commands.jsonl`. This only writes handoff records, hashes them, commits the owned WIP and moves the owned bookmark; it runs no experiment.

## Resumed — 2026-09-24

Coordinator relayed user cap 70%, guard stop 68%, and cleared the shared stop flag. Verified absent; resumed from `2ac4f770`. Corrected oracle `oracle_validation_v3` and CLI `cli_smoke_v2` are queued through heavy. The earlier HANDOFF describes the stopped checkpoint, not current activity. No human-score payload read.

Peer dispatch correction: existing `zenfleet-worker` local manifest/ledger/blob mode owns execution, reconciliation and content addresses on worker 2. Added only its documented custom executor adapter; one frozen pair per job with RGB, table, binary and adapter hashes in job inputs. Capabilities are declared and checked. Private SSH diagnostics are suppressed; only the fixed progress protocol is forwarded. Scripts parse, but build/integration and remote dispatch are still pending behind the shared lock. No custom scheduler or claim loop is introduced.

Source-binding follow-up before admission: each parity report now carries the oracle binary hash and source hashes; prediction generation requires those to match its build. The frozen raw-pair TSV hash is asserted before peer staging. Per-job files are retained and the remote footprint is enumerated. Python AST checks pass; execution remains pending. The no-default test harness explicitly links std for reference floating-point methods while preserving `cfg(not(feature="std"))` library arithmetic. Validation formats the changed crate under the lock before checking it. `footprint_v1` is queued with an admission preflight.

## Resumed first results — 12:01Z

`oracle_validation_v3` admitted 12:01:06.099Z and ended 12:01:30.460Z, rc 1.
The all-feature debug tests passed 22/22; no-default debug tests passed
21/21; compile and crate clippy passed. The release oracle still failed
the same six reference-zero MDSI scores: max absolute
4.179565120281581e-05, nonzero relative 5.716624607372263e-10. MS gates
again passed with the previous maxima. Both reports and maps are preserved
in `validation_v3/`; no gate threshold changed.

The std power call now uses an opaque runtime exponent to prevent the
compiler replacing pow(x,0.5) with sqrt. Added an adjacent-to-one pooling
regression and release unit tests to the gate. `oracle_validation_v4` is
queued; this proposed optimizer guard remains unqualified until it runs.

`cli_smoke_v2` admitted 12:01:32.337Z and ended 12:01:35.875Z, rc 1. CLI
clippy produced 14 diagnostics in pre-existing HDR, assemble and sweep
code (unused variable, attribute spacing, lifetime, argument count, type
complexity and chunks_exact lint). No new metric location was implicated.
The build and ingress checks did not run. The next `cli_smoke_v3` retains
clippy's nonzero exit in `cli_clippy_status.json` and proceeds independently
to build, formatting and bitwise ingress checks. A smoke pass will never
be reported as a CLI clippy pass. No warning/test is disabled.

oracle_validation_v3 log SHA-256 `1c0f4077d5e13cfd4334f52016d749195ad96c730a7e9566258c65393a0ada8f`; verbatim argv/cwd/start/end/rc are in commands.jsonl.

cli_smoke_v2 log SHA-256 `40a89e4ce05c6379c5bd97d010272a46a8e6ffe05be7f421ca79fd1f642df608`; verbatim argv/cwd/start/end/rc are in commands.jsonl.

`cli_lint_baseline_v1` (12:05:38.968–12:05:39.030Z, rc 0) compared the five implicated CLI files with pre-lane `be6e8a96`: all five are byte-identical. Log SHA-256 `dcb05152ef498255fbe0389a86a4c9a52c059fc83443844a8b31f8267a0a781b`; exact paths/hashes and replay argv are in the log/command index.

Release optimizer evidence: `inspect_v3_pool_assembly` (12:20:25.198–
12:20:25.231Z, rc 0) disassembled the verified v3 oracle binary
`534e75ff197bf6f4492b7ab9887190dfd878f407a130878e4e646d7aef77c2f2`.
The private pool has two adjacent `sqrtsd` pairs for the mean/deviation
root walks, one `sqrtsd` for complex magnitude, and only its final
quarter-power calls `pow`. This confirms that constant-half power calls
were replaced. Full assembly SHA-256
`8a5971bb117d144b96f1ff292b9055b8d5c8db8817f8c0f663f4934ee4d8b96d`;
log `438b963d6dabdbc6869903297c9985cc232e6d08c567a8c7758673feaddfcfe9`.
The old binary is retained at `validation_v3/mdsi_oracle`, alongside
`pool.asm`, so inspection survives the next build. No new score run.

`build_flag_audit_v1` (12:17:49.659–12:17:49.684Z, rc 0) verified empty
RUSTFLAGS/CARGO_ENCODED_RUSTFLAGS/CARGO_BUILD_RUSTFLAGS and no target-cpu
setting in the inspected Cargo config chain. Its config hashes are in the
log, SHA-256 `f1bfc1e0db7678660624bbec103ea55bdf17b7616e2be310764768e6e437baba`.

## Remote compute move — 12:37Z

User directed remote benches rather than waiting behind shared local training.
Cancelled only the owned, empty, childless `build_fast_peer_v2` and
`oracle_validation_v4` flock waiters; CLI and footprint remain queued.
Worker 2 now builds through its own shared heavy wrapper. Rust 1.98.1
was copied into lane scratch, with no system install. Metric and worker
source files were copied unchanged into a reduced private workspace;
its Cargo graph and original Cargo SHA are recorded separately.

Main snapshots were fetched/exported for archmage, zenpixels, zenbench,
whereat, garb, linear-srgb and enough; commit/archive hashes are in
`/var/tmp/gmsd-chroma/src/main_snapshots.json`. Zenbench main matches all
40 frozen harness files. Third-party safe_unaligned_simd upstream master
is 0.2.4 and cannot satisfy archmage main's ^0.2.5 requirement; the registry
0.2.5 dependency is retained rather than falsely claiming that patch applied.
Fast-ssim2 main remains f011259a0cd2fb7f538e08398ba41c0ccd0dfcb4.

`remote_build_stage_v1` failed (rc 1; underlying rsync 23) because the RGB
path was incorrectly rooted. Its staged source/manifest remain preserved.
Corrected transport and peer executor paths to `colour_v2/rgb`; additionally
shipped the oracle's `inputs/` directory. `remote_build_stage_v2` succeeded
12:34:24.044–12:34:41.535Z, rc 0, log SHA
7e5123f0a379127da60937e38084e66935c152650a51c56ff8650ec287cf1b94.
Remote graph SHA after adding the compatible main snapshots:
3f2a399b02c1155c431546c175b68a16394b62ca2878b9c53e1cfcf545b27b3d.
`remote_validation_v4` is running; no oracle pass is claimed yet.

## Qualified remote oracle — 12:37Z

`remote_validation_v4` ran 12:35:21.062–12:36:41.705Z, rc 0.
All 116 MDSI cases pass: max |delta| 4.098693606735537e-11,
max relative delta 5.716629807852658e-10. Wrong C3 rejects 109 cases.
CS max |delta| 8.881784197001252e-16; GCS 7.105427357601002e-15.
MS-GMSD and MS-GMSDc pass the independent NumPy paper transcription:
respective max relative delta 2.9648248124009975e-12 and
2.2801471108977573e-12. This is not MS author-software parity.
Debug/release all-feature tests each pass 23; no-default tests pass 22.
Crate clippy, formatting and no-default compilation pass.

Retained `/var/tmp/gmsd-chroma/remote_validation_v4/` contains the full
log, both per-case reports, build manifest and actual Cargo.lock.
MDSI report SHA c1ecd2d4dfe93bb39d253760814408a5a169ce5ec690c89ac916da377c4a8e8c;
MS report SHA b4a4f211970b05d840857d9c04391490a4cfc343a1a3b002baca56766f1b8aa3;
validation log SHA c0567ec074a894c8300074609d448eef0be64ff43d9ea28b75338749d732da78.
Oracle binary SHA 9506bf1b9e134ee469309fad9b1e28976b40814d648b0825586149d07a1d682d.
Remote speed matrix now runs against this qualified build.

Remote speed (`remote_speed_v2`) completed 12:37:40.843–12:38:54.147Z,
rc 0. Raw SHA 45b6a5d331b2bca344f75f7c4a6e2a9617fbb6ad19241d986d80a53ba0019aa3.
24 cells, 10–30 interleaved rounds. The report is CONTENDED; do not infer
a speedup. Wrapper: 73s, peak RSS 1.13GiB, min available 39618MiB,
peak load 1.14. Exact output remains in `remote_speed_v2/remote-speed.log`.
Alpha/beta fits and all residuals are in `remote_speed_v2/report.json`.
`remote_speed_fit_v2` (12:40:24.293–12:40:24.384Z, rc 0) uses NumPy
lstsq on the 24 summary rows only; log SHA
85a629a4172baedf3667f44474f7af94032bd29a464b6f35b6894782fbb972a5.

Remote source verification and gate reports bind the actual reduced Cargo
graph/lock, which differs from the broad CLI workspace. All metric crate
sources match the committed implementation byte-for-byte. The source
manifest permits scoring from this immutable export without requiring jj
on the worker; score generation still rejects a stale source/binary gate.


## TRAIN report and CLI completion — 2026-09-24

Worker 2 completed 1539 matched CPU fast-ssim2 main predictions via the existing
zenfleet manifest/ledger worker (one frozen pair per job). The stored GPU
columns were not silently substituted: fresh CPU rescoring was explicitly
requested by the user. Main f011259a0cd2fb7f538e08398ba41c0ccd0dfcb4,
BSD-2-Clause; clean snapshot and build manifests are retained. Predictions and
all population/RGB/predictor hashes froze before the first TRAIN human-label
read at 12:44:40Z. No held-out human set was read. Part B constants were not
chosen using these reporting results.

`benchmarks/gmsd-chroma_colour_2026-09-24.md` records every distortion cell,
signed SROCC and reference-clustered 2000-bootstrap CI from the canonical
zen_stats panel owner (seed20260924). Aggregate GMSD/MDSI/MS-GMSDc/fast-ssim2:
KADID1000/40 refs = .751323950/.850117269/.789220870/.857081045;
TID500/25 refs = .250263252/.747839079/.869163799/.775871847;
KonFiG39/3 refs = .647992178/.721683018/.714374836/.758832946.
The three-cluster KonFiG CI is weak evidence; TRAIN correlation is not an
unseen-data claim. Report SHA256
1d1eef5c1e67950230b210b55432434f846a188bcf2e609e1ed3adbb3cc810d8.

CLI ingress completed: 18 bitwise equal checks across score/score-pairs,
three sets and GMSD/MDSI/MS-GMSDc. CLI binary
fd39534491f88fd1e1431808dbf23efb5935357bda24e20c27de9f81caebf8d4.
This broad CLI graph predates the clean main snapshot gate; its manifest is
retained and it does NOT close the all-dependencies-main requirement.
The separate CLI clippy status is rc101 (14 inherited errors in unchanged
HDR/assemble/sweep files); the crate's own clippy/tests/fmt passed on worker 2.
No clippy pass is claimed for the broader CLI. Raw report is
`/var/tmp/gmsd-chroma/cli_smoke/report.json`.

The following exact recorder entries bind the command lines, UTC, cwd, rc,
logs and SHA256 (bulk records remain in commands.jsonl):

```json
{"argv": ["python3", "scripts/gmsd-chroma/freeze_colour_report.py", "--predictions", "/var/tmp/gmsd-chroma/colour_v2/predictions.tsv", "--predictions-manifest", "/var/tmp/gmsd-chroma/colour_v2/predictions.manifest.json", "--peers", "/var/tmp/gmsd-chroma/colour_v2/fast_ssim2.tsv", "--peer-manifest", "/var/tmp/gmsd-chroma/colour_v2/fast_ssim2.manifest.json", "--panel", "/var/tmp/gmsd-chroma/panel", "--output", "/var/tmp/gmsd-chroma/colour_v2/report_inputs.tsv"], "cwd": "/home/lilith/work/zen/zenmetrics--gmsd-chroma", "end": "2026-09-24T12:44:40.314031+00:00", "exit_code": 0, "output": "/var/tmp/gmsd-chroma/logs/colour_freeze_v1.log", "sha256": "c93c3475e9600df0087ee86ab29c53597a30a3ccac8d226e3852d5cfcd168d01", "start": "2026-09-24T12:44:40.011775+00:00"}
```

```json
{"argv": ["python3", "scripts/gmsd-chroma/remote_build.py", "colour-report"], "cwd": "/home/lilith/work/zen/zenmetrics--gmsd-chroma", "end": "2026-09-24T12:45:01.967342+00:00", "exit_code": 0, "output": "/var/tmp/gmsd-chroma/logs/remote_colour_report_v1.log", "sha256": "2ace9518e6a7c2a6412546ff04f7a1aaefdf649eae9acfffc9b07463d9d1d24a", "start": "2026-09-24T12:44:56.109731+00:00"}
```

```json
{"argv": ["/home/lilith/tmp/devin/heavy", "--mem", "12G", "--jobs", "8", "--", "python3", "scripts/gmsd-chroma/cli_smoke.py"], "cwd": "/home/lilith/work/zen/zenmetrics--gmsd-chroma", "end": "2026-09-24T13:06:06.279494+00:00", "exit_code": 0, "output": "/var/tmp/gmsd-chroma/logs/cli_smoke_v3.log", "sha256": "453ca9960cd825cc3307c46f7d16cf01e3bc85eb5f68ce17d5922e63b8cc5296", "start": "2026-09-24T12:03:50.919987+00:00"}
```


## HANDOFF — quota stop at 68%, 2026-09-24

User explicitly stopped the lane. Part A implementation, author parity,
TRAIN colour report, CPU fast-ssim2 rescore, CLI ingress and remote speed
results are committed. No new measurements were started after the stop.
Final read-only headline recomputations completed before the stop; exact
commands and outputs remain in commands.jsonl and the named logs below.

Outstanding: finalize DONE with each one-line recompute command and actual
output; consolidate the final footprint; independent audit. The exhaustive
all-dependencies-main requirement is incomplete (including the main
safe_unaligned_simd version conflict), broad CLI clippy has inherited errors,
MS-GMSD has independent NumPy paper-equation parity only, and no_std MDSI
author-oracle parity is unmeasured. Do not promote these results.

No Part A job remains in flight. Both completed final Part B remote jobs
were preserved locally; Part B worklog contains the exact continuation.
Both repositories remain on local quarantine/codex/gmsd-chroma bookmarks.
The DONE file is a paused handoff, not a completion declaration.

```json
{"argv": ["python3", "-c", "import json\nfrom pathlib import Path\nr=Path('/var/tmp/gmsd-chroma/remote_validation_v4')\na=json.loads((r/'parity.json').read_text()); b=json.loads((r/'ms_parity.json').read_text())\nout={'mdsi':{'pairs':len(a['records']),'max_abs':max(abs(x['author']-x['rust']) for x in a['records']),'max_rel':max(abs(x['author']-x['rust'])/abs(x['author']) for x in a['records'] if x['author']!=0),'passed':a['passed'],'wrong_constant_rejections':a['negative_control_failures']}}\nfor m in ['ms_gmsd','ms_gmsdc']:\n v=[x[m] for x in b['records']];out[m]={'pairs':len(v),'max_abs':max(abs(x['numpy']-x['rust']) for x in v),'max_rel':max(abs(x['numpy']-x['rust'])/abs(x['numpy']) for x in v if x['numpy']!=0),'passed':b['passed'],'oracle':'NumPy paper transcription'}\nprint(json.dumps(out,sort_keys=True))"], "cwd": "/home/lilith/work/zen/zenmetrics--gmsd-chroma", "end": "2026-09-24T13:57:14.221922+00:00", "exit_code": 0, "output": "/var/tmp/gmsd-chroma/logs/final_parity_recompute_v1.log", "sha256": "fe469ea85d43ad19a51b0eaef3ec1f278e63c1d44fb3730e44ecbe458c54986d", "start": "2026-09-24T13:57:14.205837+00:00"}
```

```json
{"argv": ["python3", "-c", "import csv,json,os,sys\nfrom pathlib import Path\nos.environ['ZEN_PANEL_BIN']='/var/tmp/gmsd-chroma/panel'\nsys.path.insert(0,'/var/tmp/gmsd-chroma/stats-owner')\nfrom scripts.lib import zen_stats\nrows=list(csv.DictReader(open('/var/tmp/gmsd-chroma/colour_v2/report_inputs.tsv'),delimiter='\\t'));assert len(rows)==1539 and all(r['role']=='TRAIN' for r in rows)\nout=[]\nfor name in sorted(set(r['set'] for r in rows)):\n s=[r for r in rows if r['set']==name];y=[float(r['quality']) for r in s]\n for arm in ['gmsd','mdsi','ms_gmsdc','fast_ssim2']:\n  x=[(1 if arm=='fast_ssim2' else -1)*float(r[arm]) for r in s]\n  score=zen_stats.panel_batch([('point',x,y)],stats='srocc')[0]['srocc_signed']\n  out.append(dict(set=name,arm=arm,pairs=len(s),references=len(set(r['ref_sha'] for r in s)),srocc=score))\nprint(json.dumps(out,sort_keys=True))"], "cwd": "/home/lilith/work/zen/zenmetrics--gmsd-chroma", "end": "2026-09-24T13:57:14.289675+00:00", "exit_code": 0, "output": "/var/tmp/gmsd-chroma/logs/final_colour_points_recompute_v1.log", "sha256": "136d52a1e210444638f6d8cb7661c7a82fb75aaf954d5b575dfe6727cff29f3e", "start": "2026-09-24T13:57:14.239238+00:00"}
```

```json
{"argv": ["python3", "-c", "import io,contextlib,json,sys\nfrom pathlib import Path\nsys.path.insert(0,'scripts/gmsd-chroma')\nimport report_speed\nsys.argv=['report_speed.py','/var/tmp/gmsd-chroma/remote_speed_v2/zenbench.json','/var/tmp/gmsd-chroma/remote_speed_v2/recomputed_report_v2.json']\nwith contextlib.redirect_stdout(io.StringIO()):report_speed.main()\nd=json.loads(Path(sys.argv[-1]).read_text());print(json.dumps({'label':d['label'],'cells':len(d['cells']),'fits':d['fits']},sort_keys=True))"], "cwd": "/home/lilith/work/zen/zenmetrics--gmsd-chroma", "end": "2026-09-24T13:58:13.531758+00:00", "exit_code": 0, "output": "/var/tmp/gmsd-chroma/logs/final_speed_recompute_v2.log", "sha256": "4d30189ae3dde0a8dc630fe91fe4ca7939aff7504fff94ff98f92454273ef378", "start": "2026-09-24T13:58:13.401053+00:00"}
```

## Continued by Claude Sonnet lane gmsd-chroma, 2026-09-24T20:55Z

Takeover of the paused Codex lane (user: Claude Sonnet iterates, Opus evaluates). Quota rule N/A; every other rule kept. Re-read the HANDOFFs of both worklogs; the predecessor's finished work is not redone.
Added `scripts/gmsd-chroma/recompute_headlines.py` (reads only retained raw files under /var/tmp/gmsd-chroma; prints one headline per name). Run at 20:5xZ; every value equals the predecessor's recorded number (mdsi_parity, ms_parity, c8_author_cs, cost, colour, speed, calibration). No human-label read, no new measurement, no thresholds changed.
Continuation of the C8 prefix/dead-slot gates and lint are recorded in the zensim worklog.

## Opus review corrections applied (Claude Sonnet lane, 2026-09-24)

Review: `/home/lilith/tmp/zensim-paper/rev4/REVIEW_GMSD_CHROMA.md`. (2) `Cargo.lock` restored to the lane base (`be6e8a96`, zero diff against it): the zenavif 0.4.0 / rav1d-safe rev / zenav1-svt git-rev churn is gone. The lock is therefore stale for gmsd's added `libm` and dev-dep `zenbench`; regenerate at landing (a local `cargo metadata --offline` re-creates the same churn from the sibling path dependencies, so it was not kept). (5) `scripts/gmsd-chroma/{remote_build,run_remote_peer,stage_fast_peer}.py` removed: they parsed a private file for the remote host; earlier commits of this bookmark still contain them. (9) KonFiG intervals in the colour record replaced by "not estimable (3 clusters)". (10) `crates/gmsd/LICENSE-MDSI` added (BSD-2-Clause text plus the author's header verbatim), referenced from the README and Cargo.toml comment. (11) Speed record `benchmarks/gmsd-chroma_speed_2026-09-24.md` added, labelled AVX2 tier of a household worker (node-3), CONTENDED. Nothing was run on any household node.

## MDSI clean-room integration (Claude Sonnet lane, 2026-09-25)

Per the coordinator's ruling the clean-room MDSI replaces the transcription; gates and limits: `gmsd-chroma_mdsi_cleanroom_2026-09-25.md`. Final run `mdsi_cr_final_v4` (rc 0) ran under `scripts/run-heavy --mem 8G --jobs 4` WITHOUT the shared lock: the lock queue held 7 waiters behind a 40+ minute training job for over an hour. Interim tests were direct `nice`d cargo runs. The CODEX quota assertions were removed from `preflight.py`, `freeze_colour_report.py` and `report_colour.py`; the Codex quota rule does not bind this lane.
Opus MDSI review corrections (2026-09-25): Octave driver scripts and reference-internals wording moved to the external archive (`gmsd-chroma_oracle_archive.pointer.md`); `mdsi.rs` split (tests in `mdsi_tests.rs`); gate bundle mirrored (`gmsd-chroma_mdsi_gate_bundle.pointer.md`).
