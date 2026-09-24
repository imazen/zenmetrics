# CVVDP SafeSyn landing worklog (2026-09-24 UTC)

Workspace: `/home/lilith/work/zen/zenmetrics--cvvdp-land`. Base:
`master@origin` `b50cde1f`. Source stack: `9f36f88b..5b035bc9`.
The original `quarantine/devin/cvvdp-safesyn` and
`quarantine/codex/cvvdp-harvest` bookmarks are retained. No push.
All bulky outputs and the Cargo target are under `/var/tmp/cvvdp-land/`.
The reviewed DONE file was not edited; r5600g teardown remains the
coordinator's open item.

## Rebase and conflict resolution

At 06:26:56–06:26:56 UTC (jj operation timestamps), from the original
`zenmetrics--cvvdp-safesyn` workspace,
`jj workspace add ../zenmetrics--cvvdp-land -r 'master@origin'` created the new
workspace. At 06:26:59–06:26:59 UTC in the new workspace,
`jj duplicate '9f36f88b::5b035bc9' --onto 'master@origin'` copied the nine
commits onto master without moving either original bookmark. jj reported
`(conflict)` on all nine copies. `jj new qtmsrzwx; jj resolve --list` at
06:27:04 UTC reported only `Cargo.lock    2-sided conflict`. The resolution
commands at 06:27:18 UTC were
`jj restore --from 'master@origin' -- Cargo.lock; jj squash` from the child
of the first copied commit. jj reported `Rebased 8 descendant commits` and
`Existing conflicts were resolved or abandoned from 9 commits`. This kept
master's complete lock. A subsequent diff from master to the rebased tip
listed no `Cargo.lock` or master CVVDP source changes.

The copied first commit adds to master:

- `crates/zenfleet-core/src/job.rs`: additive display-string JobId/routing test.
- `crates/zenmetrics-cli/Cargo.toml`: exposes its existing `zencodec` dependency
  through the `avif` feature for the opt-in decoder.
- `crates/zenmetrics-cli/src/decode.rs`: retains the master default decoder body
  byte for byte (except its function name); routes to the tagged zencodec
  decoder only when `ZEN_JOBEXEC_AVIF_DECODE=zencodec`.
- `crates/zenmetrics-cli/src/jobexec.rs`: display-scoped CVVDP ScoreFile
  metrics and opt-in pixel stamps.
- `Cargo.lock`: resolved to master, with any needed Cargo resolution to be
  checked by the fresh build.

The eight harvest commits add only their benchmark records and scripts.
`jj diff --from 'master@origin' --to d2d02532 --summary` listed those
11 benchmark files plus the four source files above, and no CVVDP files.
The default-decode static comparison returned:
`default_decode_body_matches_master= True` and
`master_body_chars= 3056 rebased_body_chars= 3064` (the eight-character
function-name suffix accounts for the length difference).

## Build and verification

`env CARGO_TARGET_DIR=/var/tmp/cvvdp-land/target CARGO_BUILD_JOBS=4
~/tmp/devin/heavy --mem 16G --jobs 4 -- cargo build -p zenmetrics-cli
--release --no-default-features --features
jobexec,png,jpeg,webp,avif,jxl,cpu-metrics,hdr` was queued at 06:28:18 UTC
and ran after the shared lock cleared. The `/var/tmp/cvvdp-land/build.log`
last line is `run-heavy: done rc=0 123s | peak-RSS 1.06GiB |
min-avail 15797MiB | peak-load 5.73`; Cargo reported `Finished release
profile ... in 2m 02s`. The new binary is
`/var/tmp/cvvdp-land/target/release/zenmetrics`, SHA-256
`fed67236f4e62fe8833684f8ebe97be3c1103af9dd35a1298747e540957b5d08`.
Build log SHA-256:
`c3d54715d2b82f3dc075c70d807344147d1bfbf8eb11376e0c87c087b88faac7`.

Cargo changed the local `Cargo.lock` during this build. A package-keyed
comparison with `master@origin` found 11 removed and 15 added package
identities, involving moved sibling source checkouts (including zenavif
0.1.7→0.2.0, jxl-encoder 0.3.2→0.4.0, and newer git pins). The new
`avif = [..., "dep:zencodec"]` activation needs no new lock package or
dependency entry: master's lock already lists `zencodec` under `zenmetrics-cli`.
The landing commit will restore
the exact master lock after local build/test verification. Tests run in this
workspace will use Cargo's current path-dependency resolution, which is
documented here rather than misattributed to the landing change.

The first fresh-binary smoke ran under `~/tmp/devin/heavy --mem 8G --jobs 4
-- /var/tmp/cvvdp-land/run_smoke.sh` at 06:53:31 UTC, exiting 1 after 16 s
(`run-heavy: done rc=1 16s | peak-RSS 0.42GiB | min-avail 16779MiB |
peak-load 2.35`). The r3 reviewer 24-pair job emitted 96 rows. The checker
found one score-bit mismatch against both the published fleet sidecar and
reviewer fleet rows: `zenjxl-e7/q80.jxl`, `cvvdp@standard_fhd`, new
`9.734886169433594` versus fleet `9.73488712310791`, absolute delta
`9.5367431640625e-07` (one f32 ULP). All other 95 scores match bitwise;
all 96 reference/distorted pixel stamps match the reviewed fleet rows.
The raw output was preserved as
`/var/tmp/cvvdp-land/glibc_smoke_r3_rows.jsonl` SHA-256
`a40170dfb26dca34de5365a0eb2b03a72e42a18dd072bac5678e33f4c1f8f9c6`;
smoke log SHA-256
`519902dec04da50dbff2b3fde8456e59bb0ba3fd1ec192f8d32a97c5e4f54de6`.
The script stopped at this strict gate, so the second 24-pair set did not run
in this first pass. A read-only scan of all 96 rows produced the mismatch
counts above. No tolerance was applied and no master CVVDP code was removed.
This binary is glibc-linked; the fleet executor was static musl. A fresh
musl-target build with the fleet feature set was run through `heavy` to
test whether that build-target difference explains the one-ULP result.

The fresh musl build ran `env CARGO_TARGET_DIR=/var/tmp/cvvdp-land/target
CARGO_BUILD_JOBS=4 ~/tmp/devin/heavy --mem 16G --jobs 4 -- cargo build
--release --target x86_64-unknown-linux-musl -p zenmetrics-cli
--no-default-features --features
jobexec,png,jpeg,webp,avif,jxl,cpu-metrics,hdr` and exited 0 after 122 s.
`/var/tmp/cvvdp-land/musl_build.log` SHA-256
`c027e78b3cd93b46a957ca86d6975fc744068d1a83ae8212cb9ba5245c2fb2da`;
fresh static binary SHA-256
`92c362f104a50e2a5455fbe434723b4abdc4de27ce2767e53f000644a5c35e31`.

`env CARGO_TARGET_DIR=/var/tmp/cvvdp-land/target CARGO_BUILD_JOBS=4
~/tmp/devin/heavy --mem 16G --jobs 4 -- cargo test -p zenmetrics-cli
--release --no-fail-fast` queued at 06:47:39 UTC and ended 07:25:09 UTC.
`/var/tmp/cvvdp-land/test.log` SHA-256
`058002d635697fa110afe86e661b2dbd0a2af6a707530b48ec426cac608c2d9c`
ends `run-heavy: done rc=0 159s | peak-RSS 1.05GiB | min-avail
15543MiB | peak-load 5.78`. All suites ran: 108 passed, 0 failed, 4
existing ignored. In particular, `tests/avif_hdr_tripwire.rs` reports
`test result: ok. 5 passed; 0 failed; 0 ignored`.

`env CARGO_TARGET_DIR=/var/tmp/cvvdp-land/target CARGO_BUILD_JOBS=4
~/tmp/devin/heavy --mem 16G --jobs 4 -- cargo clippy -p zenmetrics-cli
--release --all-targets --features jobexec` queued 06:47:50 UTC and ended
07:25:30 UTC. `/var/tmp/cvvdp-land/clippy.log` SHA-256
`1f0dc5a3bbe1c83a76a60d88475aaf68d848d98830c7e6b405083f60296f98bd`;
`run-heavy: done rc=0 21s`. Clippy reported warnings, including
pre-existing CLI lint warnings; this was a successful lint run, not a
zero-warning run.

`env CARGO_TARGET_DIR=/var/tmp/cvvdp-land/target ~/tmp/devin/heavy
--mem 4G --jobs 4 -- just fmt-check` queued 06:48:03 UTC and ended
07:25:32 UTC. `/var/tmp/cvvdp-land/fmt.log` SHA-256
`ad3b407f2b3d1c47b8c21f025ce305a3d4bf960e011bc23983790cae48f8334b`;
`run-heavy: done rc=0 1s`. This uses zenmetrics' sibling-safe
`just fmt-check` recipe rather than `cargo fmt --all`.

After the Cargo builds and checks, `jj restore --from 'master@origin' --
Cargo.lock` restored the exact master lock. `cmp -s Cargo.lock
/var/tmp/cvvdp-land/master_Cargo.lock` exited 0; `Cargo.lock` SHA-256 is
`67cc2f11b7554e27abfcd0a543c012062a3f8997b10a53f16273d6cb68ff9e00`.
The only working-copy change afterward was this worklog.

The static musl binary was run through `env
ZENMETRICS_BIN=/var/tmp/cvvdp-land/target/x86_64-unknown-linux-musl/release/zenmetrics
~/tmp/devin/heavy --mem 8G --jobs 4 -- /var/tmp/cvvdp-land/run_smoke.sh`.
Queued 07:28:12 UTC; it acquired the lock at 08:16:00 UTC and ended 08:16:08 UTC.
`/var/tmp/cvvdp-land/musl_smoke.log` SHA-256
`a538e09b58695ae24aa36a3fd80983cff29e566a32cb65ed36a886e3a1f51d95`;
`run-heavy: done rc=1 8s`. The first reviewed 24-pair job emitted 96 rows:
`bit_mismatches=1 pixel_hash_mismatches=0`. The same
`zenjxl-e7/q80.jxl` `cvvdp@standard_fhd` cell was new
`9.734886169433594` versus fleet `9.73488712310791`. All other 95 score
cells matched bitwise. `smoke_r3_rows.jsonl` SHA-256 is the same
`a40170dfb26dca34de5365a0eb2b03a72e42a18dd072bac5678e33f4c1f8f9c6`
as the glibc run, and `cmp -s` exited 0: the glibc and musl JSONL outputs
are byte-identical. The second 24-pair set did not run because the strict
first-set gate stopped the script. No source code or scoring tolerance was
changed to conceal this regression. Exact SafeSyn fleet score parity is
unmet on the rebased master code, so the landing is **not ready**.
