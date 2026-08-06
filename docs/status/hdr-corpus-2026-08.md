# HDR multi-codec corpus build — living status

**Living doc.** Updated continuously by the build's driver/harvest loops while the
run is in flight; it is not a terminal report. The terminal record is the zensim
campaign's registered appendix S plus the corpus `_MANIFEST.json`.

- **Owner:** `claude-hdrcorpus` lane.
- **Started:** 2026-08-05.
- **Mission:** build the HDR phase-2 corpus — HDR sources × multiple codecs ×
  a sweep-discipline grid, with cvvdp-mix targets, scored fleet-parallel.
- **Why:** the existing HDR training leg (`hdr_v3mix` @944) is **single-codec**
  (zenjxl only). The zensim campaign registers this as gap **Q-G6**, and models
  trained on it are codec-blind on HDR. Multi-codec breadth is the named lever.
- **Orchestration:** the zenfleet job system only (workspace mandate — no
  hand-rolled fleet scripts).

---

## Current state

| | |
|---|---|
| **Phase** | **LANE RESTARTED 2026-08-06 ~20:00Z** (`claude-hdrcorpus2`) — prior lane landed B6 + appendix S (`cf594877`) then died before building; this lane owns B1-B4/B7 + the build |
| **Blocking on** | B1 (no HDR encode job), B2 (diffmap executor), B3/B4 (codec arms), B7 (image) — all in flight this lane |
| **Ledger fill** | n/a (not declared yet) |
| **First-cell R2 persistence gate (G-S1)** | NOT YET RUN — hard gate before any fleet scale-up |
| **avifgen coordination** | their encode DRAINED 16:50Z (~565k cells); their **GPU ScoreFile run `avifgen-sf-gpu-20260806` declared 20:01Z** — node-2+lianli GPU queue is THEIRS for ~32 h. This lane's GPU score declares queue BEHIND it; encodes use the freed CPU fleet. Their CPU metric declare (`avifgen-sf-cpu-20260806`) is scripted but not yet visible on R2. |

### Build plan (restart, measured against the avifgen reusables)

Reuse verbatim: Encode→ScoreFile two-stage (persistence by construction), `pairs_from_encode_ledger.py` bridge, `declare_direct_objects.py` ZEN_FULL_URI=1, `writeback_scores.py` two-stage mode, executor-image build scripts (new TAGS), `ZEN_REQUIRE_GPU=1` score-env, G-Z2/G-Z3-shaped gates. jobexec's **ScoreFile `hdr:true` arm is already fully implemented** (incl. zensim feature-row emission from the PU21 feeding) — the scoring half needs images + declares only.

Genuinely new (this lane):
1. **B3+B4 (`sweep/hdr.rs`)**: an `HdrCodec` enum (Zenjxl / Zenavif / Zenav1Svt / JpegGainmap) confined to the HDR module — `CodecKind` untouched. svt arm = the budget harness shape (PQ RGB16 → BT.2020nc limited 10-bit 4:2:0 → `try_encode_frame_420_hbd`, still CQP) + zenavif-serialize mux (nclx: src primaries, transfer 16, matrix 9, limited; depth 10) → decode-back through the EXISTING `decode_avif_to_nits`. gainmap arm = the budget harness shape (PQ → linear f32 RGBA @203-nit white → `ultrahdr_rs::Encoder` HDR-only, gm q85 scale-4 defaults) + a bytes-based decode-back via the existing ultrahdr decode wiring. Deps: `zenav1-svt` @ `4c5c1324` (repo is PUBLIC — CI clones like other siblings), ultrahdr @ `73ab3d27` (already a path dep).
2. **B1**: `JobKind::Encode` gains `hdr: bool` with the exact `ScoreFile.hdr` append-only serde pattern (old job ids byte-stable); `parse_emit_cells`/`declare_encodes` carry it; jobexec's encode arm routes `hdr:true` through `decode_hdr_ref` + `encode_hdr`.
3. **B2**: Diffmap executor (survey of per-metric diffmap APIs first; declared over persisted encode shas like ScoreFile).
4. **B7**: rebuild executor images at the pins with `hdr,png,hdr-gainmap,hdr-encode-svt` features — canonical package, new tags.

Then: grid cells (S.4: 1,140 sources × 3 arms × 30 q), declare, **G-S1 first-cell gate**, fleet scale on CPU nodes, drain, bridge → score declares (GPU behind avifgen; cvvdp on CPU queue), writeback, `_MANIFEST.json` + orientation gate + triple mirror + provenance entries.

### Milestones

- [x] Standing-rule change recorded (node-2/node-3 permanently Ubuntu)
- [x] node-2 + node-3 PXE-flipped to Ubuntu and verified up
- [x] Fleet capability surveyed; blockers identified and registered
- [x] cvvdp GPU-vs-CPU verified; budget doc corrected in place
- [x] Appendix S grid pre-registered
- [x] **B6 fixed** — GPU scoring is now guaranteeable and self-describing
- [ ] zenfleet extended for HDR multi-codec encode + score (**B1/B2/B3/B4/B7**)
- [ ] Workers enrolled on all encode nodes
- [ ] **First-cell artifact-persistence gate PASSED** (encoded bytes + diffmaps + all metric variants confirmed on R2)
- [ ] Fleet scaled to full grid
- [ ] 50% ledger fill
- [ ] Corpus complete + `_MANIFEST.json` + orientation gate + provenance registered

## Blockers (registered as campaign appendix S §S.6)

| id | blocker | consequence |
|---|---|---|
| **B1** | zenfleet has no HDR **encode** job — `JobKind::Encode` carries no HDR flag; HDR encode exists only behind `zenmetrics sweep --hdr` and the jobexec encode path is hardcoded SDR | must extend zenfleet (not bypass) before any encode cell runs |
| **B2** | artifact persistence unmet on the scoring path — a `Metric` job **unlinks** the encode after decoding it, and `JobKind::Diffmap` has no executor | two-stage `Encode`→`ScoreFile` fixes encoded bytes; diffmaps still need an executor |
| **B3** | no `zenav1-svt` codec registered in zenmetrics (encoder itself is ready and byte-gated at 10-bit) | integration gap, not a codec gap |
| **B4** | JPEG-gainmap is decode-only; no gain-map encoder in any sweep path | arm cannot be built until an encoder arm is added |
| **B5** | AVIF arm is **user-halted** pending a zenavif settle-check; checked 2026-08-05 and zenavif is **not settled** (stale marker, uncommitted lockfile, conflicted HDR bookmark) | arm stays out; needs user confirmation, not a workaround. **Coordination note 2026-08-06 (avif-datagen lane): the user directive of 2026-08-06 lifted the avif-datagen halt; the SDR AVIF campaign runs at pinned zenavif `66e3c417` (zensim appendix Z, `docs/status/avif-datagen-2026-08.md`). Whether the lift extends to THIS lane's HDR AVIF arm remains this lane's call — the pin + directive text are there to reuse.** |
| **B6** | ~~the fleet cannot guarantee GPU scoring~~ — **RESOLVED 2026-08-05** | `ZENMETRICS_REQUIRE_GPU` drops the CPU rung; a `runtime` column records the rung that actually ran; the worker verifies a GPU at boot |
| **B7** | executor image must carry the `hdr` (and `png`) features | image rebuild, canonical name + new tag |

---

## Node roster and roles

Neutral IDs only (per repo privacy policy); identifying detail lives in the
private fleet repo.

| Node | Cores | RAM | GPU | Role in this build |
|---|---|---|---|---|
| `node-2` | 16 | 30 G | RTX 3070 8 GB | **GPU metric scoring** + encode |
| `lianli` | 24 | 29 G | RTX 2080 8 GB | **GPU metric scoring** + encode |
| `node-3` | 12 | 41 G | GTX 1660 Ti 6 GB | encode (GPU is spare capacity, not a directed scoring role) |
| `i265` | 20 | 30 G | none | encode |
| `ryzen5800xt` | — | — | — | encode (pending probe) |
| `tower` | 32 T | 62 G | none | encode only if it fits its rules: Docker-only, capped, media-first, observe-before-load |
| `wsl` | 28 | 58 G | — | operator — drives the fleet, does not measure |

**GPU metric scoring is directed to `node-2` and `lianli`** (user directive).

### Design consequence: the GPU metric queue is the binding constraint

The budget lane's headline finding is that **encode wall time is trivial**
(~3.4 min fleet-wide across the 76 HDR sources) while **the GPU metric queue is
the real cost** (~11.5 h). So the grid is sized against GPU-metric throughput on
`node-2` + `lianli`, **not** against encode capacity. Adding encode nodes buys
almost nothing; the levers that matter are metric-queue depth, per-cell metric
count, and total cell count.

---

## Source estate

- **Path:** `/mnt/v/output/imazen-26-hdr-grid-2026-06-14/`
- **1,140 HDR reference PNGs = 76 origins × 15 aspect-preserving scales**, 7.8 GB.
- Scale ladder spans ~`96x128` (tiny, ~12 kpx) to `3072x2304` (~7 MP) — a
  log-spaced size axis that already satisfies the sweep discipline's size
  requirement including the tiny bucket, so **no new resampling is needed**.
- Content classes by origin: nature 47, interiors 20, general 6, food 3.
- These are the same origins the existing single-codec HDR leg was built from,
  which keeps the new multi-codec corpus joinable to it.

---

## Incidents / decisions log

| When (UTC) | What |
|---|---|
| 2026-08-05 ~04:41 | Claimed `zensim` / `zenmetrics` / `homefleet`. |
| 2026-08-05 ~04:43 | Standing-rule change recorded: node-2/node-3 permanently Ubuntu; flip-back rule revoked. Public record neutral-ID-only. |
| 2026-08-05 ~04:44 | Both nodes found in their **other OS** with **no worker flags set** — i.e. the pre-directive default. Set both to boot the Ubuntu worker (procedure in the private fleet repo). |
| 2026-08-05 ~04:44 | The flip helper's own reboot step failed on both boxes ("couldn't reach…") — the **documented** gotcha for a box sitting in its other OS. Used the documented admin-account reboot fallback. Cleared known-hosts entries **before** polling, per the changed-key gotcha that has previously cost ~10 min of false "box never came up". |
| 2026-08-05 ~04:46 | Both nodes up in Ubuntu in ~75 s, GPUs enumerated, driver 595.84. |
| 2026-08-05 ~04:47 | Noted: the pre-existing `zensim/scripts/hdr/hdr_score_fleet.sh` is a **hand-rolled** fleet scorer predating the zenfleet mandate. Read as the HDR scoring contract, **not** extended — this build goes through zenfleet. |
| 2026-08-05 ~04:51 | **Boot-order misdiagnosis corrected.** A report that node-3's firmware had regressed to other-OS-first was investigated and not reproduced: both boxes read PXE first and both showed `BootCurrent` = the PXE entry. The real cause was that every per-box worker flag was cleared — PXE runs, finds no flag, chainloads the other OS by design. Verified with a full reboot cycle on node-3 (boot ID changed, still PXE, returned to Ubuntu unattended). **No firmware change was needed or made.** |
| 2026-08-05 ~04:52 | Yank-rule detection applied per the new user directive: node-3 had **no interactive session** (only a `manager`-class session and the operator's own SSH; no display server) ⇒ yanked. Detection method + observed state recorded in `NODES.md`. |
| 2026-08-05 ~04:54 | **cvvdp GPU claim verified.** The budget doc's `cvvdp-gpu == cvvdp-CPU` row was challenged as possibly CPU-vs-CPU. Re-measured with `nvidia-smi` sampled through each run: cvvdp-gpu allocates real device memory and shows nonzero utilization at every size ⇒ **the row is genuinely GPU and stands**. Peak GPU utilization is only 2-10%, which *measures* the CPU-prep-bound mechanism the row asserted. |
| 2026-08-05 ~04:54 | **Memory ceiling answered:** ~222 MiB/MP (cvvdp-gpu) and ~348 MiB/MP (ssim2-gpu) ⇒ ~1.6 GB and ~2.5 GB at the 7.08 MP top tier. **Every ladder size fits the 8 GB fleet cards**; no size tier is excluded from GPU scoring. |
| 2026-08-05 ~04:55 | **Silent-fallback defect confirmed (B6).** With the GPU hidden, explicit `--gpu-runtime cuda` exits 1 and refuses ("explicit backend requests never fall back"), but the **default** `auto` exits 0 and emits the **GPU column name** from a CPU computation. A code audit found jobexec hardcodes `auto` at four sites with no override, so the explicit-flag mitigation **does not reach the fleet path**. Budget doc annotated in place; raw probe data committed. |
| 2026-08-05 ~04:58 | Appendix S pre-registered in the zensim campaign (grid, persistence contract, gates, and all seven blockers stated as measured facts). |
| 2026-08-05 ~05:1x | **B6 FIXED.** `ZENMETRICS_REQUIRE_GPU=1` drops the CPU rung from the `auto` ladder in the one function all six ladder sites share, so it reaches the hand-run path, the sweep cache and jobexec alike. Failure is loud and names the reason. A `runtime` column now records the rung that actually executed. The fleet entrypoint gained `ZEN_REQUIRE_GPU=1`, which exports the flag and **verifies a GPU at boot** so a mis-scheduled box fails in one line instead of failing cells one at a time. |
| 2026-08-05 ~05:1x | Two adjacent defects found while verifying B6, both fixed. (1) The cvvdp **batch** scorer — the fleet's scoring path — skipped the liveness probe, so a GPU-less box **panicked** (exit 101) inside the CUDA driver instead of returning a catchable error; it now probes first, which is what makes the ladder and the gate behave the same there as on the single-shot path. (2) `score-pairs` exited **0 after scoring zero rows**, which on the fleet would mark a totally-failed cell Done; all-pairs-failed is now an error, with the empty parquet still written for diagnosis. |
| 2026-08-05 ~05:1x | **Scope correction to my own earlier claim.** The silent CPU fallback is **metric-dependent**, not universal: `ssim2-gpu` genuinely falls back (verified — exit 0, `ssim2_gpu` column, `runtime=cpu`), while `cvvdp-gpu` on the batch path has no CPU arm compiled and simply fails. The blanket statement "the fleet silently produces CPU numbers" was too broad; the accurate one is "some GPU metrics do, and nothing recorded which". Both halves are now covered. |
| 2026-08-06 ~20:00 | **Lane restart** (`claude-hdrcorpus2`). Prior lane died post-B6/appendix-S with nothing built. State re-read (appendix S + Z + avifgen status + budget harness). avifgen coordination facts recorded above; their run prefixes are not touched. Sources re-verified: 1,140 PQ PNGs present, sampled cICP = {1,16,0,1}. Dep pins: zenav1-svt `4c5c1324` (public repo, clean tree), ultrahdr `73ab3d27` (clean). Budget harness at `/mnt/v/output/zensim/hdrbudget-2026-08-05/harness/` supplies the proven svt+uhdr encode shapes (its `to_yuv420_bd10` ran every ladder size at p6). |
| 2026-08-06 ~20:54 | **(epochshard lane) Epoch-sharded claiming is available for this lane — intended first consumer.** Landed in zenfleet: `ZEN_CLAIM_MODE=epoch-sharded` on every worker (plus optional `ZEN_EPOCH_LEN_SECS`, default 600, and `ZEN_EPOCH_HB_SECS`, default 120), or write `{"claim_mode":"epoch_sharded"}` into the run's RunControl object to converge the fleet centrally. Workers then re-shard the remaining gap at every wall-clock boundary via rendezvous hashing over the heartbeat roster — zero claim traffic in steady state, leases kept as the seam/steal guard, exactly-once still ledger-enforced. Needs a worker-image rebuild that includes the new binary+entrypoint (any rebuild from a checkout containing commit `ad1dd3a0` or later picks it up). Adoption guide: `docs/RUNNING_JOBS.md` §5b; gates: `zenfleet-sim/tests/epoch_shard.rs` (total work == distinct cells; boundary takeover; steal bounds). One mode per run — don't mix with lease workers on the same run. |
