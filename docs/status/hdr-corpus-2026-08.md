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
- [x] zenfleet extended for HDR multi-codec encode + score (**B1/B3/B4** `d7318527`, **B2** `9093cc23`, **B7** images `exec-hdr-9093cc23` + `exec-gpu-hdr-9093cc23`)
- [x] Refs on R2 (1,140), grid cells emitted (102,600 = 1,140 × 3 arms × 30 q), run `jobs/hdrgrid-enc-20260806` declared, scoped 7-day cred minted
- [ ] Workers enrolled on all encode nodes
- [ ] **First-cell artifact-persistence gate PASSED** (encode half in flight; diffmap + metric-variant halves gate their own declares)
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
| 2026-08-06 ~20:20-21:00 | **B1+B3+B4 landed (`d7318527`)**: HdrCodec enum (4 arms), svt arm (budget-harness BT.2020nc 10-bit 4:2:0 + zenavif-serialize mux, decode-back through the ordinary AVIF path, round-trip gate <10% mean rel luma err), gainmap arm. **B4 finding:** ultrahdr's built-in HDR-only path quantizes gain bytes on the CONFIG range but stores metadata declaring the ACTUAL range — every reader reconstructs under-boosted (measured 2000→732 nits; byte math reproduces it). Filed **imazen/ultrahdr#33**; the arm composes public primitives + rewrites channel min/max to the actual quantization grid → interop-correct files, measured 1916.7/2000 nits. Encode.hdr rides the ScoreFile append-only serde pattern — golden test proves every existing encode job id (incl. the live avifgen ledger) byte-stable. |
| 2026-08-06 ~21:00 | **B2 landed (`9093cc23`)**: the Diffmap executor — butteraugli (CPU reference) + cvvdp (in-tree port) per-pixel maps matching the recorded scalars' feedings, gzip'd PFM blobs, HDR+SDR modes; ssim2 refused with the registered absent-not-failed wording (no in-tree map API). declare_diffmaps in zenfleet-ctl; declare_direct_objects.py gains ZEN_SCOREFILE_HDR=1. End-to-end gate: PQ ramp + real svt q30 variant → both maps parse, identical pair ≈ 0. |
| 2026-08-06 ~21:00 | **B7 done**: images `ghcr.io/imazen/zenfleet-worker:exec-hdr-9093cc23` (CPU: jobexec,sweep,png,jpeg,webp,avif,jxl,cpu-metrics,hdr,hdr-svt,hdr-gainmap) + `:exec-gpu-hdr-9093cc23` (GPU: +gpu,gpu-cuda, hdr-gainmap for variant decode; hdr-svt not needed to score). Both carry the avifgen lane's `92432e37` --ledger-in fix. |
| 2026-08-06 ~21:03 | Refs synced (1,140 → `s3://zentrain/refs/imazen-26-hdr-grid-2026-06-14/`), cells emitted (`scripts/jobsys/hdrgrid_cells.py`, sha-cached), **run `jobs/hdrgrid-enc-20260806` declared** (102,600 encode jobs, all hdr:true, 34,200/arm). G-S1 gate chunk running locally in the production container (`wsl-gate`). |
| 2026-08-06 ~21:10 | **Epoch-shard adoption decision** (their row above names this lane the intended first consumer): the ENCODE wave stays on lease+snapshot claiming — its images (`*-9093cc23`) predate `ad1dd3a0`, the run is 5.5× smaller than the avifgen encode that just proved the lease+`--ledger-in` pattern, and mixing modes per run is prohibited. Epoch-shard is the candidate for the SCORE wave (its GPU image gets rebuilt ≥`ad1dd3a0` while the GPU queue drains avifgen anyway). The JobId preserve_order hazard does not touch this lane: ctl + worker + executor were all built per-crate. |
| 2026-08-06 ~21:12 | **G-S1 FIRST-CELL GATE PASSED, all three arms.** zenjxl + zenav1-svt via the fleet ledger (blob sha256 == ledger `output_sha`; decode-back through the production `score-pairs --hdr`: jxl q25 → 9.48 JOD, svt q0/qp63 → 4.78 — sensible ordering); jpeg-gainmap hand-executed through the production image's jobexec against the R2 ref (chunk order hadn't reached the arm; same code path the fleet runs): valid Ultra HDR JPEG, decode-back 8.72 JOD @ q90. Ledger at gate close: 1,150 rows, **zero failures**, every kind hdr:true, per-row worker/provider provenance present. Gate artifacts: `/mnt/v/output/hdrgrid-2026-08-06/gate_*` + `gate_picks.json`. |
| 2026-08-06 ~21:12 | **Encode wave scaled — on the operator box, deliberately.** avifgen holds every CPU node (encode tail: lianli+ryzen ledger writes seconds old; then their sf-cpu queue) and per-box worker units are singletons their endgame daemon rewrites — enrolling second units would fight their flow for a ~15 CPU-h wave. Two production containers on wsl (`wsl-e1`/`wsl-e2`, --cpus 9 each, 10 cores left free) run the same image + entrypoint the fleet would. Endgame driver detached (`~/tmp/hdrcorpus2/enc_endgame.sh`): snapshot refresh every ~8 min (the `92432e37` --ledger-in contract), progress log, exact-count drain at the 99.5% floor, sentinel on every exit path. |
| 2026-08-06 ~21:22 | **Arm 3 fleet-path evidence complete**: the wave's own chunks reached jpeg-gainmap (1,080 rows) — a fleet-produced blob (wsl-e1, q94) sha-matches its ledger row, carries JPEG magic, decodes back at 8.52 JOD. Ledger 5,542/102,600 done, zero failures. Score-wave assets staged: images `exec-hdr-a7cf7df9` + `exec-gpu-hdr-a7cf7df9` (built from master ≥`ad1dd3a0` → epoch-shard capable), declare tooling committed (`96caf81f`), drain-time driver `~/tmp/hdrcorpus2/score_declare.sh` (4 queues: sf-gpu big/small split at min-dim 176 for the S.5 iwssim gate, sf-cpu cvvdp+zensim, diffmap butteraugli+cvvdp with true cell identity). WSL docker GPU passthrough is CDI-broken (`libdxcore`/NVML) — the GPU score smoke runs as the wave's node-2 first-cell gate instead (the registered G-Z3 shape), matching why avifgen gated there too. |
| 2026-08-06 ~22:05 | **(epochshard lane) Weighted sharding is ready for this lane's fresh-image GPU wave** — adopt together with epoch-sharded claiming when you build the score-wave images: `ZEN_CLAIM_MODE=epoch-sharded` + the baked `fleet/handicaps.toml` (already seeded: encode.zenavif measured from the avifgen ledger; **svt/jxl/gainmap encode types deliberately fall back to default=1.0 — no cross-type extrapolation** — run `scripts/jobsys/handicap_typebench.sh` per box on YOUR encode manifest to register real per-type rows); gpu_metric seeds node-2=lianli=1.0 (G-Z3 parity probe) with the sf-gpu-ledger derivation as the registered TODO, and CPU boxes are gpu_metric=0.0 (role exclusion). Campaign-level override: write `worker_weights` into your RunControl object (wholesale, next-pass convergence). Guide: `docs/RUNNING_JOBS.md` §5b. |
| 2026-08-07 ~00:3x-01:15 | **INCIDENT — root disk filled; encode wave ENOSPC-killed at 83%; recovered + relaunched.** The executor's warm-process source cache (`/tmp/jobexec_src_<pid>_*`) is pid-keyed and never evicted: every executor generation re-downloads the sources it touches into the container writable layer — measured **~620 GB across the two operator containers** (~15 MB retained/cell) → root 100% (1.6 TB) → both containers ENOSPC-died (~22:32Z), the supervisor pruned docker caches/images/stale containers (root back to 49%, 776 GB free). **Ledger audit**: 87,782 rows, **85,543 distinct DONE (83.4%)**, 97 failed rows over 71 distinct jobs (96 `encoder_panic` — the ENOSPC-window mislabel class also seen in the avifgen G-Z2 incident — + 1 `upload_fail`), zero half-written sidecars observed (ledger parquets upload whole-file). Failed cells are NOT in the done set → the gap re-claims them automatically; nothing to hand-requeue. **Fixes**: (1) relaunched both workers with `/tmp` bind-mounted to host scratch + a detached 20-min-TTL cleaner (`scratch_cleaner.sh` — steady-state ≈ hot-window sources ~4 GB/worker, not estate × generations); (2) fleet-entrypoint gains a race-free between-pass `jobexec_*` sweep (rides every future image; the avifgen fleet boxes carry the same slow leak — flagged to their lane via this doc). Unfixed, the wave's total scratch was ~1.5 TB (would have died twice more). Restart picked up the ledger snapshot (gap = 17,057 remaining cells); ETA ~45-70 min from 01:13Z. |
| 2026-08-07 ~01:2x | **Appendix AA hold adopted (nitsaudit lane, zensim `e5627b56`)**: the score wave's cvvdp `y_peak=1000` + butteraugli `intensity_target=1000` are class-(b) static-config values the measured-ref-peak directive replaces. Drain-time declares therefore cover ONLY the unaffected metrics (ssim2-gpu + iwssim-gpu integrated-PU feedings; zensim HDR-route score+features); **cvvdp + butteraugli score AND diffmap declares are deferred** to a phase-2 wave that uses the measured-peak fix + its executor image — zero schedule cost (the GPU queue is avifgen's for hours; those waves need the fixed image regardless). No static-1000 semantics get baked into content-addressed jobs. Fixed-scratch images `exec-hdr-9d30a00b` + `exec-gpu-hdr-9d30a00b` pushed. |
| 2026-08-07 01:52-05:59 | **ENCODE WAVE DRAINED**: 102,485/102,600 distinct done (99.888%, floor 99.5%). Residue = 115 cells, ALL zenav1-svt `encoder_panic` — reproduced locally: `restoration.rs:985 index out of bounds (len 2, idx 2)` in loop restoration, deterministic on 5 renditions (bd10 420 p6, real-photo content the synthetic robustness matrix misses) — filed **imazen/zenav1-svt#11**, enumerated in `encode_residue.json`, absent-not-failed. Scratch stayed bounded post-fix (33-36 GB peak vs 620 GB unbounded). **Score waves declared 05:58Z** (bridge: 102,485 done cells → 96,787 unique pairs after content-dedup): sf-gpu mid 2,280 jobs / huge 453 (>4MP, concurrency-1) / small 687 + sf-cpu 3,420 (zensim only — AA hold). Coordinator GPU-tax directives adopted: **chunk=30** = one ScoreFile job per (ref × codec) so the ref decodes once for the whole q ladder and spawn/CUDA-init amortizes ~30×; **pixel-derived mem hints** on every job (600 MiB base + 400 MiB/MP — first consumer of VRAM-aware admission); huge tier isolated for 1-way serving on the 8 GB cards; gpu_metric handicaps to be consumed from the avifgen lane's clean-window derivation at bringup. CPU wave LIVE on the freed operator containers (`hdrcpu1/2`, fixed image `exec-hdr-9d30a00b`, mounted+TTL-cleaned scratch). |
| 2026-08-07 ~06:07 | **PHASE-2 DECLARED** (the AA-held metrics, unblocked by `6471f4d7` measured-peak + images `exec-hdr-3783eb03`/`exec-gpu-hdr-3783eb03`): `hdrgrid-sf2-gpu-20260807` 2,280 + `-huge` 453 + `-small` 687 (butteraugli-gpu) + `hdrgrid-sf2-cpu-20260807` 3,420 (cvvdp) + **`hdrgrid-diffmap-20260807` 193,574 jobs** (the B2 wave at true scale: 96,787 pairs × butteraugli+cvvdp maps, per-variant cell identity, pixel hints). Phase-2 jobs MUST be served by ≥`3783eb03` images (measured-peak semantics; ids are image-independent — enforcement is the bringup env). |
| 2026-08-07 ~06:08 | **CPU drain-chain daemon** (`~/tmp/hdrcorpus2/cpu_chain.sh`, detached): serves the CPU queues sequentially on the operator containers — zensim (live, pre-AA image is correct for PU-fed zensim) → cvvdp (`3783eb03`) → diffmaps — with per-stage floor checks, progress log, sentinel on every exit. **GPU bringup runbook (when avifgen's GPU queues drain):** reuse `homefleet .../avifgen_score_bringup.sh` with `ZEN_WORKER_IMAGE=ghcr.io/imazen/zenfleet-worker:exec-gpu-hdr-3783eb03` + env files `~/tmp/hdrcorpus2/worker.env.{gpu,gpuhuge,gpusmall,sf2gpu,sf2gpuhuge,sf2gpusmall}` (scoped creds baked, REQUIRE_GPU set); serve `-huge` runs one box at a time at concurrency 1; consume the avifgen-derived `gpu_metric` handicap rows; phase-1 ssim2+iwssim queues first (declared 05:58Z), then sf2 butteraugli. |
