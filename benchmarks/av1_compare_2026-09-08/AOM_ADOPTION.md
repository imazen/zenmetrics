# AOM techniques worth testing in SVT

Source audit dated 2026-09-08, against the exact sibling sources used by the
comparison executable. These are candidate experiments, not demonstrated
causes of AOM's RD results. No SVT coding tools or parity expectations were
removed to obtain the measurements.

The filled preset sweep already rejects a blanket premise that AOM compresses
better. At SSIMULACRA2 70 on the 376x512 photo, zenav1-svt preset 1 estimates
4,140 bytes / 352 ms; zenav1-aom's smallest tested point is preset 1 at
4,224 bytes / 980 ms. On the 512x330 screenshot, SVT preset 0 estimates
7,773 bytes / 1,805 ms versus AOM preset 0 at 7,874 bytes / 2,014 ms.
These are bracketed quality estimates from three-round local measurements.
The high-quality screenshot remains a useful AOM advantage witness.

The C research-mode measurement is also complete: at score 80, preset -1
estimates screenshot **10,831 B / 2,495 ms**, versus preset 0 **11,136 B /
1,083 ms**. On the photo -1 is **10,608 B / 737 ms**, versus preset 0
**10,509 B / 436 ms**. The -1 ladder changes several tools, so these are whole-mode
results, not an isolated SGR attribution.

## Priority and scope

| Candidate | AOM behavior | Current SVT behavior | Experiment and acceptance evidence |
|---|---|---|---|
| Complete C research preset -1 first | Slow presets retain broader restoration/search options | Signed -1 is now wired through the Rust pipeline, AVIF wrapper and comparator locally; targeted geometry/tile parity passes, broader parity remains open | Treat this as C feature coverage. Port the complete signed derivation chain, not a cast or alias of preset 0; validate configuration, search, signaling and decoder reconstruction against C |
| Restoration-unit size search | Speed 0 searches 64..256; higher speeds select size using quality/resolution policy | C and the native Rust path fix units at 256; opt-in Rust search now evaluates legal 256/128/64 sizes | Implemented locally with SVT unit costs and frame signaling cost. Ten enabled native8/10 reconstruction cases pass, including odd/tile/SB128 boundaries. 144-encode canonical ablation complete; no measured RD win on these two origins, so keep opt-in |
| Self-guided restoration at useful still efforts | Wiener/SGR/switchable competition with speed-dependent pruning; restoration is disabled at AOM speed 5+ | SGR is wired for the video-key path and native research -1; normal all-intra presets 0+ disable it | First measure/port -1. Then independently test a still override at selected normal presets. Reuse native-depth search, apply and signaling; do not merely set a header bit |
| Content-adaptive directional pruning | Gradient orientation histogram narrows directional modes; threshold changes by speed | Slow still modes enumerate directions/angle deltas; faster modes use a coarse directional mask and existing staged candidate pruning | Test HOG or a cheaper orientation mask before expensive mode evaluation, with full-search winner retention recorded. Spend any saved time on broader useful candidates and compare at equal time |
| Learned 8x8 transform-depth pruning | A small model can prune split/non-split after the largest transform evaluation, enabled at speed 6+ | Transform-depth search uses existing coefficient-count exits and transform-type SATD/rate gates; no equivalent learned depth predictor was found | Log exhaustive SVT depth winners first. Validate or retrain for SVT's cost model and residuals; AOM's thresholds/weights cannot be assumed transferable. Measure quality loss versus time saved |
| Intra-edge filtering policy | The still sequence enables intra-edge filtering by default | SVT all-intra enables it only where its angular-search policy requests it (preset 5 in the normal ladder); video enables it throughout | Low-cost isolated ablation on directional texture/text. Prediction and sequence signaling must change together. Measure before deciding whether the different policy is beneficial |

## Source locations checked

Paths below are relative to their named sibling repository. Read the executed
branches, not just their comments; several introductory port comments are stale.

- **C SVT:** `Source/API/EbSvtAv1Enc.h` names MRS=-3, MRP=-2 and MR=-1;
  `Source/API/EbConfigMacros.h` sets normal `MIN_ENC_PRESET=ENC_MR`;
  `Source/Lib/Globals/enc_settings.c::svt_av1_verify_settings` enforces it.
- **Rust SVT:** `rust/crates/svtav1-encoder/src/speed_config.rs::NativePreset`
  now carries checked -1..13, and `SpeedConfig`/`rate_arm.rs::eff_enc_mode`
  preserve signed values. The research port map in zenav1-svt records the
  derivation, search and signaling fixes plus targeted validation evidence.
- **Restoration size:** SVT `restoration.rs` keeps 256-pixel units in the native
  path; the explicit `AomRestorationUnitSearch` consumer now evaluates legal
  smaller sizes. C `pcs.c` sets the fixed size. AOM `speed_features.rs::lr_search_sf_allintra` derives
  `min_lr_unit_size`/`max_lr_unit_size`, consumed by the real
  `key_frame.rs::pick_filter_restoration` invocation.
- **SGR:** SVT `pipeline.rs` now supplies live research controls for
  `ScArm::Allintra` at -1, disabled controls at normal still presets, and
  live derived controls for `ScArm::Video`;
  C `enc_mode_config.c::svt_aom_get_sg_filter_level_allintra` enables it for
  `enc_mode <= ENC_MR`. AOM's `lr_search_sf_allintra` supplies live SGR and
  Wiener pruning controls; `key_frame.rs` clears restoration at speed 5+.
- **Directional modes:** AOM `partition_pick.rs` calls
  `prune_intra_mode_with_hog_y` with speed-dependent thresholds. SVT
  `leaf_funnel/inject.rs` builds directions and angle deltas according to
  `angular_level`; `leaf_funnel/nic.rs` supplies existing staged pruning.
- **Transform depth:** AOM `tx_search.rs` wires `NnDepthPruneCtx` into the
  largest-depth walk and calls `ml_predict_intra_tx_depth_prune` for its
  supported 8x8 case. SVT `leaf_funnel/mds3.rs` uses
  `txs_prev_depth_exit`; `leaf_funnel/txt.rs` already implements type groups,
  SATD exits and rate gates.
- **Intra edges:** AOM `key_frame.rs` sets `enable_intra_edge_filter=true`;
  SVT `intra_arm.rs::intra_edge_filter` derives the all-intra/video policy,
  shared between prediction and sequence signaling.

## What the audit does not establish

- Palette, IntraBC, CfL, RDOQ and staged winner selection already exist in
  SVT's executed paths. Their presence in AOM is not a missing-feature finding.
- Existing SVT scratch reuse also exists. Seeing allocations is not evidence
  that another buffer-reuse patch will help; prior null/slower experiments
  are documented beside `Mds3Scratch` and must not be forgotten.
- AOM's speed-6+ depth predictor cannot explain its speed-0 compression
  advantage. It is a speed technique to evaluate on its own merits.
- Turning on a tool does not guarantee a perceptual improvement: these
  searches optimize their encoder cost functions, while the reported curves
  use full RGB SSIMULACRA2. Default SVT tuning in this run is its PSNR tune;
  IQ/SSIM tuning needs its own measured arm.
- Two resized sources cannot justify a production routing model or a
  universal new default. Keep extensions explicit until broader held-out
  images, bit depths and format combinations support the change.

The preset-fill parity audit found 53 nonidentical C/Rust SVT outputs among
168 paired cells (115 exact). The largest observed score delta was +0.608
for Rust; the largest negative delta was -0.465. These witnesses are retained
separately from optimization candidates. Successful decoding and good RD do
not close a bit-exact translation gate.

## Intra-edge continuation implementation and correctness finding

The opt-in `aom-intra-edge-filter-v1` experiment now reaches native SVT -1
prediction and sequence signaling, with explicit mainline420/hybrid3115
reference selection. Comparator request fields are `svt_reference` (the full
pinned source ID) and `zen_intra_edge_filter`; legacy omitted references resolve
to hybrid3115 and output rows record that resolution. The linked C SVT arm
remains hybrid and rejects a request to label it mainline420. Serialization
replay, refusal paths, and actual changed reference/decoded output are tested.
Analysis keeps reference, enhancement, binary and source identities distinct.

The initial120-encode,40-cell,three-round canonical photo/screenshot ablation
completed and decoded, but is SUPERSEDED for policy selection: subsequently
expanded native8/10 and tile/odd-size reconstruction gates exposed incorrect
chroma smooth-neighbor ownership after4x4 luma splits. The correction preserves
chroma-owner modes across luma-only children and selects the normative8x8-group
neighbor. Source proof and before/after witnesses are in zenav1-svt's
`rust/docs/research-preset-port-map.md` (Zen continuation section).

Post-fix checks:2626 workspace tests,136 regression spotchecks,36 off/on
geometry/depth/QP cases,6 static comparator tests all pass. Refreshed parity passes1,100 hybrid,1,100 pristine normal8 and320 pristine
research cells. The corrected120-encode rerun is complete, with20 untimed
exact reconstruction replays. See [corrected size/quality/time results](IMAZEN26_INTRA_EDGE.md):
matched-quality payloads grow1.4%/1.7% at the selected photo/screenshot targets. The pre-fix run is retained at
`~/tmp/av1-imazen26-intra-edge-2026-09-08` with a supersession note in provenance.
The experiment remains off by default; no routing/calibrated-effort claim.

[Completed restoration-unit ablation](IMAZEN26_RESTORATION_UNITS.md): two photo
quantizers selected 128-pixel units but lost measured RD, and screenshot search
was bypassed. Retain the explicit experiment; do not calibrate a default from
these two origins.
