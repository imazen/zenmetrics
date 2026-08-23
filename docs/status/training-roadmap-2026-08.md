# Training roadmap — zensim / zenavif / zenjxl / zenanalyze (2026-08-22)

Operator-requested consolidation. Sources: `DATA_PROVENANCE.md`, the sota944 campaign
doc, `CLEAN_PICKER_PROGRAM.md`, `JXL_LOSSY_KNOBSPACE_ABLATION_PROGRAM.md`,
`docs/status/{avif-datagen,hdr-corpus}-2026-08.md`. This doc orders the next concrete
steps; the program docs stay authoritative for their own details.

## Cross-cutting data rules (apply to every program below)

1. **Canonical imazen-26 = the dedicated `imazen/imazen-26` repo** (moved
   2026-08-23; predecessor = codec-corpus PR #12): `CORPUS-MANIFEST.tsv`
   (membership oracle) + `manifests/{train,validate,test}.tsv` (last-digit rule)
   + the `variant-sets/` registry (per-set sha manifests + consumer map). **Anything derived from any other
   imazen-26 copy is broken unless sha-verified against those manifests.** The
   `/mnt/v` copies are caches. Near-term blast radius is small and enumerable —
   3 renamed ids (9231/9869/9874) + metadata-only EXIF rewrites (pixels verified
   unchanged) — but verification, not assumption, is the rule.
2. **Splits**: derived datasets inherit an image's bucket by id. The avifgen corpus is
   structurally train-only (all origins even); val/test always come from odd-digit
   origins elsewhere. New sweeps must cover odd origins where a picker needs held-out
   evaluation (they never have: zenjpeg/zenavif/zenwebp odd origins were never swept).
3. **Variant generation** follows `imazen-26` repo `VARIANTS-SPEC.md` (v2) via the
   registered entrypoint `scripts/make_variant_set.py` (sets are born versioned in
   `variant-sets/`):
   density is chosen per task by the measured procedures (thumbnail coverage floor;
   k-means reps at the K-knee on content features, within-split; budget-first FPS
   with prefix-truncatable GP budgets), sizes come from one candidate grid,
   id-leading dot-chain names ⇒ mechanical split inheritance, kernel/sharpen/
   colorspace are recorded axes, mandatory `variants.tsv` with selection method +
   generator commit + sha256.
4. **Vendor-class partitioning** (zensim#56): any feature re-extraction wave
   partitions boxes by CPU-vendor class — the MSCN append slots are
   vendor-nondeterministic at ~1e-8.
5. **JobId build discipline** (#38): worker/ctl binaries never build in a combined
   cargo invocation with zenmetrics-cli until `JobId::of` is order-insensitive.

## zensim (944 era)

1. **avifgen data-gate CLEARED 2026-08-21** (04d03359: LAN drain 172/172, union
   8,623/8,623, verdict RESCUE-WINS 3/2,000, G-Z5 re-gate 0.999313, views + manifests
   + Tower mirrors sha-verified) — **wave-12 is unblocked**: launch it on the
   AC.R1-amended views (`train_944` train-only + `eval8_944`).
2. **Ship route stays distillation (wave-6 arm F)** from the seed ensembles
   (`W5_E1_k2` CID22 0.89425 / `W6_GE2_trio` KonJND 0.4543 are eval functions, not
   bakes); every freeze candidate needs `bake_dial_refit add-spline` first (G-RANGE).
   The freeze decision itself is the USER's.
3. **Classic-IQA breadth is the binding axis** on the balanced board (0/172 cells pass
   all 8 floors; CSIQ/LIVE ≥ 0.83 is what fails) — attack via distill mixes that
   include the classic anchors, not by more single-corpus tuning.
4. **Publish chain (user-gated)**: zenpredict 0.2.x with the v3 runtime →
   `zensim 0.3.0` (#46) → unblocks zenwebp/zenavif/imageflow migration to profile B.

## zenavif

1. Campaign CLOSED 2026-08-21 (04d03359); `gpu_metric` handicaps derived from the
   rescue clean window already landed (c3ea0fef). Nothing left here but consumption.
2. **Clean picker, still undelivered**: newest committed `.bin`s are 2026-05-04; the
   2026-06-28 clean/veto picker was validated but never committed. Path: canonical
   `modes_full`-class re-sweep **including odd origins** (LAN fleet + LAN store, job
   system) → `train_hybrid` with knob-vetoes (the "we cannot encode more than once"
   rule: K=1 + vetoes, no top-K verify) → bake → commit `.bin` into
   `zenavif/benchmarks/` + wire per PICKERS task A.
3. HDR AVIF arm remains the hdr-corpus lane's call (B5) — the 2026-08-06 directive
   lifted only SDR.
4. Fix `imazen/zenav1-svt#11` (restoration.rs OOB on bd10 420 p6 real-photo content;
   115 hdrgrid cells enumerated absent-not-failed).

## zenjxl / jxl-encoder

1. **Ablation program P0 fleet confirmation** per the registered runbook — with two
   amendments: run it on the **household fleet + LAN store** (the runbook predates
   both; Hetzner remains the fallback), and fix **#34 (`encode_ms` ~31 s/MP additive
   parasite)** first — per-effort timing is a P0 analysis axis and is currently
   garbage.
2. Then P1 pairwise crosses → P2 code-the-settled (only `progressive→Single` has
   graduated so far) → P3 picker + oracle gap (student overhead 17.66% at 60 imgs;
   ≤1% needs the fleet-scale dense corpus).
3. **`chroma_subsampling=420` wiring** (jxl-encoder #47 chunk 4) — the single
   highest-value missing web axis; prerequisite for jxl-lossy picker v0.2.
4. **jxl-lossless K=1 gate** stays open (val argmin 9.9% < 10% floor; worst tails are
   effort/predictor picks, not palette) — apply the avif-proven per-knob veto approach
   before considering architecture changes.

## zenanalyze / zenpredict

1. **Publish zenpredict 0.2.x with the v3 runtime** (the #46 unlock) — release
   sequence is user-gated; staging is already done on main.
2. **dense-percentiles branch stays experimental-gated** (decision on record): run the
   selection against a real codec target, promote only surviving features; that is the
   gate on 1.0. Do not merge the 90-feature surface wholesale.
3. `zenanalyze-api` stays frozen (0.1.x contract).

## Sequencing note

With the avifgen gate cleared, wave-12 training is the highest-value next launch; the
jxl P0 confirmation and the zenavif odd-origin sweep are CPU-fleet work
and can run concurrently on the LAN fleet under the job system. Nothing in this
roadmap needs paid cloud except (optionally) jxl P0 scale-up, capped per its $25
budget line.
