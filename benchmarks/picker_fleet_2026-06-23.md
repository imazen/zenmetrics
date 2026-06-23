# Picker training on the Hetzner-CPU fleet (2026-06-23)

Scaled the local picker pipeline to a real fleet (Hetzner CPU; vast.ai available
but not needed for the scalar sweep). Proves fleet→aggregate→train end-to-end and
shows fleet-scale data measurably improves the pickers.

## Pipeline

`scripts/sweep/hetzner_cpu_sweep.sh` — each cheap box fetches its chunk of
renditions from R2 and runs **one-pass** `zenmetrics sweep --metric ssim2
--metric zensim` (the deadlock-fixed CPU binary in the PUBLIC
`ghcr.io/imazen/zenfleet-worker-exec:2026-06-22` image, current codecs) → an omni
TSV (the exact picker training format, both metrics per cell) → uploads it.
Mints scoped 3h R2 creds (never root on a box); `docker-ce` image (no apt at
boot); biggest-first type×location fallback; **persists encoded variants** as a
tarball/box (the master record — 372 zensim features re-extractable on GPU,
diffmaps, any future metric, with no re-encode).

## Constraints found (operational, honest)

- **Hetzner account caps at 5 servers.** Freed the idle `zen-trellis-sweep` box
  for a slot; otherwise slot-limited → sequential codec cycles. Raise the
  Servers limit for wide parallelism.
- **cpx41 is phased out**; big dedicated boxes (ccx53 32c / ccx63 48c) are
  capacity-out → **cpx51 (16c) is the practical ceiling per box**. Biggest-first
  fallback lifts off the 8c straggler but can't conjure capacity.
- CPU zensim emits **300** features; the full **372** (Profile-A) needs the GPU
  WithIw regime — hence persisting variants for GPU re-extract rather than
  storing CPU's partial 300.

## Results — jpeg (1398 renditions / 201,313 cells, vs local 154 img / 34k)

| jpeg picker | local | **fleet** |
|---|---|---|
| predict-ssim2 | 7.18% | **5.13%** |
| predict-zensim-a | 9.36% | **6.32%** |

Held-out mean bytes-overhead; lower = better. Fleet-scale data cuts the overhead
~2–3 pp — consistent with the measured ±2–3 pp seed-noise floor at 154 imgs
(`picker_smoothness_2026-06-22.md`): more data, less noise, better picks.

## Heterogeneous SPLIT — GPU metrics on vast.ai (proven)

Encode once on cheap Hetzner CPU, score many GPU metrics on vast.ai over the
**persisted variants** (no re-encode). `zenmetrics score-pairs` reads a pairs TSV
(`image_path codec q knob_tuple_json ref_path dist_path`), decodes each variant,
scores it → parquet sidecar.

- **All 6 GPU metrics verified** (local RTX 5070, then vast.ai), real scores, 0
  failed on 210 jpeg variants: butteraugli-gpu (`butteraugli_max_gpu` 1.9–7.7,
  +`_pnorm3`), **cvvdp** (`cvvdp_imazen_v0_0_1`, 8.85–9.95 JOD), ssim2-gpu
  (45–83), **zensim-gpu** (25–77 — the Bug#1/#2 fixes), dssim-gpu, iwssim-gpu.
- **Images (public):** `zenmetrics-sweep:v29-2026-06-23` (GPU binary, all 6
  metrics + CUDA 12.6); `zenmetrics-sweep:v29-split` (FROM v29 + `split_score_worker.sh`).
- **vast quirk:** vast runs `--onstart-cmd`, **not** the image ENTRYPOINT (it does
  its own ssh init) — launch the worker via `--onstart-cmd bash
  /usr/local/bin/split_score_worker.sh`, not by relying on the entrypoint.
- CPU zensim emits **300** features; the **372** (Profile-A) needs the GPU WithIw
  regime — so persist variants and re-extract 372 on GPU.

## Open / queued

- **Quality under/overshoot %** — the eval reports bytes-overhead + argmin-acc +
  scalar-RMSE but NOT whether the picked config hits the target zq. Add to
  `train_hybrid` (deferred while zenanalyze is mid-redesign by another session).
- webp / jxl / avif fleet runs (sequential, 2 slots) + a jpeg persist pass; then
  the vast SPLIT GPU-metric pass over all persisted variants.
