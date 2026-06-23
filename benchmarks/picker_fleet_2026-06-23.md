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

## Open / queued

- **Quality under/overshoot %** — the eval reports bytes-overhead + argmin-acc +
  scalar-RMSE but NOT whether the picked config hits the target zq. Adding it to
  `train_hybrid` (deferred while zenanalyze is mid-redesign).
- webp / jxl / avif fleet runs (sequential, 2 slots) + a jpeg persist pass for
  variants.
- (Optional) GPU pass to materialize the 372 zensim features as a parquet.
