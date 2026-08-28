# butteraugli-gpu — multires strip-vs-whole drift on a real corpus (zenmetrics#47 item 4) — 2026-08-28

**Question.** `Butteraugli::new_multires_strip` is documented (and gated in
`crates/zenmetrics-cli/src/orchestrator_runner.rs`) as sitting "up to ~1e-4 rel"
from `Butteraugli::new_multires`, which is above the Phase-7.7.1 parity gate
(bit-exact, or ≤ ~5e-5 atomic-reorder noise) — so butteraugli stayed
`metric_orchestrator_eligible == false`. The owner's decision on #47 was:
*"1e-4 is ok for better mem and perf, but do check the record on this issue and
if that error grows depending on test corpus and deltas."*

**Answer.** The error does not grow — because it is not 1e-4 to begin with. On
**6,810 measured cells** across 95 real images × 15 JPEG quality levels × 4
strip bodies, the multires strip walker's **max-norm score is bit-identical to
the whole-image multires score in 6,810 of 6,810 cells** (`rel_score` max
`0.00e0`). The `pnorm_3` aggregate differs by at most **1.72e-6** relative
(p99 4.6e-7) — that residual is host-side partial-fold ordering, and it does not
grow with distortion magnitude, resolution, strip count, quality or content
class. The **production surface** (`ButteraugliOpaque` `MemoryMode::Full` vs
`MemoryMode::Auto`, which is what the legacy CLI and the orchestrator actually
construct) is likewise identical in 6,810/6,810 cells.

The "~1e-4" figure in the source comments is the **assertion tolerance** the
`multires_strip` / `strip_parity` tests were written with, not a measured drift.
The one measurement that ever produced ~1e-4 (`butter_strip_halo_2026-05-31.md`)
was the **pre-fix** `HALO_ROWS = 40` half-res sibling; the same document's
post-fix table already reported `0.00e0`. This run confirms that post-fix
behaviour holds far outside the single synthetic checkerboard it was measured on.

---

## Provenance

| | |
|---|---|
| date | 2026-08-28 |
| repo commit | `f7ce8d2a1f4e0c311e8f03d2f7dc11d1c4429b8e` (`master`) |
| host | Mac mini, Apple M4 Pro, 12 CPU cores (8P/4E), 16 GPU cores, 24 GB unified, macOS 26.5.2, arm64 |
| backend | **`cubecl-wgpu` → Metal** (`vram_cap_bytes()` = 8 GiB, the default cap — no live probe override) |
| **CUDA** | **not available on this host — the CUDA envelope is NOT measured here** (see "What is still open") |
| build | `cargo build --release -p butteraugli-gpu --no-default-features --features wgpu,cubecl-types --example strip_drift_corpus` |
| reduction | portable per-thread-partials (`fast-reduction` **off**, the crate default — Metal silently no-ops `Atomic<f32>::fetch_add`) |
| harness | `crates/butteraugli-gpu/examples/strip_drift_corpus.rs` (committed with this record) |
| raw data | `benchmarks/butter_strip_drift_2026-08-28.csv` |

### Reproducing

```bash
# manifest.tsv is headerless `class<TAB>absolute_png_path`
DRIFT_MANIFEST=manifest.tsv DRIFT_OUT=drift_native.csv \
DRIFT_QUALITIES=100,98,95,90,85,75,65,50,35,20,10,5 \
DRIFT_BODIES=auto,128,256,512 DRIFT_MAX_MP=8.0 \
cargo run --release -p butteraugli-gpu --no-default-features \
  --features wgpu,cubecl-types --example strip_drift_corpus
```

`DRIFT_SCALES=128,256,512,1024,2048,3072` adds a Lanczos3 size ladder per image
(downscale only — never upscale). `DRIFT_CPU=1` adds the CPU-crate leg.

---

## Grid

Four passes, 6,810 rows total, 77 distinct geometries.

| pass | rows | what it varies |
|---|---:|---|
| `native` | 4,320 | 95 images at native size × q{100,98,95,90,85,75,65,50,35,20,10,5} × body{auto,128,256,512} |
| `size_ladder` | 1,290 | 16 images × Lanczos3 long-edge {128,256,512,1024,2048,3072} × q{95,85,70,50,30,15} × body{auto,128,256} |
| `tall_native` | 120 | the 5 tall web screenshots (8.0–10.5 MP, up to 75 strips) × q{95,85,70,50,30,15} × body{auto,128,256,512} |
| `cpu_context` | 1,080 | same 95 images × q{95,85,70,50,30,15} × body{auto,256}, **plus** the CPU `butteraugli` crate leg |

Coverage actually exercised:

- **size**: 0.004 MP (128×32) → 10.51 MP (1313×8008); tiny / small / medium / large all present.
- **distortion**: whole-image butteraugli max-norm **0.32 → 31.62** — from
  near-transparent (q100) to visually destroyed (q5). Low-q is sampled at least
  as densely as high-q.
- **strip count**: 1 → 75 strips dispatched.
- **strip body**: `auto` (= what `memory_mode::resolve_auto` picks for the
  production 8 GiB cap), plus pinned 128 / 256 / 512.

### Corpus (95 images, 8 content classes)

| class | source | images |
|---|---|---:|
| `photo-cid22` | codec-corpus `CID22/CID22-512/training` (512²) | 21 |
| `photo-clic` | codec-corpus `clic2025/training` (~2–3 MP) | 8 |
| `photo-gb82` | codec-corpus `gb82` | 13 |
| `screen-gb82sc` | codec-corpus `gb82-sc` (screen content, 0.38–5.6 MP) | 10 |
| `screen-web` | codec-corpus `qoi-benchmark/screenshot_web` (tall page captures) | 13 |
| `screen-imz` | imazen-26 K300 `8100-lilith-web-screenshots` | 8 |
| `lineart-plot` | imazen-26 K300 `7000-lilith-plots` | 8 |
| `illustration` | imazen-26 K300 `9094-lilith-ai-illustrations` + `9000-lilith-ai-clipart` | 14 |

The imazen-26 rows were pulled from the K300 representative manifest
(`imazen/imazen-26` → `manifests/imazen26_representatives_K300_2026-06-14.tsv`),
PNG-v3 layer, public R2 URLs. The rest are the local `codec-corpus` checkout.
`phoboslab.org.png` (27 MP) was excluded — a whole-image multires reference at
that size wants ~6.8 GB of device buffers and this box is shared.

### Negative control

Every row carries `singleres_whole_score` — `Butteraugli::new` (single
resolution) on the same pair. That is the algorithm the opaque Strip arm ran
*before* the multires-strip fix, and it diverges from multires by a **median
6.3 %, max 25.9 %**. A `rel_score` column of all zeros only means something next
to a control column that is emphatically not zero; this one is.

---

## Result

| comparison | cells | max rel | notes |
|---|---:|---:|---|
| **`new_multires_strip` vs `new_multires` — score (max-norm)** | 6,810 | **0.00e0** | bit-identical in every cell |
| **`new_multires_strip` vs `new_multires` — pnorm_3** | 6,810 | **1.72e-6** | p99 4.6e-7; host-fold ordering |
| **`ButteraugliOpaque` Full vs Auto (production surface)** | 6,810 | **0.00e0** | bit-identical in every cell |
| strip result vs strip *body size* | 2,118 groups | **0.00e0** | `body_invariant == yes` in 2,118/2,118 |
| *(control)* single-res whole vs multires whole | 6,810 | 2.59e-1 | median 6.3 % — harness is sensitive |
| CPU `butteraugli_strip(256)` vs CPU `butteraugli` | 1,080 | 9.37e-8 | the CPU crate's own walker is exact too |
| GPU multires whole vs CPU `butteraugli` | 1,080 | 1.29e-2 | median 2.02e-4, p95 2.37e-4 — a **separate** axis, see below |

`rel = |strip − whole| / max(|whole|, 1e-12)` on `f64`-widened `f32` scores.

### Does it grow with anything?

No. Full bucket tables in the CSV; the maxima by axis:

**By distortion magnitude** (whole-image butteraugli score) — flat:

| bucket | cells | max rel_score | max rel_p3 |
|---|---:|---:|---:|
| [0, 0.5) | 100 | 0.00e0 | 1.98e-07 |
| [0.5, 1) | 567 | 0.00e0 | 5.33e-07 |
| [1, 2) | 1168 | 0.00e0 | 5.21e-07 |
| [2, 4) | 1961 | 0.00e0 | 9.46e-07 |
| [4, 8) | 1931 | 0.00e0 | 1.72e-06 |
| [8, 16) | 839 | 0.00e0 | 7.30e-07 |
| [16, ∞) | 244 | 0.00e0 | 6.91e-07 |

**By image size** — flat:

| bucket | cells | max rel_score | max rel_p3 |
|---|---:|---:|---:|
| [0, 0.1) MP | 498 | 0.00e0 | 3.74e-07 |
| [0.1, 0.3) MP | 1548 | 0.00e0 | 3.91e-07 |
| [0.3, 1) MP | 1098 | 0.00e0 | 3.53e-07 |
| [1, 3) MP | 2538 | 0.00e0 | 1.30e-06 |
| [3, 6) MP | 750 | 0.00e0 | 1.72e-06 |
| [6, 12) MP | 378 | 0.00e0 | 9.46e-07 |

**By strip count** — flat (this is the axis that *would* show a halo shortfall):

| bucket | cells | max rel_score | max rel_p3 |
|---|---:|---:|---:|
| 1 strip | 2430 | 0.00e0 | 1.72e-06 |
| 2–4 | 2502 | 0.00e0 | 1.30e-06 |
| 5–9 | 1092 | 0.00e0 | 1.72e-06 |
| 10–24 | 648 | 0.00e0 | 1.72e-06 |
| 25–59 | 126 | 0.00e0 | 1.72e-06 |
| 60–75 | 12 | 0.00e0 | 9.46e-07 |

**By content class** — flat; the max `rel_p3` cell is `screen-gb82sc`
(`gmessages.png`, 1440×3088, q20): `whole_p3 = 2.350464582` vs
`strip_p3 = 2.350468636`, i.e. 4.05e-6 absolute on a 2.35 score.

| class | cells | max rel_score | max rel_p3 |
|---|---:|---:|---:|
| illustration | 1008 | 0.00e0 | 1.30e-06 |
| lineart-plot | 612 | 0.00e0 | 5.86e-07 |
| photo-cid22 | 1356 | 0.00e0 | 3.91e-07 |
| photo-clic | 648 | 0.00e0 | 3.22e-07 |
| photo-gb82 | 912 | 0.00e0 | 2.70e-07 |
| screen-gb82sc | 816 | 0.00e0 | 1.72e-06 |
| screen-imz | 636 | 0.00e0 | 3.72e-07 |
| screen-web | 822 | 0.00e0 | 9.46e-07 |

**By JPEG quality** — flat across q100 → q5 (per-bucket max `rel_p3` ranges
2.94e-7 … 1.72e-6, with no trend in q).

### Why bit-identity on the max-norm is the *expected* result, not luck

`HALO_ROWS = 80` gives the half-res sibling 40 real halo rows against a 34-row
requirement (`strip.rs` module docs). With the halo fully covering the blur
support, every body pixel's diffmap value is computed from exactly the same
inputs as in the whole-image pass, so the diffmap is bit-identical. The
max-norm fold is `max`, which is exact and order-independent in floating point —
so the score is bit-identical too. Only `pnorm_3` sums, and summation order is
where the ~1e-7…1e-6 comes from. Body-invariance (2,118/2,118) is the direct
corollary: changing where the strip boundaries fall cannot move a value that
never depended on them.

---

## The other axis: backend choice (NOT a strip question)

Flipping `metric_orchestrator_eligible` does more than swap whole→strip. The
orchestrator's chooser also picks the **backend**, and `prefer_cpu` from
`OrchestratorMetricSpec::from_cli` is **not** forwarded into the `Task` — it only
drives a build-config error message (`orchestrator_glue.rs:228`). `chooser.rs`
lists butter under `cpu_wins_oneshot_max_pixels == u64::MAX`, so in
`ExecContext::OneShot` a butter task prefers **CPU** at any size when
`cpu-butter` is compiled in (`chooser.rs:346`). In `ExecContext::Batch` — the
fleet/sweep default — ranking is on warm `ns_per_px` and GPU wins.

This is a pre-existing property shared by all ten currently-eligible metric
kinds, not something specific to butter. Measured here so the size of it is on
record rather than assumed:

- CPU `butteraugli_strip(256 rows)` vs CPU `butteraugli` (whole): **max 9.37e-8**
  over 1,080 cells — the CPU adapter's strip walker is exact too.
- GPU multires whole vs CPU `butteraugli` (whole): **median 2.02e-4, p95 2.37e-4,
  p99 1.46e-3, max 1.29e-2** (`photo-gb82/mc3-lossless.png` q30: CPU 4.3905 vs
  GPU 4.3341).

So the CPU↔GPU envelope is ~2e-4 typical but has a ~1.3e-2 tail — two to four
orders of magnitude larger than anything the strip walker contributes. If
bit-identical butter columns matter for a given sweep, the lever is backend
pinning (or `--use-legacy-scheduler`), not the strip mode.

---

## What is still open

1. **CUDA is not measured here.** Everything above is `cubecl-wgpu`/Metal with
   the portable reduction. The 54-cell CUDA parity sweep
   (`scripts/orchestrator_parity_sweep.py`) named in the old
   `metric_orchestrator_eligible` doc comment has not been re-run on a real card.
   The bit-identity argument (exact halo → identical diffmap → exact `max` fold)
   is backend-independent, but `--features fast-reduction` (CUDA-only opt-in)
   replaces the deterministic fold with `Atomic<f32>::fetch_add`, whose ordering
   is nondeterministic — under that flag neither the whole nor the strip path is
   reproducible run-to-run and this measurement says nothing.
2. **Backend choice**, per the section above — a policy question for the owner,
   not a defect found here.
3. `phoboslab.org.png` (27 MP) and other >11 MP cells were not measured; the
   largest cell here is 10.5 MP / 75 strips.
