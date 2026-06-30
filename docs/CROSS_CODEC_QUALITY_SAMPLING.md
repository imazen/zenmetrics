# Cross-codec quality sampling — making learning + eval math valid

**The bug this prevents (2026-06-30):** the picker oracle credited AVIF above zq90 by default —
not because AVIF was best, but because the sweep had no webp/jxl data there. The corrected
(support-aware) oracle inverts it: lossy family mix **jxl 45% / webp 31% / avif 23% / jpeg 1%**,
not the biased avif 35% / jxl 35%.

## Why "more data" can't fix it

The sweep dials a **generic `q` (1–100)** that each codec resolves to its OWN native control
(`quality_to_quantizer` for jpeg, `resolve_distance_for_quality` for avif/jxl). The q→achieved-
quality map is **continuous and codec-specific**, so:

- Equal `q` is NOT equal achieved quality (`q=90`: jpeg→~zq94, avif→~zq96, jxl→~zq90).
- Two codecs will **never** have the same sample density at a given target zq/ssim2 — that's
  geometric, not a budget problem. You cannot re-sweep your way to identical density.

The picker/oracle work in **achieved-quality space** (target zq). So a fixed-`q` sweep gives
ragged, non-comparable achieved-quality coverage, and the oracle `min(bytes at target zq)`
silently drops codecs that have no sample there → biased labels that eval can't see (it scores
against the same biased oracle). Same failure family as the dropped experimental features and
the lossless-png contamination: **internally-consistent math on poisoned data.**

## The two-layer fix (both required)

### 1. DATA LAYER — make the math valid given the inevitable density mismatch (SHIPPED)

`scripts/picker/picker_data.py` is the canonical builder; every picker trainer routes through it.

- Resample each codec's RD curve onto a **common achieved-quality target grid** (zq or ssim2).
- A codec is present at a target ONLY with **measured support** — the target lies inside its
  measured [min,max] achieved range, so `bytes_at` interpolates, **never extrapolates**.
- Build the oracle/label ONLY on cells whose **required support set is complete**
  (`oracle_rows(require='all')`); else **exclude + count by missing codec**. A `min` over an
  incomplete support set is a biased label.
- **Gates** (CLI/script safety): `picker_data.assert_quality_parity()` (in-process) and
  `scripts/picker/check_quality_coverage.py` (standalone, CI-able) FAIL when high-band coverage
  is too asymmetric to build complete cells. Run before any train/eval.

This makes the comparison valid by **masking** the under-covered band (honest, but the picker
then can't choose there — defer to the gate/lossless).

### 2. SWEEP LAYER — quality-targeted sampling, to FILL the band (the real sampling fix; TODO)

Stop sweeping a shared `q` grid. Instead, for each `(source, codec, target_quality)` on a common
grid (e.g. zq {50,55,…,97} / ssim2 analog), **binary-search the codec's native param to hit the
target** (±tol, a few encode+score probes), encode, and **record the ACHIEVED quality** (not the
requested `q`). Then every codec has a measured sample at every target → support is complete →
the unbiased oracle covers the full range, not just ~zq82.

- New sweep mode (`zenmetrics sweep --quality-targets zq:50,55,…,97 --quality-tol 0.5`), a
  target-quality search loop around the existing per-codec encode+metric. ~3–6 extra encodes per
  (source,codec,target) for the search — bounded, and it's the only way cross-codec RD math is
  honest at the top end. Record `achieved_quality` alongside `requested_param`.
- Until it lands, the data layer masks the gap and `check_quality_coverage.py` keeps it honest.

## The rule

**Never compare codecs on a shared encoder-`q` grid. Sample and gate in the space you compare in
(achieved quality), require cross-format support parity before trusting an oracle, and prefer
quality-targeted sampling so support is complete by construction.** (Belongs in the CLAUDE.md
sweep discipline next to the low-q-density rule — that rule guarded the low end and left high-end
cross-codec parity unguarded, which is exactly the hole that bit.)
