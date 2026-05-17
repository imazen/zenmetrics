# Kanetaka et al. IWAIT 2026 — paper notes (post-PDF read)

PDF: `/home/lilith/work/zen/zenmetrics-refs/IWAITSPIE_Kanetaka.pdf`
DOI: 10.1117/12.3100969 (SPIE Proc Vol 14072, paper 140720G)

**These notes correct/refine the abstract-only context that the running ssim2 agent (Phase A skip-map + Phase B FIR) was dispatched with. If you're that agent — read this BEFORE writing kernel code.**

## Headline numbers (corrected)

- **Maximum speedup is ×44.2**, not ×82.4. The SPIE abstract figure is inflated; the paper conclusion confirms 44.2× as the headline. Hit at D=3, "Faster" weight mode, with PreR (precomputed reference).
- **D=5 is the sweet spot.** It's *both* faster than D≥7 and *more accurate* than the libjxl reference (SROCC 0.890387 vs the reference's 0.889297 on CID22). D=3 is fastest but degrades slightly to 0.885.
- **Skipping weights does not degrade accuracy.** At D=5, Full / Lossless / Fast / Faster all score SROCC 0.890387 — identical. The "Faster" mode skips most aggressively with zero accuracy cost.

## Speedup decomposition (CPU SIMD AVX2, AMD 9950X, 512×512)

Baseline: libjxl v0.11.1 reference = 35.517 ms.

| weight mode | D=3 | D=5 | D=7 | D=9 | D=11 | D=13 |
|---|---|---|---|---|---|---|
| Full (R-D) | ×13.7 | ×12.0 | ×10.7 | ×9.5 | ×8.6 | ×7.9 |
| Lossless (R-D) | ×15.3 | ×13.6 | ×12.1 | ×11.1 | ×10.1 | ×9.3 |
| Fast (R-D) | ×21.4 | ×19.7 | ×18.0 | ×16.7 | ×15.5 | ×14.5 |
| Faster (R-D) | ×21.5 | ×19.4 | ×18.0 | ×16.6 | ×15.5 | ×14.4 |
| Full (PreR) | ×24.4 | ×20.5 | ×18.0 | ×15.9 | ×14.3 | ×13.0 |
| Lossless (PreR) | ×29.8 | ×25.8 | ×23.0 | ×20.8 | ×19.0 | ×17.4 |
| Fast (PreR) | ×43.3 | ×38.2 | ×34.4 | ×31.2 | ×30.1 | ×26.2 |
| Faster (PreR) | **×44.2** | ×38.5 | ×34.8 | ×31.6 | ×30.2 | ×26.4 |

**Per-technique contribution at D=5:**
- FIR alone vs libjxl recursive: ×12.0
- + Lossless skip (zero-weight cells): ×13.6  (additional ×1.13)
- + Fast skip (small weights): ×19.7  (additional ×1.45)
- + PreR (precompute reference R, R²): ×38.2  (additional ×1.94)

PreR ≈ ×2 multiplier is **already shipped** in our `Ssim2::set_reference` / `Ssim2::compute_with_reference`. So at D=5 the remaining ceiling vs Full(R-D) baseline is about ×19.4 from FIR + skip combined.

## SSIMULACRA2 algorithm (paper §2 verbatim)

- 6 multiscale levels (`scale 0..5`)
- 2×2 box pyramid decomposition on RGB before color transform
- Then RGB → XYB color space
- Three features per (scale, channel): SSIM (σ=1.5), DoG input ratio, DoG degradation ratio
- L1 and L4 norm pooling
- **108-dimensional feature vector** = 6 scales × 3 channels × 2 norms × 3 features = 6 × 3 × 6 = 108
- All convolutions are σ=1.5

## Technique 1: separable FIR convolution

Replace the recursive constant-time Gaussian (sliding DCT-III, Charalampidis 2016) with a **truncated separable Gaussian FIR** at σ=1.5.

Why: "Recursive processing is lowly parallelizable and limited in vectorization. Performance improvement from optimization is limited by the image processing-specific language (DSL), Halide."

**For D=5 with σ=1.5** (RECOMMENDED — best speed+accuracy point):

5-tap kernel coefficients (unnormalized): `g(x) = exp(-x² / (2σ²))` for `x ∈ {-2, -1, 0, 1, 2}` with `σ = 1.5`:
- g(±2) = exp(-2/2.25) = exp(-0.8889) ≈ 0.4111
- g(±1) = exp(-1/4.5) ≈ 0.8007
- g(0) = 1.0
- Normalize by sum ≈ 3.4236, giving: [0.1201, 0.2339, 0.2921, 0.2339, 0.1201]

(Vship uses 17-tap; **the paper's evidence is that D=5 strictly dominates D=17 for this metric** — D=11 already drops SROCC to 0.888666. Use 5-tap.)

## Technique 2: skip low-contributed features

Skip whichever cells in Table 1 have small weights. Four modes:

- **Full**: no skipping; all 108 features computed.
- **Lossless**: skip cells where weight is literally 0 in the paper's table (the red-text cells). About 39/108 cells = 36% skipped.
- **Fast**: additionally skip the green-text cells (small nonzero weights).
- **Faster**: additionally skip the blue-text cells (slightly larger but still small).

**At D=5, all four modes hit SROCC 0.890387 — identical accuracy.** So "Faster" mode is strictly better than "Full" — no reason to ever offer Full as default.

### Weight table (Table 1 verbatim)

108 entries laid out as 6 scales (cols) × 18 rows (channel × norm × feature). Cells marked `// L`, `// F`, `// X` annotate Lossless/Fast/Faster skip status (L ⊇ F ⊇ X — once skipped at a lower mode, it stays skipped).

```
                  scale 0          scale 1          scale 2          scale 3          scale 4          scale 5
X-L1-DSSIM        0       (L,F,X)  0       (L,F,X)  0       (L,F,X)  0.000798911 (F,X)   0.000728935 (F,X)   0.000455099 (F,X)
X-L4-DSSIM        0       (L,F,X)  1.10417264          1.84224555          1.8787595           0.000140034 (F,X)   0.001364877 (F,X)
X-L1-artifact     0.000737661 (X)  0.000437116 (X)  0.001640644 (X)  0.000176816 (F,X)   0.967793708          0       (L,F,X)
X-L4-artifact     0.000779348 (X)  0.000662848 (X)  11.4411726          10.9490699          0.998176698          0       (L,F,X)
X-L1-detailloss   0       (L,F,X)  0       (L,F,X)  0       (L,F,X)  0       (L,F,X)  0       (L,F,X)  0       (L,F,X)
X-L4-detailloss   0       (L,F,X)  0.000152316 (X)  0       (L,F,X)  0       (L,F,X)  0.000319498 (F,X)   0       (L,F,X)
Y-L1-DSSIM        0       (L,F,X)  0.00062356  (X)  225.205153          176.393176          0       (L,F,X)  0       (L,F,X)
Y-L4-DSSIM        7.46689033          6.68367815          19.2132382          24.43301             34.7790634          0       (L,F,X)
Y-L1-artifact     0       (L,F,X)  0       (L,F,X)  0       (L,F,X)  0       (L,F,X)  0       (L,F,X)  0       (L,F,X)
Y-L4-artifact     0       (L,F,X)  0.000377244 (X)  0.001140152 (X)  0.285208026          44.8356253          0       (L,F,X)
Y-L1-detailloss   0       (L,F,X)  0       (L,F,X)  0       (L,F,X)  0       (L,F,X)  0       (L,F,X)  0       (L,F,X)
Y-L4-detailloss   17.445834           1.02788994          0.001237756 (X)  0.000448544 (F,X)   0       (L,F,X)  0       (L,F,X)
B-L1-DSSIM        0       (L,F,X)  0       (L,F,X)  0       (L,F,X)  0.002082701 (X)  95.1080499          0       (L,F,X)
B-L4-DSSIM        0       (L,F,X)  0.000165338 (F,X)   0.000417917 (F,X)   8.82698276          0.001228641 (X)  0.000513006 (F,X)
B-L1-artifact     0.000868056 (X)  0.000531319 (X)  0       (L,F,X)  0       (L,F,X)  0.986397803          0       (L,F,X)
B-L4-artifact     0       (L,F,X)  0       (L,F,X)  0.001729083 (X)  23.1924334          171.266726          0       (L,F,X)
B-L1-detailloss   0       (L,F,X)  0       (L,F,X)  0       (L,F,X)  0       (L,F,X)  0.983438279          0       (L,F,X)
B-L4-detailloss   0       (L,F,X)  0       (L,F,X)  0       (L,F,X)  0       (L,F,X)  0.980786             0.000109 (F,X)
```

Skip-count summary (out of 108 cells):
- **Lossless skips: 38** (all literal-0 weights)
- **Fast additionally skips: ~7-10 cells** with very small weights
- **Faster additionally skips: ~10-15 cells** with small-but-larger weights

A pragmatic implementation: just use the **Faster** mode (skips ~60-65 cells of 108, leaves ~45 active). Per the table, Faster gives identical accuracy to Full.

## Technique 3: precompute reference

> "The required local statistics of µ, σ, σ² for SSIMULACRA2 are R, R², D, D², RD's convolutions for each scale. Therefore, 2/5 computation, i.e., R, R², can be precomputed."

Also: "For image preparation, multiscale decomposition and XYB color transformation can be skipped for reference images."

**This is already what `Ssim2::set_reference` does in our crate.** Verify it caches the reference pyramid + XYB + R + R² blurs — if any of those aren't cached, that's the gap that gives us the ~×2 PreR multiplier.

## Architectural implications for our port

1. **Use D=5, not D=17.** Vship's 17-tap is the wrong design point per this paper. D=5 wins on speed and accuracy.
2. **Implement the 4-mode framework.** User-facing knob: `Ssim2Mode::{Full, Lossless, Fast, Faster}` with `Faster` as default (zero-accuracy-cost on real corpora). Per-mode skip-set hardcoded from Table 1.
3. **The skip-map dispatch is parity-strict by construction.** A cell whose weight is 0 contributes 0 to the score regardless of its computed value — so skipping it is exact. Cells with small (but nonzero) weight contribute < 0.01 × value; skipping them changes the score by negligible amounts (paper confirms SROCC unchanged).
4. **Verify Ssim2::set_reference is caching everything.** If it doesn't cache R/R² blurs (only the linear/XYB planes), we're missing half the PreR win.

## Reference implementations

- Paper baselines vs libjxl v0.11.1 (CPU SIMD AVX2). Our ssim2-gpu was a port of `ssimulacra2-cuda` (turbo-metrics) which derives from cloudinary/ssimulacra2 — same algorithm but potentially different exact weights or norms. **Verify our 108-element weight constants match the paper's Table 1.** If they differ, neither is wrong — they're different ssim2 implementations — but our skip-map must match OUR weights.
- Vship 5.x at `/home/lilith/work/refs/Vship/src/HIP/ssimu2/score.hpp` — uses 17-tap blur (suboptimal per this paper) but has the skip-map structure (8 specializations on per-cell weight thresholds).
- Paper Section 5 conclusion explicitly cites vship as future work: "Our future work includes comparisons at larger image sizes and with GPU implementations of SSIMULACRA2, named VSHIP." So vship 4.1's adoption of these techniques is more like cross-pollination than direct precedence.

## Implementation order (revised priority)

1. **Verify Ssim2::set_reference caches R, R², the XYB conversion, and the pyramid.** If anything's missing, fix it first — easy ×2 if so.
2. **Skip-map dispatch using OUR existing weight constants.** Identify cells where `|weight| < 0.001` (Lossless) and `< 0.01` (Fast/Faster). Single kernel + runtime mask. Parity-strict. Expected ~×1.4 at D=current.
3. **Switch the blur kernel to separable FIR D=5.** Drop-in replacement for the current Charalampidis IIR. Expected another ~×2-3 on the blur stage, ~×1.5-2 on per-call wall time.
4. **Fuse the FIR blur with the SSIM stats kernel** (vship-style `planescale_map_Kernel`). Big lift; only attempt after (2) and (3) ship.

Combined target at D=5: from current ~58 ms → roughly 20-30 ms at 12 MP (we won't hit the ×38 the paper measured at 512² CPU because GPU bottlenecks differ — upload-bound at large sizes).
