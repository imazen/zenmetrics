# AVIF Re-encode — Value-Optimized Plan (2026-07-02)

**Status: PLAN, ready to fire — NO prerequisite blocker (§1 corrected 2026-07-02).**
Backed by a live Hetzner box-value benchmark (2026-07-02) + live server-type
pricing/availability.

## 1. CORRECTED 2026-07-02 — the fixed zenrav1e IS already wired in; there is no blocker
**The production re-encode goes through `zenmetrics-cli`'s own sweep path
(`zenavif` crate, path dep) — it never touches `cavif`/`ravif` directly, and the
prior "wire it into cavif" framing below was simply the wrong crate to check.**
`zenmetrics/Cargo.toml`'s own `[patch.crates-io]` already pins
`zenrav1e = { path = "../zenrav1e" }`, which overrides crates.io for EVERY crate
in the dependency graph that pulls it — including `ravif`'s (zenavif's dep)
`zenrav1e = "0.1.4"` pin. **Verified empirically, not assumed:**
- `Cargo.lock`: `zenrav1e` resolves to `0.2.0` with no `source = "registry+…"`
  line (i.e. it's the local path dep, not crates.io).
- `../zenrav1e` HEAD (`2fac1af6`) has the RD-fix commit `4563cc5b`
  (partition-search: `PARTITION_HORZ/VERT` + `HORZ_4/VERT_4` + ordinal-gate)
  as an ancestor — `git merge-base --is-ancestor 4563cc5b HEAD` confirmed.

**So any normal `zenmetrics-cli` build (the sweep/fill/jobexec binary — the
actual re-encode vehicle per §7) already gets the RD-fixed encoder. No release,
no patch edit, no rebuild-with-different-flags needed. The re-run can fire
immediately.**

<details><summary>Superseded original §1 (WRONG — kept for the record, see verify-premises-before-cascading memory)</summary>

Original claim: "the RD win... is NOT wired into any cavif build today: `ravif/ravif/Cargo.toml`
pins `zenrav1e = "0.1.4"` (crates.io) with no `[patch]`. A fresh `cargo build` of cavif
produces the OLD (+2.1%) encoder." This was true of a *standalone* `cavif` build in isolation,
but wrong for the actual production path: it never checked whether the CONSUMING workspace
(zenmetrics) patches `zenrav1e` at a higher level, which it does. Checking a dependency's own
Cargo.toml pin without checking the top-level workspace's `[patch]` table is the mistake —
Cargo patches apply transitively from the root workspace down, regardless of what a
mid-graph crate's own manifest says.
</details>

## 2. Box-type value — MEASURED (2026-07-02, size-normalized Mpix/€)
**Method note:** this box-value comparison used the standalone `cavif` CLI as a throughput
PROXY for AV1 encode cost (cheap to ship as one static binary for a quick cross-box bench) —
it is NOT the production re-encode path (that's `zenmetrics-cli` → `zenavif`, per §1/§7).
cavif and zenavif both ultimately drive rav1e-family encoding, so relative box value
(cx23 vs ccx vs cpx) should transfer, but the ABSOLUTE Mpix/s numbers below are cavif's, not
zenavif's — re-verify per-core throughput with the real `zenmetrics-cli` sweep binary before
finalizing box count, since zenavif's mandatory-axis modes_full grid (§5) costs more per cell
than cavif's default single-mode encode.
**Ship the PRE-COMPILED cavif binary** (glibc 2.35, run in an `ubuntu:22.04` container), do NOT
build on-box — on-box builds failed twice (missing nasm; then a mktemp bug) before the binary
approach yielded valid numbers. `-j 1` per encode × all cores parallel.
Corpus 12 imgs (4.14 Mpix, 64²–1024²), speeds {2,4,6}. `Mpix/€ = mpix_per_s × 3600 ÷ €/hr`.

| box | kind | cores | €/hr | region | Mpix/s @s6 | **Mpix/€ @s6** | Mpix/€ @s4 | Mpix/s/core @s6 |
|-----|------|-------|------|--------|------------|----------------|------------|-----------------|
| **cx23** | shared Intel | 2 | 0.0104 | **EU nbg1** | 1.12 | **386,342** | 41,054 | 0.56 |
| cpx51 | shared AMD | 16 | 0.134 | US hil | 10.55 | 283,304 | 31,554 | 0.66 |
| ccx33 | dedicated AMD | 8 | 0.261 | EU hel1 | 4.99 | 68,771 | 7,953 | 0.62 |
| cax (arm) | shared arm | — | €0.005/core | EU fsn1 | _not-yet-measured_ | _≥ cx23 expected_ | — | — |

**Measured findings:**
- **cx23 (cheap EU Intel) is the value WINNER — 386k Mpix/€, beating US-only cpx (283k) and dedicated
  ccx (69k) at every speed.** Intel per-core throughput (0.56) is only ~12% below AMD (0.62–0.66) but
  cx is 6.4× cheaper/core than ccx → **cx23 is 5.6× better value than dedicated ccx, 1.4× better than cpx.**
- **Shared ≫ dedicated:** the shared-vCPU-throttle worry did NOT materialize (cpx 0.66 vs ccx 0.62
  per core). **Dedicated ccx is the WORST value tested** — do not use it for value.
- **Speed is a ~60× lever** (ccx s2 0.08 → s6 4.99 Mpix/s). Choose the re-run speed deliberately.
- **cx23 caveat (a finding):** 2 cores → low per-box throughput (1.12 Mpix/s @s6; s2 is ~55 min for
  the K=16 corpus). Fan WIDE (128-server cap → ~120 boxes), don't rely on per-box speed.
- **arm (cax) is the cheapest per core (€0.005 < cx23's €0.0052) and likely edges cx23 on value**, but
  wasn't measured (new-cax creates were capacity-out during the bench).
  **DECISION (user, 2026-07-02): do NOT pursue arm — cx23 is the pick.** The ~4% per-core price edge
  doesn't justify the arm64-image + arm64-cavif work vs cx23's already-decisive win (5.6× over
  dedicated) and reliable EU x86 launchability. The persistent cax dev boxes (`zen-arm-dev`,
  `zen-arm-big`) are NOT to be torn down or commandeered.

## 3. Availability reality (2026-07-02 — gates the plan more than raw value)
- **arm (cax): capacity-out** across all EU sizes/locations — cheapest per core but UNOBTAINABLE now.
- **shared AMD (cpx): EU-discontinued** ("can no longer be ordered"); only US `hil` left → the value
  winner is NOT an EU option.
- **cx line: only cx23 reliably available** in EU (user, 2026-07-02); cx43/cx53 provisioned for the
  bench but aren't broadly obtainable.
- **dedicated AMD (ccx): EU-available**, pricey.
- **⇒ the reliably-scalable EU options are `cx23` (cheap, small, EU) and `ccx*` (pricey, big, EU).**

## 4. Recommended fleet
- **Box: cx23 (EU, cheap Intel) is the value pick** — €0.0052/core-hr is 6.4× cheaper than ccx's
  €0.033, and the cpx result shows shared-vCPU per-core throughput ≈ dedicated (no catastrophic
  throttle), so cx should win EU value handily. **Confirm cx per-core throughput on the smoke** (§2's
  cx23 K=2 number, pending); if Intel per-core is within ~2× of AMD, cx23 wins. Fall back to ccx43
  only if cx per-core collapses.
- **Region:** EU default (`fsn1 nbg1 hel1`) — already committed (`585f06d6`).
- **Scale (raised caps, `d5a142e0`):** 128 servers / 640 dedicated vCPU / 4 TB volumes. cx23 is SHARED
  → counts only against the 128-server cap → up to ~120× cx23. ccx is DEDICATED → 640-vCPU cap
  (e.g. 20× ccx33 / 13× ccx43). Fan cx23 wide to offset its low per-box throughput.
- **Bid vs on-demand:** Hetzner has **no spot/bid** — on-demand only. (The bid lever is vast-only, for
  the GPU-scoring half — ~40–60% cheaper there; use it.)
- **RAM caveat:** cx23 = 4 GB. avif ≤1024² is safe; if the corpus has larger renditions, cap per-box
  `SWEEP_JOBS` so N×peak-encode-RSS < 4 GB.

## 5. Encode config (mandatory-axis + persist-everything)
- `PLAN=modes_full` — covers the mandatory RGB/420/444 + qm axes (picker-mandatory-axis rule; do NOT
  ship a subsampling-crippled avif set).
- q-grid: web-weighted, **dense low-q** (q5–q60 at the same density as q60–q100 — sweep discipline).
- **Persist everything** (HQ-fill lesson): encoded variants as per-box tars (→ feeds the tar-shard GPU
  scoring), 372 zensim features, ALL 6 metrics, content-addressed. New dated canonical with
  `codec_commit=<fixed rev>`; ARCHIVE the old `22a58d58` avif; never mix revs (codec-rev-pairing).
- **Encode-once/score-many:** Hetzner CPU encodes+persists → score 6 metrics either (a) in-place on
  the encode box (CPU butteraugli fixed, `0b5c3eb5`) or (b) GPU via the tar-shard/warm-ref fill
  (`86e9b0b7`). In-place CPU avoids the 219 GiB variant re-download.
- **Speed:** pick deliberately (§2: ~60× lever). Match the canonical avif sweep's speed for consistency.

## 6. Sizing (fill at launch, from the confirmed per-box Mpix/s)
- clean-picker-corpus: `N` renditions × avif cells (modes × q-grid) = `M` encodes = `P` Mpix.
- At `<confirmed Mpix/s/box> × <box count>` → `<wall h>`, `<€ total>`. **Measure, don't extrapolate.**

## 7. Reuse (don't hand-roll)
`scripts/sweep/hetzner_cpu_sweep.sh CODEC=zenavif PLAN=modes_full` (EU default, variants→R2 tars).
Set `TYPES=cx23` (+ `STYPE`). Feeds the tar-shard scoring pipeline. This is the actual production
path — it builds/ships the `zenmetrics-cli` binary (which links `zenavif`, already carrying the
fixed `zenrav1e` per §1), NOT `cavif` — `cavif` was only used for the isolated §2 box-value bench.
