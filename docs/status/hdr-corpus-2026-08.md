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
| **Phase** | node flips done; grid pre-registration in progress |
| **Blocking on** | appendix S registration, then zenfleet capability survey |
| **Ledger fill** | n/a (not declared yet) |
| **First-cell R2 persistence gate** | NOT YET RUN — hard gate before any fleet scale-up |

### Milestones

- [x] Standing-rule change recorded (node-2/node-3 permanently Ubuntu)
- [x] node-2 + node-3 PXE-flipped to Ubuntu and verified up
- [ ] Appendix S grid pre-registered
- [ ] zenfleet extended for HDR multi-codec encode + score (if needed)
- [ ] Workers enrolled on all encode nodes
- [ ] **First-cell artifact-persistence gate PASSED** (encoded bytes + diffmaps + all metric variants confirmed on R2)
- [ ] Fleet scaled to full grid
- [ ] 50% ledger fill
- [ ] Corpus complete + `_MANIFEST.json` + orientation gate + provenance registered

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
