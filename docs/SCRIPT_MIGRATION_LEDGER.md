# Script → owner migration ledger (started 2026-08-27)

The standing rule (zensim CLAUDE.md "NO DUPLICATE IMPLEMENTATIONS"): every task
has ONE owner; scripts that re-derive an owner's semantics migrate INTO the
owner, with a parity gate, and are deleted the same commit their callers move.
This ledger records what moved, what stays (with the reason), and what is
queued — so the next session extends the owners instead of the script pile.

## Migrated (wave 1, 2026-08-27)

| script | owner now | parity evidence |
|---|---|---|
| `pool_reconcile_report.py` | `zenfleet-ctl report` (`account_rows` in the ctl lib, unit-tested) | declared exact (193,574), done monotone on the live draining run; byte-compatible TOTAL/VERDICT lines; 1s vs 11s |
| `compact_ledgers.py` | `zenfleet-ctl compact` (`snapshot_rows`; `--upload` writes the ONE run-root key both worker modes read) | python-done ⊆ rust-done on the draining run (87,223 ⊇ 84,820), done∩failed = ∅; zstd by construction (write_ledger) |

## Migrated (wave 2, 2026-08-27)

| script | owner now | parity evidence |
|---|---|---|
| `pool_progress.py` | `zenfleet-ctl progress` (`parquet_num_rows` in zenfleet-ledger — footer-only) | EXACT: 490,173/490,173 from 66/66 snapshots, both tools, R2 pools |
| `pairs_from_encode_ledger.py` | `zenfleet-ctl pairs` (LedgerView latest-wins — the owner semantics the Python approximated with a ts>= scan) | key-set equal on avifgen-enc (562,857) modulo ONE hand-appended SNAPPY sidecar (3 rows) the rust reader correctly refuses (ledger parquet is zstd-only by contract); recompressed replacement staged, upload pending store space |
| `declare_direct_objects.py` | `zenfleet-ctl declare-scorefiles` (`declare_scorefile_jobs`, unit-tested; `Sha256Hex::raw_object_key` documents the direct-object input contract the wire always had) | canon-identical on a 2,000-row slice for score_file; diffmap identical EXCEPT two Python bugs fixed: (1) jobs now carry invariant-5 `requires` tokens (hand-rolled JSON never did); (2) per-row TRUE cell identity — the Python's member→identity dict did last-write-wins when a content-addressed member appears under multiple cells (62 such members per 2,000 rows measured) |

Also fixed at owners en route: gunzip/gzip child-process pipe deadlock (writer
thread; 5 MB in / 97 MB out fills the pipe while a same-thread `write_all`
blocks); endpoint precedence honors the resolver's `EP` contract output
(preferring ambient `ZEN_S3_ENDPOINT` broke `ZEN_STORE=r2`: LAN endpoint + R2
creds → 403).

## Kept as scripts, with reasons

| script | why it stays |
|---|---|
| `fleet_sentinel.sh`, `lan_score_launch.sh`, `launch_fleet.sh`, `pool_launch.sh`, `hetzner_scorefile_launch.sh`, `build_executor_image*.sh`, `gpu_*` diag shells | ssh/docker orchestration — shell IS the native tool; they call the CLIs, they don't re-derive semantics |
| `refresh_snapshots.sh` | cron wrapper around `zenfleet-ctl compact --upload` (migrated caller) |
| `hdrgrid_cells.py`, `declare_bf944*.py`, `avifgen_*.py` | campaign DATA generators — they EMIT the inputs the ctl declare commands consume (cells JSONL / manifest rewrites); one-shot, provenance-documented |
| `fleet_top.py`, `fleet_power.py` | operator TUIs/reports composing owner mechanisms (snapshot footers, ledgers) for display; python-owner domain |
| `writeback_scores.py` | **wave-3 queue** (below) |
| `build_scorefile_from_pairs.py`, `build_scorefile_hdr_pairs.py` | **MIGRATED-IN-PLACE 2026-08-27** (the exemplar pattern): tar/sha ETL stays python; DesiredJob emission via `zenfleet-ctl declare-scorefiles` bridge parquets — from_pairs carries per-src codec columns (multi-codec corpora, CLI uses identity codec), hdr_pairs uses --full-uri verbatim keys + --cell-knobs scorefile-hdr + the hdr_transfer id-minimality conditional |
| `build_scorefile_manifest.py` | **wave-3 exemplar, MIGRATED-IN-PLACE 2026-08-27**: keeps its tar/sha ETL, hands DesiredJob emission to `zenfleet-ctl declare-scorefiles` via a bridge parquet (fixture parity: canon-identical incl. chunk boundaries, modulo the stamped `requires`); the CLI gained control.json CREATE-IF-ABSENT (the unconditional write would unpause an existing run on re-declare) and snappy for pairs READS only (pyarrow's default; the ledger contract stays zstd-only) |

## Wave-4 queue (recorded, not started)

- `writeback_scores.py` vs `zenmetrics assemble` — **AUDITED 2026-08-27,
  verdict: complementary today, one real seam.** assemble owns the typed
  full-key JOIN (PairKey's four-field constructor makes the ref-only collapse
  inexpressible; safe_join errors on dupes; leak + constant-per-ref guards —
  the structural answer to the 2026-05-25 corruption). writeback owns
  blob-stream INGEST (ScoreFile JSONL → per-(sha, metric) score maps +
  regime-pure batched feature writers) — python-natural ETL, not a join
  duplicate. The SEAM: writeback's cells→sha→scores join builds its own key
  and guards (regime width purity) instead of assemble's typed key + guards.
  Wave-4 shape: either expose assemble's join/guards as a callable surface
  the ingest shells, or grow assemble an `ingest-jsonl` mode; writeback's
  ingest half stays python either way.
- `backfill_overnight_manager.py`, `fleet_smoke_image.py`: classify on next touch.
