# `cvvdp-harvest` worklog — 2026-09-24 UTC

Lane scope: harvest the frozen `cvvdp-safesyn-20260923` ScoreFile run. All result data are TRAIN-role metric labels. No human labels or sealed holdouts were read. Store selection uses `ZEN_STORE=tower` and `scripts/lib/s3env.sh`; credentials and endpoint are not recorded here. The parent is the frozen zenmetrics commit `9f36f88b8a23e645bd791c1193842d950eccf4be`. This lane uses `quarantine/codex/cvvdp-harvest`; the Devin bookmark remains at the frozen parent.

## Inputs and provenance

- `FLEET_RUN.md`: 3,218 jobs, 196,086 pairs, four score arms, image digest `sha256:d110a3a79d98f06ade7c7dae3920c47b71c10362c959bd852341f6c5ff046050`.
- `build_meta.json` sha256 `59c460ccd6e3f06ddf34f0f704b1a5448b6fd2811d3cb0e5b271c873c423dc00`; it names zenmetrics `9f36f88b`, zensim linked commit `353e6bcc`, 13 sibling repositories, and the baked decode route.
- Sept. 14 audit JSONL sha256 `a6766a5aa973c73c3e9c8d0e94a86bb1034bde691a6e21a7f4b2a7ef96064b22`; SafeSyn pairs TSV sha256 `5a53976070a5e21b2bb7fe0d05f58b510b93e141dd9207dd3e2cd337fd15cd3b`.
- Incremental verifier `verify_progress.log` sha256 `fcb611e8504ca0390eabca7d06f81ceb81e4db3ce2db69beeed68297c5b37c6f`: final line `02:58Z +31 blobs +7616 rows (cum 784344) err=+0 px_missing=+0 px_bad=+0`. This log triggered the lane, but the independent store check below is the completion authority.

## Harvest code gate

- Cwd: `/home/lilith/work/zen/zenmetrics--cvvdp-safesyn`. Commands: `diff -u /var/tmp/cvvdp-safesyn/harvest_safesyn.py benchmarks/cvvdp_harvest/harvest_safesyn.py`, `python3 -m py_compile benchmarks/cvvdp_harvest/harvest_safesyn.py`, and `jj describe` on this lane's child commit. Exit codes: diff 1 (expected differences), compile 0, describe 0. Corrected script sha256 `d45068d3e71c1851e043d40c6018b30b31262c7bbb19c65e8ad51f37d5697370` at commit `e2f3aa7c`.
- Review of the patch: finite scores required; unknown or unscored metric rows and unmapped input URIs fail; duplicate score floats are compared by IEEE-754 bytes; zero errors, zero pixel-hash omissions/mismatches, and the exact frozen build commit are required; the output manifest records the sidecar hash and all 13 sibling repositories. The corrected script was copied to `/var/tmp/cvvdp-safesyn/harvest_safesyn.py` before execution.
- The first copy entered jj's mutable working-copy commit and briefly moved the Devin bookmark. I restored the original tree, reset that bookmark to the exact original `9f36f88b`, abandoned my orphaned rewrite, and then created this lane as a child. Final bookmark check: Devin `9f36f88b`; Codex descends from it. No source diff remains on Devin's bookmark.

## Direct store completeness (read-only, before harvest)

- UTC 04:10:19–04:10:20; cwd as above; command: `source scripts/lib/s3env.sh; for kind in ledger claims blobs; do s5cmd --endpoint-url "$EP" ls "$base/$kind/" > "/var/tmp/cvvdp-safesyn/precheck_${kind}.ls"; done`. Exit 0. Output lines: `ledger=217`, `claims=216`, `blobs=3218`. Listing sha256: ledger `27dc100666a46af4c234635aca77465ad6228dc324a8f89617cef62e4907ec39`; claims `25f14698fa24d87c9be57d400f1cfbc23d799f3f3242abb42bf996a4af920286`; blobs `507fefc311905b20c44ffaa592aef7aab6c36db58aeaad4445ac7fcacd93bb59`.
- UTC 04:10:37–04:10:38; cwd as above; commands: `s5cmd --endpoint-url "$EP" cp "$base/manifest.json" "$out/store_manifest.json"`; `s5cmd --endpoint-url "$EP" cp "$base/ledger/*" "$out/store_ledger/"`. Exit 0. Output: `manifest_bytes=20679222 ledger_files=217`. Store manifest sha256 `9a1e1eb93df76a8c143038d9db59babee337e7f0f76322adf679dbfa21d61a49`, exactly equal to the declared local manifest.
- UTC 04:10:41–04:10:42; cwd as above; command: `python3 benchmarks/cvvdp_harvest/verify_store.py --dir /var/tmp/cvvdp-safesyn | tee /var/tmp/cvvdp-safesyn/store_verify.log`; exit 0. Exact output: `manifest_jobs=3218 ledger_rows=3222 ledger_jobs=3218 blob_objects=3218`; `missing_ledger=0 unexpected_ledger=0 missing_blob=0 failed_or_not_done=0`; `status={"done": 3218}`; `tier={"r3500": 1080, "r5600g": 709, "tower": 1429}`. Report sha256 `d7d07a3e25d9555da5d7970e2937814d9c15a76fb56759de2185bd34b293f2d4`; log sha256 `861cd7fbd29bdc3307a96a1afcbd71427e162cee90ba9b34381046dbd28c08b9`.
- UTC 04:10:46; command: `touch /var/tmp/cvvdp-safesyn/STOP_VERIFY`; exit 0. The verifier had already reported all 784,344 rows with zero error and pixel-hash discrepancies. Four extra ledger rows are duplicate `done` records with identical output blobs; all 3,222 raw ledger rows have `status=done` and `attempts=1` (independent Parquet read).

## Pending pipeline

`/home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- bash /var/tmp/cvvdp-safesyn/store_inventory.sh` is queued on the shared heavy lock. It will repeat the store check, harvest all score blobs, check fresh SSIM2 by keyed join against the audit by family and worker tier, and compute descriptive rank agreement through `zen_stats.panel_batch_indexed`. Script sha256 at last inspection: `c58a1c90ac771c41dd7b6f0e791d675477ba36371dc9a93452da2cf7f0d8e52c`. The script checks `CODEX_QUOTA_STOP.md` at safe points.
