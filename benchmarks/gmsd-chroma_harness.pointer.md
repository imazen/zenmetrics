# Frozen timing harness and raw evidence

Bulk root: `/var/tmp/gmsd-chroma/`, lane `gmsd-chroma`, 2026-09-24.
Nothing has been pushed or admitted as qualified evidence.

The imazen zenbench source snapshot is `zenbench-snapshot/`. Its inventory,
`zenbench_snapshot.json`, has SHA-256
`394a18b62ba4a67b6782548d781ed9bf51c57663bb2e368d336291f3c1064da6`.
Each inventory entry pins the copied file's SHA-256. It retains raw rounds,
which published 0.1.9 does not. Licences are MIT OR Apache-2.0, included in
the snapshot. No source files were modified while copying.

`scripts/gmsd-chroma/build_checks.sh` verifies the snapshot and applies
`cargo --config 'patch.crates-io.zenbench.path="/var/tmp/gmsd-chroma/zenbench-snapshot"'`.
The package's normal dev-dependency remains version 0.1.9. Reproducing this
lane's timing requires the verified snapshot; substituting registry source
loses the raw-round schema. The source may be recovered from the original
imazen zenbench checkout only if every inventory hash matches.

After build_checks_v2 exposed clippy's child-Cargo lockfile resolution,
`ensure_cargo_patch.py` also writes this patch into the lane's private
`cache/cargo/config.toml`. It refuses an unexpected existing config. No
user Cargo configuration or credentials are changed. The example build
manifest records the actual resolved Cargo.lock hash and its pre-build
hash separately; all code/manifest source hashes must stay unchanged.

`commands.jsonl` records exact UTC intervals, cwd, argv, return codes and
log SHA-256. Raw oracle inputs/maps, scoring sidecars and timing outputs
stay under this root; measured outputs are listed in the eventual result
record. Reference acquisition, 116-case input staging and author MDSI
execution have succeeded; Rust image parity and timing remain pending.
