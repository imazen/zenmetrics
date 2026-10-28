# EVIF fleet runner

This isolated build exports the existing zenfleet worker binary without the
codec workspace's path dependencies. It uses the worker/core/ledger owners
unchanged except for the private EVIF output-envelope persistence path.
The root Cargo.lock is never regenerated here.

`just chroma-fleet-build` builds it. EVIF jobs use
`scripts/jobsys/evif_exec.py`, a pinned external executor with Encode and
`ssim2-safesyn-b7c2b4b3` Metric arms. Set `ZEN_EVIF_BIN_DIR`, a disk-backed
`TMPDIR`, and PYTHONPATH for pyarrow. Both native JPEG and EVIF are persisted
by the worker's configured BlobStore before DONE. The ledger output is the
EVIF container; ordinary metric JSON harvesters do not apply to these jobs.
Every job must require `evif-cas-v1` and the exact teacher capability. Bind
source bytes and canonical parameter JSON as job inputs, plus the tool pins.
CVVDP and butteraugli values in the container are diagnostics only.

Verified image: `ghcr.io/imazen/zenfleet-worker@sha256:32f2c55791a5cf169020a2f67b062c8cc77481f393aa709f3e25ca8b8b20121a`.
Its real one-cell proof uses both job kinds, verifies both container hashes
and embedded/native JPEG CAS equality, and replays the A2 label double exactly.
Diagnostics use zenmetrics 0a61830db89a768ab9462f166d1412e948e4efa9,
executable SHA256 37c3bc112d8bdc86b949005f8fcafd419510b88b7dd7c7a8938ba59251c4e50f.
CVVDP explicitly uses `standard_4k`; butteraugli records max and 3-norm.
The E26 HDR-only build has no CPU butteraugli arm and is not used here.

The view path is `zenfleet-ctl declare-evif --spec SPEC --out MANIFEST
--grid-out /absolute/rungs.jsonl`. Its strict specification carries prepared
source byte/pixel hashes, source identity, four labelled parameter arrays,
implementation hashes, memory hint and `diagnostics: none|all`. One Metric job
renders the complete union once, reuses EVIF's decoded RGB8 reference in one
frozen teacher process, then imports all scores in one EVIF process. Every
native JPEG and scored EVIF reaches CAS before the view manifest becomes DONE.
The staged-output envelope is private to the worker/executor transport.

`ZEN_ARCHIVE_BLOBS_DIR=/absolute/archive` selects a sharded immutable filesystem
CAS (`aa/bb/<sha256>`) for output bytes; claims/control/ledgers stay in the
normal S3 job system. The archive must be mounted identically on all workers.
This opt-in is scoped to BlobStore writes/existence checks; S3-only audit tools
cannot audit that archive. Independently hash-check the view manifests and all
referenced objects when admitting data. FUSE/NFS can retain transient hidden
files after a successful hard-link publication; cleanup failures are logged
separately from the verified result and do not mark the upload failed.

The canonical LAN launcher accepts `ZEN_DATA_HOST_DIR` (read-only prepared
inputs), `ZEN_ARCHIVE_BLOBS_DIR` (writable CAS), and a hash-verified private
`ZEN_RUN_HEAVY_HOST_PATH`/`ZEN_RUN_HEAVY_SHA256` binding. The helper stays outside
the image. Set CPU affinity/shares, Docker memory/swap caps and disk scratch
explicitly; set `ZEN_CORE_OVERSUBSCRIBE=1` for one thread per admitted job.

`declare-evif-diagnostics` declares a later ScoreFile pass over selected EVIF
hashes and bound source identities. Its executor reads already persisted PNG
renditions, calls the pinned diagnostic ports, and emits joined thin sidecars
and raw port tables; it has no render/encode command. Selection is a scientific
input, independent of scheduler claims and retries. It preserves target-only
containers, including their exact target doubles.
