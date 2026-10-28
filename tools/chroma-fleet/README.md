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
