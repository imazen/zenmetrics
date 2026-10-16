# IQA compute-only competition

Work in progress. No completed matrix or performance claim yet.

The driver extends `cpu-profile`, reusing zenbench and the CLI metric inventory.
The initial VMAF CLI timings include embedded-model parsing and are diagnostic,
not qualified final compute-only observations. The benchmark snapshot adapter
caches only immutable model parameters after warmup; owned-model clones stay
inside the call. No shipped metric source is changed. Its source hashes and
score/map parity gate must accompany the final rows.

Input preparation decodes the selected real AIC pair once, then uses zenresize
Lanczos for targets within both source dimensions and mirrored source pixels
for larger targets. Raw RGB files are loaded before timing. Their hashes belong
in each cell's input manifest. Learned model initialization runs before timing. VMAF uses the separately
qualified snapshot adapter described above to exclude embedded SVM parsing.

Each Rust group uses warmup and at least ten interleaved zenbench rounds.
Percentiles describe per-round batch means; raw batches and iteration counts
must be retained so these are not confused with individual-call latencies.
Scores and native-map calls are distinct arms. Unsupported routes and refused
sizes remain explicit rows. Every competitor must use the same pair and record
its score difference from the corresponding Rust implementation.

Raw data is external: `~/tmp/iqa-compete/<host>.jsonl`. Committed summaries
will be split below 30 KB. All heavy steps use the shared host resource wrapper;
individual lock holds have a hard bound below twenty minutes. No codec or file
read belongs inside a measured region. Third-party dependencies remain confined
to this benchmark harness.

## Initial qualification

The pinned release build and driver clippy pass. The initial CID22/2K
qualification covers PSNR, SSIMULACRA2, GMSD and FSIM at budgets 1 and 20;
CVVDP also has a CID22 one-worker check. All 30 timed score/map arms match
scores exactly; four PSNR map requests explicitly report unsupported.
Compact observations: [CID22](native-smoke-cid22.json), [2K](native-smoke-2k.json).
The full 112-name CPU inventory pass is still in progress; these smoke
checks do not establish full coverage.

The hybrid all-core panel is descriptive. Zenbench's uniform-core class
certification cannot describe that required configuration; the harness
records unrestricted placement plus explicit OS affinity (one performance
core for one-thread cells; all available CPUs for all-core cells). No
baseline certification or promotion is claimed. The ordinary load/resource
admission gate stays enabled.

The newer SSIMULACRA2 native-map adapter uses the same pinned sibling as
the CLI through a benchmark-only alias. The legacy profiling driver's
registry dependency remains unchanged. CVVDP allocates its scorer before
timing; both paths consume the same named standard_fhd display.

The benchmark-only model cache passes 24 bitwise comparisons against the
original parser (six VMAF names, CID22/2K, one/all 20 workers). Each probe
checks the cold and warm scalar outputs. VMAF-v1 also compares every one
of its 43 native maps by geometry and raw-byte SHA-256. The
[receipt](model-adapter-parity.json) binds both executable hashes.
The cached build uses `--cfg iqa_compete_ingress --cfg iqa_compete_cached`;
the reference uses only the ingress flag. These flags do not select SIMD
or change metric arithmetic. The initial VMAF timing rows remain diagnostic.

[DSSIM/Zensim qualification](extra-map-smoke.json) adds 16 score/map arms
at CID22/2K and one/all workers, with zero score difference.
C/C++ competitor adapters are buildable scaffolding; their timing and
same-score qualification are not yet complete.
