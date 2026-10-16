# IQA compute-only competition

Work in progress. No completed matrix or performance claim yet.

The driver extends `cpu-profile`, reusing zenbench and the CLI metric inventory.
Input preparation decodes the selected real AIC pair once, then uses zenresize
Lanczos for targets within both source dimensions and mirrored source pixels
for larger targets. Raw RGB files are loaded before timing. Their hashes belong
in each cell's input manifest. Model initialization runs before timing.

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
