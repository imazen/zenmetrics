# Primary SSIMULACRA2 scalar grid, 2026-10-19

Missing: an assigned cross-implementation acceptance bound, runtime CMS-branch
closure, and qualified cold/warm timing ratios. No pass/fail gate is inferred
from a selected golden test or these observations.

All [24 scalar cases](primary-ssim2.json) returned finite scores: three raw
content pairs at 64², 255×257, 1024² and 3840×2160, with 1/28 threads. The
largest absolute difference is 2.474855875789217e-9 on the 4K stained-glass
pair at both budgets (Rust 48.248057126249854; primary 48.24805712872471).
Cold and warm primary values match within each process. No image decoder or
resampler runs on these retained RGB8 buffers.

The primary is unchanged Cloudinary v2.1 (`fb5d0dbe`) with its vendored libjxl,
owned Highway 1.2.0 and Little-CMS 2.13. Earlier ingress controls match the
official file CLI's eight-decimal output. This study uses the separate raw-RGB
adapter and preserves its explicit source/build identity. Requested workers
control canonical ingress; its score body runs on the caller thread.

The primary exposes a scalar and pooled features, with no native pixel-map
API, so no comparable primary pixel map is synthesized. Rust's optional maps
are a distinct API. This scalar study does not establish runtime CMS-branch
closure or select a numerical gate. [Metadata](primary-ssim2.meta) retains
commands, hashes and both the whole-panel resource receipt and per-process RSS.
