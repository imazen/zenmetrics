# Frozen SafeSyn metric

This isolated program rebuilds the metric called by the preserved
`extract-native-admission` executable (SHA256
`7c7ffbbfa033e8ca1a8f103d472b61ccde061c2394b03d519af2852ee8eeda87`).
It preserves the archived PNG-to-RGB8 projection and calls fast-ssim2 0.8.2.
The published package records VCS revision
`b7c2b4b33f3a26b753caabd5264a7fef9c4f4545` and crate checksum
`8dcca2000fbfa5923eb54b5bedddadbe5f7c83f7ef65285ebe0ed10daa78d0f0`.
The independent lock preserves the archived math dependencies, including
archmage/magetypes 0.9.28 and yuvxyb-math 0.1.1. It does not inherit the
current workspace's metric patches.

Feed JSONL `{ "reference": "ref.png", "distorted": "candidate.png" }` on stdin.
One JSON score record is emitted per pair, with exact f64 bits and both pixel
hashes. The program refuses missing inputs, geometry differences and nonfinite
scores. There is no current-metric fallback.

Before Stage B declaration, `zensimx/scripts/byconstruction/chroma_b0_replay.py`
replayed four stored SafeSyn controls and 128 A2 cells (four families, eight
views, four rungs), all score bits and checked pixel hashes exact: 132/132.
Run receipt: `/var/tmp/byc-chroma-20261028/b/teacher-replay-final/PARITY.json`.
This identifies and reproduces the metric component; it does not claim to
reconstruct every feature of the historical extractor.
