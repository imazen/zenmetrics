# Native HDR chroma tool

An isolated, unpublished tool for the owner-approved HDR teacher leg. It derives
its render kernel from `imazen/evif` commit
`5ee33f89caf0789708ed1ed40d97b03e6cd3a39d`, `evif-probe/examples/chroma_hdr.rs`.
Imazen owns that code; its AGPL/commercial license is retained here. No EVIF
container or EVIF checkout is changed by this tool.

`prepare SOURCE NEW_DIR` decodes the native RGB16 PQ source once, uses pinned
zenresize Mitchell linear-light resizing for ten nonupscaled long edges
32..960, and takes central native 256/512 crops where supported. Every output
is RGB16 PQ with explicit primaries, no ambiguous ICC/gamma metadata, and
sample/signaling roundtrip verification. All views stay below one million pixels.

`batch REFERENCE NEW_DIR` decodes the prepared view once and emits four
fixed-luma-quality chroma ladders: q95/85/75/60, shared U/V deltas
0/8/16/24/32/40/48/56/63. `a3` adds the frozen twenty-point luma control;
`decode AVIF NEW_PNG` emits a native PQ16 rendition. Each render also emits
its exact imazen-decoded PQ16 rendition in the same process. The codec pins
match HDR-A3: zenavif a7638f6e, zenav1-svt 0f63ca13, zenpixels 1d3ddd21.
4:4:4 is explicitly refused; no implementation is attempted.

The source build reproduces all 208 HDR-A3 AVIF hashes and all 208 natural
SSIMULACRA2 score bits. Full-grid generation/admission is a separate gate.
`just chroma-hdr-render-check` runs the isolated native contracts and lint;
execute builds/checks through run-heavy on an eligible host.
