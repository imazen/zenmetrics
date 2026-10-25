# Margarine development

Read [DATA_PROVENANCE.md](DATA_PROVENANCE.md) and [README.md](README.md) first.
The original Butteraugli metric remains the frozen teacher; this crate owns
its approximation and research instruments. No API release has been approved.

## Known bugs and resolved experiment defects

The historical full-Malta final-block implementation (`a85ba912`) produced
incorrect LF responses in optimized x86 builds. The uniform overlapping-block
loop (`2d9d736a`) fixed it and reproduced all 344 LIVE maps/norms. Keep the exact
border/tail and strided-map tests; do not relax their expectations.

Historical zensim feature probes needed a 64-row minimum halo to handle the
769-row bottom-tail case. The lab's optional research dependency also includes
the verified x86 blur-tail correction. These probes do not define the selected
Butteraugli-lineage candidate. Full detail and original links are preserved in
[the original project notes](https://github.com/imazen/butteraugli/blob/13c49cbcab6e2b2bb65f60e3cb28344463bacbf7/CLAUDE.md).

### Existing vendor-check failure (2026-10-25)

`verify_sources.py` fails on `blur.rs` and `malta.rs`: `45e46e998` renamed their
v4 dispatch to v4x, while SOURCE.json retains the original upstream hashes.
The mismatch is present before the steering work. Keep the established check
unchanged and failing pending owner approval. Exact-byte restoration and a
strict declared-patch proposal are described in
[STEERING_REFERENCE.md](STEERING_REFERENCE.md). No metric arithmetic was changed
by that dispatch rename; its hardware-tier selection changed.
