# MDSI author-score gate bundle pointer (2026-09-25)

Everything the caller-controlled 116-pair gate (`GMSD_MDSI_GATE=require`) needs, kept outside the repository (397 MB):

- local: `/var/tmp/gmsd-chroma/mdsi_gate_bundle/`
- tower mirror: `output/zensim/gmsd-chroma-2026-09-24/mdsi-gate/` (under the coefficient share; copy verified with `sha256sum -c` against the lists below, 0 mismatches)

Layout: `target.tsv` (116 pairs, paths relative to the table), `inputs/` (224 unique raw RGB8 files; 232 table entries, identity pairs share files), `octave/` (697 files: `scores.tsv` plus per-pair CS/GCS/H/M maps as little-endian f64, produced by running the authors' reference software), `SHA256SUMS.inputs`, `SHA256SUMS.octave`, `README.txt`.

| Item | SHA-256 |
|---|---|
| `target.tsv` | `4442518041a387e06123925133da9fda902b321df00dfcc9cc7c4e60011bcbe1` |
| `SHA256SUMS.inputs` (per-file sums of `inputs/`) | `bc7880d438ba92ba0927fca91fff12a15a9322f0475a5bcdb7e8f6ff02248446` |
| `SHA256SUMS.octave` (per-file sums of `octave/`) | `4a3f484922bfa41df17ea2f9e37b083981be1aad7ab05244476cc13c331039f2` |
| `octave/scores.tsv` | `cb4bb01fc625e960f9dac169a3027f1bf31ab552e7c58d44fb8079295f1cbe83` |

Run: `GMSD_MDSI_GATE=require GMSD_MDSI_TARGETS=<bundle>/target.tsv just test-gmsd-mdsi-gate`. Expected: 116 pairs, max abs 4.298904288102534e-11, max rel 4.836930021352448e-10, wrong constant fails 109/116.
