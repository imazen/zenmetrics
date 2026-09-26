# Oracle archive pointer (2026-09-25)

The reference software for MDSI (used only as a black-box oracle, never part of this repository), its Octave
driver scripts and the unredacted originals of records edited at landing live outside the repository:

- local: `/var/tmp/gmsd-chroma/ref/`
- tower mirror: `output/zensim/gmsd-chroma-2026-09-24/oracle-archive/` (under the coefficient share)

| File (relative to the archive) | SHA-256 |
|---|---|
| `mdsi.zip` | `8b78597d180d5e250a928fbee7b5ffb8a714ce354e3e8640eec7703692db472b` |
| `mdsi/MDSI.m` | `4dff198b28e57b39b20877c7bcb75af2c6278c95f9f321362a2046ee3e37351b` |
| `mdsi/license.txt` | `0b717c37c30188b069cdf9cf19435a076f5d3b5f8f9bcb8f90e946f0e6949a2f` |
| `lane_scripts/prepare_octave.py` | `d87cee3040474d5d0f69a8d11f7b70db1df03de1403c82271c383c6a5aff7c85` |
| `lane_scripts/octave_container.sh` | `75127f10b18f047b175f56e12e8d0a4c79985997a9750ab0efc1b71a3cab7d2d` |
| `unredacted_2026-09-25/gmsd-chroma_prereg_2026-09-23.md` (original of the landed copy) | `d0e65fce1c437bab26b2d24c5261d61e0138a0c48c0d22b26c4b875411441908` |
| `unredacted_2026-09-25/gmsd-chroma_WORKLOG.md` (original of the landed copy) | `3c94450c5ced74c8d4371d5a960b94f24d1f6d4f17f942e8346c5b9534b4d012` |
| `unredacted_2026-09-25/gmsd-chroma_WORKLOG_part1.md` (original of the landed copy) | `0538bfeb6c07df4be7a664b5eaca87b3ff66892667f10d76589e58665e3cb840` |
| `unredacted_2026-09-25/gmsd-chroma_reference_2026-09-24.md` (original of the landed copy) | `76d01301356f7f250a34ccab9de41438783049f7721ac8b2a8840fcf8aa60b47` |

The landed copies of the preregistration, the two worklogs and the reference inventory drop wording about the
reference software's internals; the gates, statistics and decision rules are unchanged. The Octave driver, its shim
and the container script were moved out of the repository into this archive; `validate_oracles.sh` only consumes
their outputs (the score and map dumps in the gate bundle, see `gmsd-chroma_mdsi_gate_bundle.pointer.md`).
