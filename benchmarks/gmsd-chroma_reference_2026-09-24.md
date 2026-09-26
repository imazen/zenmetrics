# gmsd-chroma reference inventory

Status: author MDSI software ran on 116 cases. The first Rust comparison
failed six final-score gates; the pooling correction remains untested at
quota stop. MS variants passed the independent NumPy equation comparison.
No MS-GMSD author software was obtained; no author-software parity is claimed.

MDSI author release:
<https://www.mathworks.com/matlabcentral/fileexchange/59809-mdsi-ref-dist-combmethod>

Versioned download:
<https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/59809/versions/1/download/zip>

| External file under `/var/tmp/gmsd-chroma/ref/` | SHA-256 |
|---|---|
| `mdsi.zip` | `8b78597d180d5e250a928fbee7b5ffb8a714ce354e3e8640eec7703692db472b` |
| `mdsi/MDSI.m` | `4dff198b28e57b39b20877c7bcb75af2c6278c95f9f321362a2046ee3e37351b` |
| `mdsi/license.txt` | `0b717c37c30188b069cdf9cf19435a076f5d3b5f8f9bcb8f90e946f0e6949a2f` |

The author files were used only as a black-box oracle and remain external,
unmodified and unpublished; no part of them is in this repository. The MDSI
in `crates/gmsd` was replaced (2026-09-24) by a clean-room implementation
written from the paper alone, validated against the scores this software
produced (see `crates/gmsd/docs/MDSI_CHOICES.md`).

The pinned amd64 image is
`gnuoctave/octave@sha256:185db7993e000d4f3f6e7bbbf7fb3f999f52e799ea52231ad8a15353381e0dcb`.
The smoke run reports **Octave 11.3.0** and no installed packages. One Octave function absent from the base image was
shimmed; the driver, the shim and the container script are kept in the external
archive (see `gmsd-chroma_oracle_archive.pointer.md`).

The full author run completed at 2026-09-24 09:45:21.963Z, rc 0. There are
116 unique score rows and 696 CS/GCS/H/M map files. Its log SHA-256 is
`715ad6ee3225b2aeeb74784f1826b3bd5839be7ccbc5ab6fad99085090cd9c1a`;
`octave/scores.tsv` is
`cb4bb01fc625e960f9dac169a3027f1bf31ab552e7c58d44fb8079295f1cbe83`.
Every instrumented score matched the untouched author function exactly.
This establishes the oracle outputs, not agreement with the Rust port.

Post-run integrity check, 2026-09-24 10:23:06Z: the archive, author source
and licence still match the three hashes above. The instrumented wrapper's comparison is recorded in the external archive.
`author_artifact_recheck.log` has SHA-256
`5db74fb0cc89fe42759deb9084abb4767b688e73c6e216e59206602d39fdb454`.
Its exact output is:

```json
{"author_files_hash_match": 3, "map_files": 696, "unique_scores": 116, "wrapper_body_unchanged": true}
```

The exact command is retained in `commands.jsonl`. Re-run that read-only
check without reconstructing its quoting:

```bash
python3 -c 'import json,subprocess; r=next(r for r in map(json.loads,open("/var/tmp/gmsd-chroma/commands.jsonl")) if r["output"].endswith("/author_artifact_recheck.log")); subprocess.run(r["argv"],cwd=r["cwd"],check=True)'
```

For MS-GMSD, the primary paper and author publication pages were inspected:
<https://www.cse.ust.hk/~psander/docs/gradsim.pdf> and
<https://www.cse.ust.hk/~psander/cv/pvs-cv.html>.
MathWorks and GitHub discovery found no author release. This is a search
outcome, not proof that no release exists. A paper implementation is now
prepared under the preregistered YIQ, border and downsampler conventions.
Its independent NumPy oracle is `scripts/gmsd-chroma/ms_numpy.py` and its
gate is `compare_ms.py`; both ran on the 116-case population. MS-GMSD and
MS-GMSDc passed, with maximum absolute errors 5.013697790268168e-14 and
3.202646481348381e-14. Exact recomputation and the failed MDSI result are
recorded in the worklog and `validation_v1_summary.log`. The source uses the same
f64x8 archmage tier and 64-row band pattern as MDSI. Its libm exponential
is separate from the original GMSD arithmetic. It claims no author-software
parity; this establishes agreement only with the independent transcription.

Acquisition, hashes and image resolution are recorded in
`/var/tmp/gmsd-chroma/commands.jsonl`, with full logs under `logs/`.
The committed preregistration states the numeric gates and input roles.
