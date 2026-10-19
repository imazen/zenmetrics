# Upstream configuration evidence (hdrvdpcx, 2026-10-19)

The exact AIC2026 invocation is still unconfirmed. These are source records,
not inferred calls or an exact-reproduction claim.

## Official releases

Downloaded from the [official SourceForge release directory](https://sourceforge.net/projects/hdrvdp/files/hdrvdp/).
The directory lists 3.0.7 (2023), 3.0.6 (2020), 2.2.2 (2020), and 2.2.1 (2015),
with no later release listed at retrieval. The extra 2.2.1 archive permits an
older-version control. Local ZIPs and extracted trees are on r5900xt under
`~/tmp/hdrvdpcx/upstream/`; none is bundled in a shipped crate.

| release | official URL | bytes | sha256 |
|---|---|---:|---|
| 2.2.1 | [ZIP](https://downloads.sourceforge.net/project/hdrvdp/hdrvdp/2.2.1/hdrvdp-2.2.1.zip) | 641782 | `d6d7f2af0a5e6ff96c9e2726b43dc31fb425853d1a0e6a24b69a389816759767` |
| 2.2.2 | [ZIP](https://downloads.sourceforge.net/project/hdrvdp/hdrvdp/2.2.2/hdrvdp-2.2.2.zip) | 6297944 | `b651bc474a47050b8c107484c0aff8571f5b237264eaa6d898f85a2ae7766820` |
| 3.0.6 | [ZIP](https://downloads.sourceforge.net/project/hdrvdp/hdrvdp/3.0.6/hdrvdp-3.0.6.zip) | 11208179 | `6fc6bb0eaa1d5b58cb9c055e82b05d0aa38d2dcf8f66d5ec1e2cb105ec708382` |
| 3.0.7 | [ZIP](https://downloads.sourceforge.net/project/hdrvdp/hdrvdp/3.0.7/hdrvdp-3.0.7.zip) | 10875321 | `baa5c25cca2e38176345a2a5423793459fbf571282396640c5e4ac85bf8381b3` |

Fresh 2.2.2 and 3.0.7 ZIP hashes equal the predecessor's retained copies.
The [official project page](https://hdrvdp.sourceforge.net/wiki/) announces
Octave support with 3.0.7. Its ChangeLog says 3.0.7 updates matlabPyrTools,
removes MEX binaries, and adds GPU support; 3.0.6 predates that update.

## Written AIC conditions: a new peak-luminance lead

[JPEG AIC-4 CTC v2.0, WG1 N101246](http://ds.jpeg.org/documents/jpegaic/wg1n101246-108-ICQ-Common_Test_Conditions_on_Objective_Quality_Assessment_v2_0.pdf),
section 4, page 7: HDR-VDP-2 is version **2.2.2**, with SDR **37.84 ppd**
and **203 cd/m² peak**. HDR-VDP-3 is configured the same way. The HDR
1000 cd/m²/56.55 ppd and `rgb_display=led-lcd-wcg` settings are a separate
contract. The CTC names `Q`, not `Q_JOD`. It does not give the SDR EOTF,
colour encoding, rounding, black level, task, age, or exact invocation.
Do not assume these CTC conditions were used for the later AIC2026 columns.

The CTC coefficient values appear in the site's separately named
`HDR-VDP-2 Q` and `HDR-VDP-3 Q` entries (`isCTC=true`). Those entries store
the threshold under `c` (85/10) and exponent under `b` (3.0698/0.6412),
a different parameter representation from the `HDR_VDP_2`/`HDR_VDP_3`
column mappings; do not pass CTC entries to the column-mapping formula. The retained closest routes used 200 cd/m²,
so 203 cd/m² needs a direct oracle control rather than a fitted correction.

HTTPS retrieval failed certificate validation; plain HTTP delivered the PDF.
The local PDF is `~/tmp/hdrvdpcx/docs/ctc.pdf` (r5900xt), 311239 bytes,
sha256 `3816e4cbdc78f45fb9e856f583bf2d24152432bbb6948f90a49b60fa85d21c6f`.
This records transport provenance, not an authenticated PDF signature.

The [AIC2026 paper](https://arxiv.org/abs/2607.22783) explicitly names
`cvvdp -d standard_fhd` for CVVDP (37.84 ppd, 200 cd/m², black 0.2,
reflection 0.3979); its HDR-VDP table names versions 2 and 3 and cites their
papers, without a runnable HDR-VDP invocation.

The retained [jpeg-ai-qaf](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-qaf)
`0628a6b` source (`metrics.py`, class `HDR_VDP`) accepts PQ/HLG, uses a
10000-nit clipping ceiling and 100-lux reflection, and returns v3 `Q_JOD`.
That path does not specify this dataset's 200/203-nit SDR conversion and
cannot by itself establish the AIC2026 call.

## Retained Octave 2.2.2 modifications

The predecessor's five-file diff remains at
`~/tmp/hdrvdpres/hdrvdp222_octave_tree.diff` (r5900xt), sha256
`16b95af1108c78cf77adc3d1778c3378aed52b803320f16e650da700ccaade0b`.
`hdrvdp.m` adds diagnostics only. The two `reconSpyr` edits force the
return-value fallback instead of the MEX in-place path. Reconstruction is
after `Q` accumulation and before returning `res.Q=100-Q`; it does not
feed the raw Q accumulator. The other two edits remove MATLAB's
`ifft2(...,'symmetric')` argument while retaining `real(ifft2(...))`.
Those FFT semantics require their own control before calling the modified
tree an arithmetic-equivalent canonical oracle.
