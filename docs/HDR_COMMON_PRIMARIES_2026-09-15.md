# Explicit native HDR metric ingress — September 15, 2026

`score-pairs --hdr --hdr-common-primaries` admits full-range RGB PQ PNG/JXL
with structured cICP primaries 1 (BT.709), 9 (BT.2020) or 12 (Display P3).
It decodes native samples to absolute nits and uses the existing
zenpixels-convert row converter to common linear BT.709, without clipping
out-of-gamut values before the metric display model. JXL pixel-format
preferences are storage choices; actual cICP defines color. ICC or ambiguous
metadata requires a CMS and is refused by this contract. EXR/AVIF/HLG are not
silently admitted under this specifically PQ contract.

The Parquet footer carries `common-bt709-pq-native-v3`. The flag requires
`--hdr`, refuses `--feature-output` before opening payloads, and refuses any
metric without a native HDR result. Historical `--hdr` behavior and legacy
feature sidecars retain their original contracts. The CPU CVVDP metric used
the legacy 8-bit shell when the generic faithful route excluded CVVDP;
this explicit version calls the existing native HDR scorer instead.

CVVDP here models a BT.709 HDR display with reference-measured peak and its
own gamut clipping. This is not native P3-display qualification. SSIM2 uses
the existing integrated PU-linear HDR route, not an SDR 8-bit proxy.

Validation: independent f64 BT.2020/P3 matrices, absolute-light scale and
negative out-of-gamut preservation; unknown-primary/EXR refusal; CLI dependency
and feature-sidecar refusal before nonexistent input is opened. Full corrected
Zensim TRAIN audit recomputed CVVDP and PU-SSIM2 on all 7,425 declared-PQ pairs,
zero failed rows. The pinned measurement tool hash is
`91f886ca9329e4c6f58ff07077c4faeeae0a4a90c8084bab776b11714188eb5f`.
Subsequent comments/tests do not change numeric behavior. Implementation:
`1f5aa1c5`. The Cargo lock update records synchronized sibling versions,
including path fast-ssim2 0.9.0 and registry 0.8.2 for remaining pinned users.

Source admission, commands, exact input/tool hashes and results are under
`/var/tmp/zensim-validation-2026-09-15/hdr-corrected/`; the sibling Zensim
`benchmarks/recovery_completion_2026-09-15.md` reports the candidate results.
A valid native input path is separate from a model's HDR quality: both frozen
SDR recovery candidates failed to match BHdr on native human UPIQ EVAL.
