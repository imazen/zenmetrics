# zenmetrics-pyiqa-ports

Ports of image-quality metrics whose only reference implementation is
[pyiqa](https://github.com/chaofengc/IQA-PyTorch), following pyiqa 0.1.16's
default conventions. These are the conventions behind several JPEG AIC2026
published metric columns (see `docs/AIC2026_REPRODUCTION_STATUS.md`).

**Licence: non-commercial.** This crate is derived from pyiqa and carries its
terms: PolyForm Noncommercial 1.0.0 (`LICENSE-pyiqa`) and the S-Lab License 1.0
(`LICENSE-S-Lab`). The rest of the workspace does not. The crate is never
published, and no code from it may be copied into another crate. The CLI
includes it only when built with the non-default `pyiqa-ports` feature.

**Weights: non-commercial.** Every model's checkpoint is distributed by pyiqa
under non-commercial terms. Loading one requires a `WeightsConsent`; on the CLI,
`ZENMETRICS_NN_ACCEPT_WEIGHTS=<column>`.

Models follow `crates/zenmetrics-nn/ADDING_MODELS.md` (card, export, goldens,
parity gates), except that the code source is pyiqa.

| Model | Column | Status |
|---|---|---|
| TOPIQ-FR (KADID) | `TOPIQ` | `topiq-fr`: parity within 1e-3 JND on rten, ort-cpu, ort-cuda; matches the column (median 3.5e-5, Spearman 1.0) |
| TOPIQ-FR (PIPAL) | `TOPIQ-PIPAL` | `topiq-fr-pipal`: parity within 1e-3 JND on rten, ort-cpu, ort-cuda; matches the column (median 2.4e-5, Spearman 1.0) |
| WaDIQaM-FR (KADID checkpoint) | `WaDIQaM` | in progress |
| AHIQ (pyiqa convention) | `AHIQ` | in progress |
| CW-SSIM (pyiqa convention) | `CW-SSIM` | in progress |
| DMM (pyiqa convention) | `DMM` | in progress |
