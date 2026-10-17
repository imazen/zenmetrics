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
`ZENMETRICS_NN_ACCEPT_WEIGHTS=<column>`. Two exceptions: `cw-ssim-pyiqa` has no weights, and
`dmm-pyiqa` uses only torchvision's VGG16 (permissive), so neither asks for consent.

Models follow `crates/zenmetrics-nn/ADDING_MODELS.md` (card, export, goldens,
parity gates), except that the code source is pyiqa.

| Model | Column | Status |
|---|---|---|
| TOPIQ-FR (KADID) | `TOPIQ` | `topiq-fr`: parity within 1e-3 JND on rten, ort-cpu, ort-cuda; matches the column (median 3.5e-5, Spearman 1.0) |
| TOPIQ-FR (PIPAL) | `TOPIQ-PIPAL` | `topiq-fr-pipal`: parity within 1e-3 JND on rten, ort-cpu, ort-cuda; matches the column (median 2.4e-5, Spearman 1.0) |
| WaDIQaM-FR (KADID checkpoint), `wadiqam-pyiqa` | `WaDIQaM` | parity pass on rten, ort-cpu, ort-cuda: worst 1.1e-5 JND (ladder), 2.7e-6 JND (AIC2026 61) |
| AHIQ (pyiqa convention), `ahiq-pyiqa` | `AHIQ` | parity pass on rten, ort-cpu (3 ladder pairs), ort-cuda: worst 1.2e-5 JND (AIC2026 61), 9.3e-6 JND (ladder) |
| CW-SSIM (pyiqa convention), `cw-ssim-pyiqa` | `CW-SSIM` | parity pass: worst 5.4e-13 JND |
| DMM (pyiqa convention), `dmm-pyiqa` | `DMM` | parity **fails** 1e-3 JND (worst 0.071 JND, ladder 256x256): SVD-solver residual, see its card; vs the published column the same as pyiqa CPU |
| PieAPP (pyiqa convention: stride 27), `pieapp-pyiqa` | `PieAPP` | parity pass on rten, ort-cpu, ort-cuda: worst 2.3e-6 JND |
