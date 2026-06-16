# Commercial Licensing

This repository is dual-licensed:

- **Open-source:** [AGPL-3.0-only](LICENSE-AGPL3). If your project is itself
  AGPL-compatible, this is what you want.

- **Commercial:** Imazen offers a paid commercial license for use in
  closed-source products. Two separate tracks apply, depending on the
  crate:

  ## `butteraugli-gpu`, `ssim2-gpu`, `zensim-gpu`, `zenmetrics-cli`, `zenmetrics-corpus`

  Standard Imazen commercial license — covers the GPU implementations
  Imazen authored from scratch for this repository, the test corpus,
  and the unified CLI. Contact `support@imazen.io` for terms.

  ## `dssim-gpu` (separate track)

  `dssim-gpu` is a CubeCL port of the algorithm shipped by the
  upstream [`dssim`](https://github.com/kornelski/dssim) /
  [`dssim-core`](https://crates.io/crates/dssim-core) crates by
  Kornel Lesiński. Imazen authored the GPU port itself, but the
  underlying algorithm and the parity-validation reference are
  Pornel's. As a result, commercial use of `dssim-gpu` requires
  **both**:
  - An Imazen commercial license for the GPU port code, AND
  - An upstream commercial license from Pornel for the underlying
    DSSIM algorithm.

  Both can be arranged via `support@imazen.io`; we coordinate the
  upstream license alongside ours. This is the same pattern Imazen
  uses for any port whose root algorithm carries a third-party
  copyright.

If you're not sure which track applies, ask `support@imazen.io`.
