# Third-party data — ColorVideoVDP

The JSON files in this directory are vendored verbatim from
[ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP) (the
Graphics and Displays group's PyTorch reference implementation of
the cvvdp perceptual metric).

| File | Upstream path | Purpose |
| --- | --- | --- |
| `display_models.json` | `pycvvdp/vvdp_data/display_models.json` | Named device presets (resolution, viewing distance, peak luminance, ambient, colorspace selector) loaded by `presets::DisplayModel::by_name` and `presets::DisplayGeometry::by_name`. |
| `color_spaces.json` | `pycvvdp/vvdp_data/color_spaces.json` | EOTF + RGB→XYZ primaries lookup keyed by the `colorspace` field of each display preset. |
| `cvvdp_parameters.json` | `pycvvdp/vvdp_data/cvvdp_parameters.json` | Model parameters (pooling exponents, masking, CSF correction, JOD mapping). |

Both files are MIT-licensed; the upstream license text is also
vendored as `UPSTREAM_LICENSE_MIT.txt` in this directory.

Fetched 2026-05-25 from `main` branch
(https://github.com/gfxdisp/ColorVideoVDP/tree/main/pycvvdp/vvdp_data).

Checked 2026-09-23 against the **v0.5.7** release: `display_models.json` and
`color_spaces.json` are content-identical to it (the `65inch_hdr_pq_*` /
`lg_oled_2026_hdr_pq` presets taken from `main` shipped in v0.5.7), and
`cvvdp_parameters.json` is now a byte-copy of v0.5.7's (it differs from
v0.5.4's only in the `version` string).
Refresh by re-pulling these two files and re-running
`cargo test -p cvvdp-gpu --features cubecl-types presets::tests` to
confirm the registry still parses every preset.

## Cite

If you publish work using ColorVideoVDP's perceptual model, please
cite the upstream paper per the repository's citation guidance:

Mantiuk, R. K., Hammou, D., Hanji, P. (2024). *ColorVideoVDP: A
visual difference predictor for image, video and display
distortions.* SIGGRAPH.
