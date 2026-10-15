# LDR-FLIP and CW-SSIM: JPEG AIC-4 checks

FLIP is a Rust port of the LDR algorithm in [NVlabs/flip](https://github.com/NVlabs/flip/blob/d3ab02a4425eb167d8a4021b382a2bafc6e6c838/src/cpp/FLIP.h), pinned to `d3ab02a4425eb167d8a4021b382a2bafc6e6c838`. The CLI metric `flip` emits `flip` (lower is better), using the reference viewing conditions: distance 0.7 m, width 3840 pixels, physical width 0.7 m. The BSD-3-Clause notice is retained in `crates/zenmetrics-cli/LICENSE-flip`.

The oracle is `flip-evaluator==1.7`; its source distribution SHA-256 is `88204ec5000fa90b26e00f6f21e8d73f18e5afac7e8a23cb0803b117b3da081b`. It reproduces all 61 published AIC `FLIP` values exactly. Version 1.7 sums its map in row order in f32. The CLI retains that convention; the pinned newer reference's f64 sum would change the published column. Pixel arithmetic, CSF filtering, Hunt-adjusted colour difference, edge/point detection and error exponent follow the C++ reference. Images use native resolution with nearest-sample border extension.

The 61-pair published gate has median/max scalar error **0 / 0**. The 21 ladder maps (three images at 64, 128, 256, 512, 1024, 2048 and 4096 square pixels) have maximum per-pixel error **2.9802322387695312e-8** and maximum scalar error **0**, below the requested 1e-6 gates. Packed and strided RGB ingress must agree bit for bit.

`flip_probe.rs` exports the production Rust map and mean without involving a decoder. `verify_flip.py` decodes the same RGB8 PNGs for both implementations and compares every raw cell; no colormap or screenshot is compared. Paths to the external AIC and ladder fixtures are explicit in the script. Install the oracle in a separate virtual environment; it is not a runtime dependency of the Rust CLI.
