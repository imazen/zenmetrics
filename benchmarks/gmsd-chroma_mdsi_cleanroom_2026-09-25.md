# MDSI clean-room integration record — 2026-09-25 (quarantined, not pushed)

Implementation commit `19b8c03c87934b1df2e078aa689534d17777713c` (`crates/gmsd/src/mdsi.rs`); driver `scripts/gmsd-chroma/run_mdsi_cleanroom_final.sh` (run `mdsi_cr_final_v4`, rc 0, 137 s, peak RSS 1.32 GiB, `run-heavy` without the shared lock; see the worklog). Every command's UTC/argv/rc/log sha is in `/var/tmp/gmsd-chroma/commands.jsonl`.

## Provenance

The MDSI implementation was written on an isolated machine from the IEEE Access 2016 paper alone, with no MDSI code in any language available to it, and delivered as `mdsi_cleanroom.rs` + `CHOICES.md` + `scores.tsv` (`/var/tmp/mdsi-cleanroom-result/`, `mdsi_cleanroom.patch` sha256 `d7fd27d8f6154ab920e08899519c175789dd1913de9caee3513896a0bebf4420`). Its validation target was the black-box scores our lane had produced with the authors' reference software (`target.tsv`, 116 pairs, author scores identical to the lane's `parity.json` values). This lane integrated it: the delivered arithmetic became `mdsi::reference` (unchanged apart from plumbing) and the public entry is an optimized path pinned to it bit for bit. Only after that validation was the previous transcription read, and only for optimization ideas: the transcription's SIMD scaffolding was reused; none of its arithmetic order, pooling or boundary handling was. (Reused scaffolding: the 7-plane layout, the `similarity_row` row-kernel structure, the tier-dispatch macro and `maps()`, the integer downsample-factor expression, the `black_box` exponent pin.) The transcription, `LICENSE-MDSI`, the Octave driver scripts and wording about the reference software's internals are removed from the tree (drivers and unredacted originals: `gmsd-chroma_oracle_archive.pointer.md`; gate data: `gmsd-chroma_mdsi_gate_bundle.pointer.md`).

## Gates (run `mdsi_cr_final_v4`)

| Gate | Result |
|---|---|
| 116-pair author-score gate (`author_score_gate`, `GMSD_MDSI_GATE=require`) | max abs 4.298904288102534e-11, max rel 4.836930021352448e-10 (bound 1e-9, unchanged); wrong C3=5500 fails 109/116 (the 7 survivors are exact-zero pairs) |
| Oracle score parity (`compare_oracle.py`, 116 pairs) | passed: same maxima; wrong constant C3=55 rejected 109; 6,727 negative-GCS samples in the author maps exercised |
| Per-pixel maps vs the authors' maps | GCS max abs 5.773159728050814e-15, max rel 2.7362473158272502e-11; CS max abs 5.551115123125783e-16, max rel 5.663248074773508e-16 (gate 1e-12 abs / 1e-9 rel) |
| Fast path vs straight-line reference | bit-identical score and GCS map: 14 sizes × packed/strided (tests) and all 116 oracle pairs (`mdsi_oracle` asserts it) |
| Tiers and threads | scalar, v3, v4 (AVX-512) and 1/8 threads bit-identical on the 116 pairs (`bitwise_checks`) |
| Crate tests | 28 debug + 28 release (all features), 25 no-default: all pass; clippy `-D warnings` (all targets, all features; no-default) and `fmt --check` clean |
| MS-GMSD / MS-GMSDc | unchanged (max rel 2.96e-12 / 2.28e-12 vs their NumPy transcription; not author-software parity) |
| Colour panel | 52 rows identical to the previous table (see `gmsd-chroma_colour_2026-09-24.md`) |

## Found on the way (recorded, fixed)

- Runs `mdsi_cr_final_v2` and `_v3` (rc 101) failed the no-default test: the straight-line reference used the crate's no_std Newton `sqrt_f64` (can be 1 ulp off) while the SIMD path's `sqrt` is correctly rounded, so one GCS pixel differed by 3e-16 (score 0.1509332651429939 vs 0.15093326514299593 at 2x2). Fixed with a local correctly rounded `sqrt` (`libm::sqrt` in no_std); logs retained. The `black_box` pins on the fourth-root exponent and the complex-root angle are defensive; no libm rewrite of 0.25 was observed.

## Limits (not measured or not claimed)

- The fourth root is the platform libm `powf(x, 0.25)` (`libm::pow` in no_std): the score can differ in the last bits across platforms, well inside the 1e-9 gate (about 2× margin). The gate test is `std`-only, so **no_std parity with the authors' scores is unmeasured** (no_std fast-vs-reference identity is tested).
- Tiers exercised: scalar, v3, v4 on an AVX-512 x86 host. neon and wasm128 were not exercised.
- The 4.8e-10 residual against the authors' scores is not explained (`docs/MDSI_CHOICES.md`). It was not chased.
- Speed is CONTENDED (see `gmsd-chroma_speed_2026-09-24.md`). The optimized path is 3× to 33× faster than the straight-line reference and, at 4096² single-threaded, faster than plain GMSD on this host.
- Correction to earlier wording: the scores the clean-room was validated against were *computed by running the authors' released software*, not numbers the authors published.

## Files (sha256)

`/var/tmp/gmsd-chroma/parity.json` 8fbc1f9e886f8677ad7a6cba533071b8e258ec8542a1b380cf277fc12e92b593; `ms_parity.json` 07ab868c815240adf46f1ad4ba3dbbaed2121153d58ad4e7737588a7e82c24bf; `colour_v3/predictions.tsv` 8e71a70901d020c592956687d7c5d19feb0873aed60b8c46943dd2ccae5ec646; `colour_v3/report.json` 8663ff467048c8ca05185f8b56c51a28b73ff6b583c700b5db5ac44925bd8593; `speed/report.json` 388489cec554479ace0fb808afd6868094143ac3aa863c6feacc97b6a5a3d266; `mdsi_vs_reference.tsv` 53dd38cae80523b43b9df1e2ec5d6aed71be5165f1500d5f18ac15ee2e7e01cd; gate table `mdsi_gate/target.tsv` 310d9617690a37c4a880a3bcd0cda51c87ab76c26d521ebe89016d1d7a35374a (kept outside the repo). Recompute: `python3 scripts/gmsd-chroma/recompute_headlines.py {mdsi_parity,mdsi_maps,colour,speed,mdsi_vs_reference}`.

## Re-validation at the review-corrected tip

After the review corrections (test module split, records, scripts moved out) the validation was re-run at commit `10b79a1da8d06e9a25e22c9ccaa53acfce4a3b7f` (`mdsi_cr_validate_v5`, `validate_oracles.sh`, rc 0, log sha `aca7932c06b3…`): 28 debug + 28 release + 25 no-default tests pass with the gate required, clippy/fmt clean, oracle score and map parity unchanged (max abs 4.298904288102534e-11, max rel 4.836930021352448e-10). New `parity.json` sha256 `84b6946e62e3e5cb6447ef51ba96822bf64f4bae8577862f1ce611c0198944ec`, `ms_parity.json` `fd7e9a005dcc7be992ccf6aa170bcc9d6c4c804932ddc70cadd48300e5117e7d` (the v4 copies are in `/var/tmp/gmsd-chroma/run_v4_outputs/`). The colour predictions and the speed record above were produced at `19b8c03c`; the later commits move tests, comments and records only, so they were not re-run.
