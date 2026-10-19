# Primary GMSD score and map grid

Missing: qualified cold/warm ratio intervals. The odd-size primary C cases fail before producing a usable score or map; they remain findings.

All 18 even-size cases pass: three retained content pairs at 64², 1024² and 3840×2160, with thread budgets 1 and 28. Every GMS map is bit-identical. Maximum scalar absolute difference is 3.587408148320037e-15, maximum relative difference is 1.4294167329816467e-13 against the existing 1e-12 relative bound. The map mean is identical in all 18 cases against its existing 1e-14 absolute bound. These bounds come from the [prior GMSD validation record](../gmsd_parity_2026-09-22.md) and the crate’s unit tests.

The primary is unchanged [libgmsd de646c9a](https://github.com/clunietp/libgmsd/tree/de646c9a957a892e4b9d78ff94601b83b363c158). The adapter compiles the original CLI source with its main renamed and calls its unchanged `rgb2gray` function. Rust uses the retained e2d414c3 baseline rlibs, with actual crate source checked byte-for-byte against Git. Every map-probe result also equals the existing frozen CLI score exactly; both Rust map calls repeat exactly. The source and compiler receipts bind all three probe executables.

All six 255×257 primary C runs fail with an AddressSanitizer heap-buffer-overflow in `downsample_2x2`, `src/sample.c:24`. The write reaches the end of the floor-sized 65024-byte output allocation. The requested inputs are retained at their original geometry. Rust returns its documented floor-sized map; no canonical odd-size parity is claimed, and the diagnostic C build supplies no performance result.

[Result index](primary-gmsd.json) binds three small TSV shards and the raw artifact manifest. The exporter rechecks all raw hashes, case membership, actual map bytes and unchanged scalar bounds. Per-process RSS for C, the Rust map probe and the frozen CLI is retained. The whole panel’s resource receipt is rc=0, 18s, peak RSS 0.26 GiB, minimum available 14452 MiB, peak load 2.10. [Commands and evidence hashes](primary-gmsd.meta) also retain the first adapter-formatting failure; its exact transport assertion remains unchanged.
