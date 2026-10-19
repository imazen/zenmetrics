# Primary HaarPSI scalar comparison

Missing: map comparison and qualified cold/warm ratio intervals. No speed ratio is inferred from this scalar panel.

All 48 cases pass the crate README’s unchanged 5e-5 scalar gate: `haarpsi` and `haarpsi-y` × three retained content pairs × 64², 255×257, 1024² and 3840×2160 × thread budgets 1 and 28. The largest absolute difference is 5.942815561876813e-8. Rust source is frozen at `e2d414c33a6d3c7da50a818fa36e56654d3ebeef`; all scores repeat within their respective process.

The reference is the unchanged [author HaarPSI.m](https://github.com/rgcda/haarpsi/blob/3b9599b0c1911ee08f1bb991bb19e19d86727819/HaarPSI.m), executed through Octave 6.4. Its hash matches the vendored reference. RGB feeds the original uint8 route; luma explicitly forms unrounded f64 `0.299 R + 0.587 G + 0.114 B` before the author’s grayscale route. Raw input file reads precede score calls. The adapter reshapes the raw input and performs luma conversion inside its diagnostic clocks; these clocks have no paired timing qualification.

[Result index](primary-haarpsi.json) binds the three scalar shards and raw artifact manifest. The publisher rechecks actual stdout score bits, canonical repetitions, recomputed differences, unchanged bound, exact case coverage and GNUtime memory receipts before creating output. Each row records process maximum RSS for both arms; the shared run-heavy receipt covers the whole grid: rc=0, 58s, peak RSS 1.19 GiB, minimum available 12705 MiB, peak load 2.28. [Commands and evidence hashes](primary-haarpsi.meta) preserve the local raw-artifact paths.
