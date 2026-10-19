# Primary MDSI scalar grid

Missing: qualified cold/warm ratio intervals. The author M-file returns one scalar and exposes no map output.

All 24 cases pass the crate README’s existing 1e-9 relative bound: three retained content pairs × 64², 255×257, 1024² and 3840×2160 × thread budgets 1 and 28. Maximum absolute difference is 8.17512724182734e-13; maximum relative difference is 7.364215176352893e-12. Both canonical calls repeat exactly within each process.

The reference is the unchanged author [MDSI.m release](https://www.mathworks.com/matlabcentral/fileexchange/59809-mean-deviation-similarity-index-mdsi), archive SHA256 `8b78597d180d5e250a928fbee7b5ffb8a714ce354e3e8640eec7703692db472b`, M-file SHA256 `4dff198b28e57b39b20877c7bcb75af2c6278c95f9f321362a2046ee3e37351b`. It runs through Octave 6.4 with image 2.12 `fspecial`; this path uses average kernels and `conv2`, without the resize dependency from the separate VSI finding. The reference’s default summation method, constants and complex-root expression are unchanged. Octave’s default `mad` implements mean absolute deviation, as the caller expects.

Rust is the frozen e2d414c3 build. Raw RGB8 input bytes are common to both arms; file reads and raw-to-array layout preparation precede the reference calls. [The result index](primary-mdsi.json) binds three small scalar shards and all raw artifact hashes. Each row includes both process maximum RSS values. The whole panel’s resource receipt is rc=0, 20s, peak RSS 0.62 GiB, minimum available 13822 MiB, peak load 1.25. These diagnostic first/warm clocks supply no paired speed ratio. [Exact commands and hashes](primary-mdsi.meta) preserve the source, runtime, binary and input bindings.
