# Primary Wang MS-SSIM scalar grid

Missing: comparable internal map checks and qualified cold/warm timing ratios.
All [24 scalar comparisons](primary-msssim.json) pass the existing 1e-4 absolute
bound. The largest difference is 4.752062978119476e-6. Both original calls
return identical scores within each process, across three raw RGB content
pairs, four geometries and thread budgets 1 and 28.

The primary is Wang's unchanged `msssim.m` plus `ssim_index_new.m`, retained in
[MAD Competition](https://github.com/LabForComputationalVision/MAD_Competition/tree/f67840136fb1b56227b1a7a62a36d61275f176de/msssim).
The caller supplies the crate's adaptive level count, original first-L weights,
an 11x11 Gaussian, and unrounded MATLAB luma. At 64 squared the route uses three
levels; the original default five-level call returns `-Inf` and remains in the
log. No source body changes. The separate Metrix alternative and its approved
0.002 bound are not this reference or gate.

The top-level primary returns a scalar. Its internal SSIM helper exposes maps;
comparison with Rust's optional level-zero map remains open. The whole panel
receipt is rc=0, 44s, peak RSS 0.93 GiB, minimum available 12865 MiB and peak
load 2.34. [Metadata](primary-msssim.meta) binds commands, unchanged source,
executables, inputs, raw logs and per-process RSS. First/warm diagnostic clocks
do not supply paired speed ratios.
