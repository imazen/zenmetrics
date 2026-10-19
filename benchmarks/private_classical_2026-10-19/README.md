# Three-host canonical CW-SSIM-pyiqa timings

Each host has **48 preferred qualified scalar cells**: six frozen RGB geometries,
threads 1/8, three fresh-process constructor-inclusive cold repeats and one warm
cell containing ten clean zenbench rounds at a 100 ms target. RGB loading is
outside the cold timer; warm calls recompute the full pair after initialization.
No maps or 8K cells are included. This is the weightless native Rust route.

The CSVs retain 145 observations across three hosts and select 144 preferred
logical cells. Mac retains one original resource-flagged warm observation and
its clean replacement, using the remaining original attempt. Select
`preferred=True` for comparisons; do not count the diagnostic as another cell.
Private NN routes remain outside this completed classical panel.

Each row includes input/binary/log hashes, score bits, timing statistics and
memory evidence bound to its worker log. Wrapper peak RSS is rounded to 0.01 GiB;
exact native OS maximum RSS is included only when recorded. Memory covers the
worker scope including setup, rather than metric heap or GPU VRAM. All 145 rows
have bound memory evidence. Each host used its shared 12G/8-job resource policy;
platform guard semantics remain those recorded by the existing resource wrapper.

The per-host protocol JSON binds the CSV, source reports, build receipt and
worker source revision. Worker and dependency versions remain explicit; these
rows are not pooled with historical implementations or portable 5/3 panels.
The private collector is revision `fcf025d0ba54`; the public attempt exporter is
`c986ff68987f`. Only allowlisted aggregates and provenance hashes are published.

These timing observations do not extend native-oracle numerical reproduction or
AIC published-column agreement qualification. Across the 48 matching cells,
i265 and r7900x score bits agree; Mac differs by at most
4.440892098500626e-16. All reference and distorted input hashes match.
