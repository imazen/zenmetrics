/* musl link-compat for the `avif-aom` arm's C oracle (libaom v3.14.1, built by
 * aom-sys-ref with the HOST toolchain against glibc headers): glibc's LFS
 * headers rename fopen -> fopen64, a symbol musl deliberately stopped exporting
 * (1.2.4+; file offsets are always 64-bit there). The oracle only ever opens
 * optional dump/stat files, so aliasing to fopen is exact on musl. Compiled ONLY
 * for *-musl targets (see build.rs); a glibc build never sees this file. */
#include <stdio.h>
FILE *fopen64(const char *path, const char *mode) { return fopen(path, mode); }
