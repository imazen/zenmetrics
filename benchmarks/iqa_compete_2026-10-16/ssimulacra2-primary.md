# ssimulacra2 primary oracle build

Builds the **declared-primary SSIMULACRA2 source** — `cloudinary/ssimulacra2`
tag `v2.1` = commit `fb5d0dbe866e5113d320666a09ddbc8d145b5fde` — with its
unmodified dependency contract (`find_package(HWY 0.15.0)`,
`find_package(LCMS2 2.13)`).

## Source pins

| dep | source | identity |
|---|---|---|
| ssimulacra2 | [official commit archive](https://github.com/cloudinary/ssimulacra2/archive/fb5d0dbe866e5113d320666a09ddbc8d145b5fde.tar.gz) | tag `v2.1`, vendored libjxl 0.8.0; top-level Apache-2.0 license, vendored files retain their notices; tarball sha256 `530e9b23b5be10f5981367ba06ce804b3b536a409fcb3b510fa420170304d9c8` |
| Little-CMS | [official tag archive](https://github.com/mm2/Little-CMS/archive/refs/tags/lcms2.13.tar.gz) | tag `lcms2.13` → commit `ac4fe46f2b4d84bae2ed6cc77b044976282098a5`, MIT; tarball sha256 `16de4bc1e83a5f8aa4e395d9734396cb6ffe7263648d35f39b0fcf933082f93b` |
| hwy | system `/usr/local/lib` (pre-existing) | `libhwy.a` 1.2.0 — satisfies the `HWY 0.15.0` floor; **source/compiler provenance UNBOUND** — the script hashes the cache-resolved archive and flags this explicitly |
| png/jpeg/z/exr | system `/usr/lib/x86_64-linux-gnu` | image codec deps of the vendored extras |

Third-party archives/binaries live outside git (owned `~/tmp` prefixes);
only this script + doc are tracked.

## Build

```bash
benchmarks/iqa_compete_2026-10-16/build_ssimulacra2_primary.sh \
    <ssim2_tarball> <lcms2_tarball> <output_dir>
```

Inputs are the two **official archives**, not extracted trees:

1. **Hash gate** — both tarballs are sha256-verified against the pinned
   constants in the script *before any output mutation*; a mismatch is a
   loud `MISMATCH` + rc 1.
2. **Reservation** — `<output_dir>` must not exist (any pre-existing
   file, dir or symlink is refused); it is created atomically and every
   build artifact lands inside it. Receipts are immutable — a second run
   needs a fresh output dir.
3. Sources are extracted inside the reserved dir and only those trees
   are built: lcms2 out-of-tree autotools `--disable-shared` → static
   `liblcms2.a` in `<output>/lcms2-prefix`; ssimulacra2 `cmake
   -S <src>/src -B <out>/build -G Ninja -DCMAKE_BUILD_TYPE=Release
   -DCMAKE_PREFIX_PATH=<prefix> -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
   with `PKG_CONFIG_PATH=<prefix>/lib/pkgconfig`, target `ssimulacra2`.
4. No `-march=native`: upstream `src/CMakeLists.txt` sets `-O3`; hwy
   does its own runtime dispatch. `LCMS2 2.13` is met unmodified — no
   version-floor patching, no `SKCMS`.
5. Paths are passed as argv and normalized to absolute paths. The helper
   does not interpolate them into shell source. Quoted archive paths pass
   a real build; quoted output paths pass extraction/configure argv
   controls. Full upstream-tool compatibility with quoted build directories
   is not established. Scratch lives inside the reserved output directory.

Reservation, extraction, build and receipt stages log `rc=N` before
exiting on failure. Stage functions run with errexit in a subshell; an
OR-list around a function would disable that protection.

Receipts emitted (each as `== stage` + `rc=N`): helper hash, both archive hashes,
`liblcms2.a` sha, binary sha, actual cache-selected compiler paths/hashes and versions,
CMake/Ninja/glibc versions,
`CMakeCache.txt` + `compile_commands.json` shas, the cache-resolved
static `libhwy.a` path+sha (with the unbound-provenance flag),
`readelf -d`, `ldd`, and sha256 of every `ldd`-resolved library path and its symlink target
(direct NEEDED + transitives). The ldd/lib-hash stage reports a
*library-path* receipt only — it is not a full source/runtime closure
claim (the static deps' own provenance is not binary-bound).

## Controls (usage-only, no scoring)

`ssimulacra2` with `argc != 3` prints the usage block to stderr and exits
1 — `ssimulacra2_main.cc:13-42` shows that path performs no file reads
and calls no scoring API. No image/parity/timing use of the binary is
authorized by this document; score-bearing runs need a separate
coordinator-approved protocol.

## Build-only verification, 2026-10-19

Numeric qualification and complete source/runtime closure remain missing.
HWY's source/compiler provenance is unbound. This build does not bind the
historical reference-capture binary or establish a raw-RGB adapter contract.

The fresh build at `~/tmp/swe2opt/ssim2-primary-codex-20261019-r3/`
uses GCC 11.4.0, CMake 4.2.1, Ninja 1.13.0.git.kitware.jobserver-pipe-1
and glibc 2.35. It satisfies LCMS2 2.13 without source edits. The upstream
configure script prints `LIB_PLUGINS: command not found` but returns zero;
its static-only configuration and build complete.

- Binary SHA256: `f33395a90183e9494f4d50f4d292d4e0fcdde53a2339cd2713cd353ec9dd2e47`.
- CMake cache SHA256: `6c80f17fea714fc2c30f7ba7303ae96fc0f0c267d5023d2dd3a7731351e27d50`.
- Compile commands SHA256: `6572284113e49df465f1a7b4588a9a233ef0ab2527f259579445e6b2c2aafd4a`.
- Resolved HWY archive SHA256: `4d9566701ae8cf1520020139cda9a39bacaab11c337266992c0a7112f69c2380`.

Full stdout/stderr and all direct/transitive dynamic library paths, resolved
targets and hashes are retained in
`~/tmp/handoff/aic2026-ssim2builder-codex-20261019/build-r3.log`.
The two host heavy locks serialize the build; `run-heavy --mem 12G --jobs 8`
reports `done rc=0 15s | peak-RSS 0.42GiB | min-avail 11358MiB | peak-load 3.76`.
Both wrapper and tee return zero. These are build-resource observations,
not metric performance measurements.

Controls refuse either wrong archive hash without creating the output,
and refuse existing directories, files, live symlinks and dangling symlinks.
A retained sentinel stays byte-identical. Injected extraction and configure
failures retain their exact exit codes; a failed first extraction command
prevents the next command. Quoted-path controls inspect exact argv.
The final HWY receipt guard is separately tested with a present and absent
archive (rc 0 and 1). It was tightened after compilation; no build command
or input changed. The build log records the pre-tightening helper hash.
No-argument execution prints usage and returns 1, without scoring.
Control logs and commands live beside the build log under `controls/` and
`configure-argv/`; `report.md` records the frozen candidate and file hashes.
All previous source, build and control artifacts remain preserved.

## Opt-in raw RGB target

Append `rgb-input` to the builder's three arguments to select the benchmark
adapter in [ssimulacra2_primary_compute.cpp](ssimulacra2_primary_compute.cpp).
The default remains the official file-decoding CLI. A CMake project hook adds
the adapter target and compiles the exact upstream `ssimulacra2.cc`; upstream
source files remain unchanged. This starts a separate executable cohort.

The adapter accepts the existing harness argument order:
`ssim2 width height reference.rgb distorted.rgb threads score samples`.
It checks positive integers, minimum 8×8 geometry, size overflow and exact
RGB8 file lengths. File reads, raw buffers and the reusable thread runner are
outside timing. Each timed call creates metadata and image bundles, converts
both RGB8 inputs, calls the canonical metric and destroys per-call objects.
Metadata declares RGB sRGB, 8-bit samples and the upstream SDR intensity of
255. It does not pre-linearize input or substitute another color-management
implementation. Scores retain double precision with 17 significant digits.

The first measured call emits `COLD`; subsequent calls emit `SAMPLE`.
Cold here means the first score call after file loading and runner creation,
not process startup. The requested budget controls ingress workers; the
upstream metric has no pool parameter and scores on the calling thread.
`score+map` emits `UNSUPPORTED_MAP` before file reads; no map is synthesized.
The new output and scope require explicit cohort selection by the collector.
Source/build and refusal controls alone do not establish numerical parity or
qualified ratios. Existing library routes and metric code are unchanged.

The opt-in target built on 2026-10-19 with binary SHA256
`2e78406631721d54524849319340ac887a34a34edce8372c653c96aaf670b399`.
Its complete source extraction matches all 196 regular files in the pinned
official archive. Minimum-size, odd-size and 64×64 synthetic pairs agree with
the official CLI's eight-decimal display at ingress budgets 1 and 2; cold and
warm score transport agrees exactly within each process. This compares two
ingress routes to the same canonical scoring body, not Rust against C++.
Argument, overflow, absent/truncated/oversized file and unsupported-map
controls also pass. The map refusal succeeds with absent input paths.

The repeatable [control program](check_ssimulacra2_primary.py) runs no scoring
unless supplied `--official`. The corresponding `oracle.just` ingress recipe
explicitly opts into both canonical binaries. Full logs and input/output
receipts are retained under `~/tmp/portcheck-primary-rgb-controls-20261019T0951/`.
Build receipt: `run-heavy: done rc=0 15s | peak-RSS 0.42GiB | min-avail 11281MiB | peak-load 6.43`.
Control receipt: `run-heavy: done rc=0 0s | peak-RSS 0.02GiB | min-avail 13180MiB | peak-load 3.72`.
Both hold the two host heavy locks and use a 12 GiB cap. These receipts cover
whole build/control processes, not metric-call memory or speed comparisons.
