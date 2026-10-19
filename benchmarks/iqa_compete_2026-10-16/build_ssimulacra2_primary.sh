#!/usr/bin/env bash
# Build the declared-primary SSIMULACRA2 oracle (cloudinary/ssimulacra2
# v2.1, commit fb5d0dbe866e5113d320666a09ddbc8d145b5fde) plus an owned
# Little-CMS >= 2.13 prefix, satisfying upstream's unmodified
# "find_package(LCMS2 2.13)" requirement.
#
# Usage: build_ssimulacra2_primary.sh <ssim2_tarball> <lcms2_tarball> <output_dir> [rgb-input [hwy_tarball]]
#   ssim2_tarball : official GitHub archive of commit fb5d0dbe
#   lcms2_tarball : official Little-CMS lcms2.13 archive (commit ac4fe46f)
#   output_dir    : must NOT exist — atomically reserved by this script;
#                   any pre-existing file/dir/symlink is refused
#
# Source identity is bound by full sha256 of BOTH input archives (below),
# verified before extraction; the build uses only the trees extracted
# inside the reserved output dir. Paths are argv throughout — never
# interpolated into shell source.
#
# Source pins (official URLs):
#   https://github.com/cloudinary/ssimulacra2/archive/fb5d0dbe866e5113d320666a09ddbc8d145b5fde.tar.gz
#   https://github.com/mm2/Little-CMS/archive/refs/tags/lcms2.13.tar.gz  (tag lcms2.13)
#
# Emits per-stage "rc=N" lines and receipts to stdout; a failing stage's
# rc is logged before the script exits with it. Caller wraps in
# locks/run-heavy/tee. No -march=native anywhere: upstream CMakeLists
# carries -O3; hwy does runtime target dispatch.
set -euo pipefail

SSIM2_COMMIT=fb5d0dbe866e5113d320666a09ddbc8d145b5fde
SSIM2_URL=https://github.com/cloudinary/ssimulacra2/archive/$SSIM2_COMMIT.tar.gz
SSIM2_SHA=530e9b23b5be10f5981367ba06ce804b3b536a409fcb3b510fa420170304d9c8
LCMS2_COMMIT=ac4fe46f2b4d84bae2ed6cc77b044976282098a5
LCMS2_URL=https://github.com/mm2/Little-CMS/archive/refs/tags/lcms2.13.tar.gz
LCMS2_SHA=16de4bc1e83a5f8aa4e395d9734396cb6ffe7263648d35f39b0fcf933082f93b
HWY_COMMIT=457c891775a7397bdb0376bb1031e6e027af1c48
HWY_URL=https://github.com/google/highway/archive/$HWY_COMMIT.tar.gz
HWY_SHA=5124b0501c98d9930dbb065bfa1a5bbbd59ce0f12facb7e1e33aaef01a5f1f1a

[ $# -ge 3 ] && [ $# -le 5 ] || { echo "usage: $0 <ssim2_tarball> <lcms2_tarball> <output_dir> [rgb-input [hwy_tarball]]" >&2; exit 2; }
build_target=ssimulacra2
cmake_extra=()
if [ $# -ge 4 ]; then
  [ "$4" = rgb-input ] || { echo 'unknown adapter mode' >&2; exit 2; }
  adapter_dir=$(dirname -- "$(realpath -e -- "$0")")
  cmake_extra=("-DCMAKE_PROJECT_LIBJXL_INCLUDE=$adapter_dir/ssimulacra2-primary-rgb.cmake")
  build_target=iqa-ssim2-primary
fi
ssim2_tgz=$(realpath -e -- "$1")
lcms2_tgz=$(realpath -e -- "$2")
output=$(realpath -m -s -- "$3")

stage() {
  local name=$1; shift
  echo "== $name"
  local rc=0
  # Do not use an OR-list: it disables errexit inside shell functions.
  set +e
  (set -e; "$@")
  rc=$?
  set -e
  echo "rc=$rc"
  [ "$rc" -eq 0 ] || exit "$rc"
}
die() { echo "refuse: $*" >&2; exit 1; }

check_archive() { # tag path want-sha
  local got
  got=$(sha256sum -- "$2" | cut -d' ' -f1)
  [ "$got" = "$3" ] || { echo "MISMATCH $1: want $3 got $got ($2)"; return 1; }
  echo "$1 $2 $got  (pin: $4)"
}

# --- preflight: verify both official archives before ANY output mutation ---
echo "ssim2 pin  $SSIM2_URL  commit=$SSIM2_COMMIT"
echo "lcms2 pin  $LCMS2_URL  commit=$LCMS2_COMMIT (tag lcms2.13)"
stage hash-ssim2 check_archive ssim2 "$ssim2_tgz" "$SSIM2_SHA" "$SSIM2_URL"
stage hash-lcms2 check_archive lcms2 "$lcms2_tgz" "$LCMS2_SHA" "$LCMS2_URL"
if [ $# -eq 5 ]; then
  hwy_tgz=$(realpath -e -- "$5")
  stage hash-hwy check_archive hwy "$hwy_tgz" "$HWY_SHA" "$HWY_URL"
fi

# --- preflight: refuse any pre-existing/symlink output; reserve atomically ---
if [ -e "$output" ] || [ -L "$output" ]; then
  die "output path exists or is a symlink: $output"
fi
stage reserve-output mkdir -- "$output"
export TMPDIR="$output/scratch"
stage scratch-dir mkdir -- "$TMPDIR"
prefix=$output/lcms2-prefix
stage builder-receipt sha256sum -- "$(realpath -e -- "$0")"
if [ "$build_target" = iqa-ssim2-primary ]; then
  stage adapter-receipt sha256sum -- "$adapter_dir/ssimulacra2-primary-rgb.cmake" "$adapter_dir/ssimulacra2_primary_compute.cpp"
fi

# The pinned archives have one top-level directory; strip only that component.
# Every command runs inside stage(), including extraction failures.
extract_one() {
  mkdir -- "$2"
  tar -xzf "$1" --strip-components=1 -C "$2"
}
ssim2_src=$output/src-ssim2-extract
lcms2_src=$output/src-lcms2-extract
stage extract-ssim2 extract_one "$ssim2_tgz" "$ssim2_src"
stage extract-lcms2 extract_one "$lcms2_tgz" "$lcms2_src"
echo "ssim2_src=$ssim2_src"
echo "lcms2_src=$lcms2_src"

# --- lcms2: out-of-tree autotools, static only (configure runs with
# CWD = build dir via a subshell — no shell-source interpolation) ---
stage lcms2-build-dir mkdir -- "$output/lcms2-build"
lcms2_configure() { (cd "$output/lcms2-build" && "$lcms2_src/configure" --prefix="$prefix" --disable-shared); }
stage lcms2-configure lcms2_configure
stage lcms2-build    make -C "$output/lcms2-build" -j"${CARGO_BUILD_JOBS:?}"
stage lcms2-install  make -C "$output/lcms2-build" install
stage lcms2-receipt  sha256sum "$prefix/lib/liblcms2.a"

# Optional fully owned HWY source/build. Explicit cache paths prevent selecting
# the pre-existing system archive or headers. Default three/four-arg paths stay
# separate historical build modes with their documented provenance limitation.
if [ $# -eq 5 ]; then
  hwy_src=$output/src-hwy-extract
  hwy_prefix=$output/hwy-prefix
  stage extract-hwy extract_one "$hwy_tgz" "$hwy_src"
  stage hwy-cmake cmake -S "$hwy_src" -B "$output/hwy-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX="$hwy_prefix" -DCMAKE_INSTALL_LIBDIR=lib \
    -DHWY_ENABLE_CONTRIB=OFF -DHWY_ENABLE_EXAMPLES=OFF -DHWY_ENABLE_TESTS=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  stage hwy-build cmake --build "$output/hwy-build" --target hwy --parallel "${CARGO_BUILD_JOBS:?}"
  stage hwy-install cmake --install "$output/hwy-build"
  stage hwy-build-receipt sha256sum -- "$output/hwy-build/CMakeCache.txt" \
    "$output/hwy-build/compile_commands.json" "$hwy_prefix/lib/libhwy.a" "$hwy_src/LICENSE"
  cmake_extra+=("-DHWY_INCLUDE_DIR=$hwy_prefix/include" "-DHWY_LIBRARY=$hwy_prefix/lib/libhwy.a")
fi

# --- ssimulacra2: unmodified upstream CMake against the owned prefix ---
export PKG_CONFIG_PATH="$prefix/lib/pkgconfig"
if [ $# -eq 5 ]; then export PKG_CONFIG_PATH="$hwy_prefix/lib/pkgconfig:$PKG_CONFIG_PATH"; fi
stage ssim2-cmake \
  cmake -S "$ssim2_src/src" -B "$output/build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$prefix" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "${cmake_extra[@]}"
stage ssim2-build \
  cmake --build "$output/build" --target "$build_target" \
        --parallel "${CARGO_BUILD_JOBS:?}"
stage ssim2-retain cp "$output/build/$build_target" "$output/ssimulacra2"
stage ssim2-receipt sha256sum "$output/ssimulacra2"

# --- receipts: toolchain, cache, static-HWY identity, library paths ---
toolchain_versions() {
  cc --version
  c++ --version
  cmake --version
  ninja --version
  getconf GNU_LIBC_VERSION
  local compiler
  while IFS= read -r compiler; do
    echo "CMake compiler=$compiler resolved=$(realpath -e -- "$compiler")"
    sha256sum -- "$compiler"
    "$compiler" --version
  done < <(sed -n -E 's/^CMAKE_(C|CXX)_COMPILER:FILEPATH=//p' "$output/build/CMakeCache.txt")
  sha256sum -- "$lcms2_src/COPYING" "$ssim2_src/LICENSE"
}
stage toolchain-versions toolchain_versions
stage cache-receipt \
  sha256sum "$output/build/CMakeCache.txt" "$output/build/compile_commands.json"
hwy_receipt() { # build-dir: hash the cache-resolved static HWY + record provenance caveat
  local lib
  lib=$(sed -n 's/^HWY_LIBRARY:FILEPATH=//p' "$1/build/CMakeCache.txt")
  echo "HWY_LIBRARY(resolved)=$lib"
  if [ -n "${hwy_src:-}" ]; then
    echo "HWY provenance: owned archive=$HWY_SHA commit=$HWY_COMMIT; compiler/cache receipts above"
    test "$lib" = "$hwy_prefix/lib/libhwy.a"
    grep -Fx "HWY_INCLUDE_DIR:PATH=$hwy_prefix/include" "$1/build/CMakeCache.txt"
  else
    echo "HWY provenance: pre-existing system archive; its own source/compiler UNBOUND"
  fi
  test -f "$lib"
  sha256sum -- "$lib"
  grep -E '^HWY_(LIBRARY|INCLUDE_DIR):' "$1/build/CMakeCache.txt"
}
stage hwy-identity hwy_receipt "$output"
stage ssim2-readelf  readelf -d "$output/ssimulacra2"
lib_paths() { # ldd-resolved paths: direct NEEDED + transitives
  ldd "$1" | awk '{for(i=1;i<=NF;i++) if($i ~ /^\//) print $i}' | sort -u
}
lib_hashes() {
  local lib
  while IFS= read -r lib; do
    echo "library=$lib resolved=$(realpath -e -- "$lib")"
    sha256sum -- "$lib"
  done < "$output/library-paths.txt"
}
stage ssim2-ldd ldd "$output/ssimulacra2"
capture_lib_paths() { lib_paths "$output/ssimulacra2" > "$output/library-paths.txt"; cat "$output/library-paths.txt"; }
stage ssim2-lib-paths capture_lib_paths
stage ssim2-lib-hashes lib_hashes "$output/ssimulacra2"
echo "== done"
