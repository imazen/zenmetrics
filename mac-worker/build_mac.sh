#!/usr/bin/env bash
# build_mac.sh -- build the native macOS worker binaries + assemble the payload, ON a Mac that has the
# zen workspace cloned (the executor needs the sibling repos). The macOS analog of build_win.bat.
#
# Produces  mac-worker/payload/  containing:
#   zenmetrics  zenfleet-worker  s5cmd  run.sh  install.sh  uninstall.sh
#   com.imazen.zenworker.plist  worker.env.example
# Then: cp worker.env.example payload/worker.env, fill in creds (ubuntu-node/mint_cred.sh), and
#       `cd payload && sudo bash install.sh`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"          # the zenmetrics repo root
OUT="$HERE/payload"

# --- host arch -> rust target + s5cmd asset ---
case "$(uname -m)" in
  arm64)  TARGET="aarch64-apple-darwin"; S5="macOS-arm64" ;;
  x86_64) TARGET="x86_64-apple-darwin";  S5="macOS-64bit" ;;
  *) echo "unsupported arch $(uname -m)"; exit 1 ;;
esac
echo "== building for $TARGET =="
command -v cargo >/dev/null || { echo "FATAL: cargo not on PATH (install rustup)"; exit 2; }
rustup target add "$TARGET" 2>/dev/null || true

# --- sanity: the executor's sibling repos must be present (path = ../… in Cargo.toml) ---
missing=()
for s in ../zenpng ../zenwebp ../zenavif ../zenjxl ../jxl-encoder/jxl-encoder ../zensim/zensim \
         ../../butteraugli/butteraugli ../fast-ssim2/fast-ssim2; do
  [ -d "$REPO/$s" ] || missing+=("$s")
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "FATAL: the zenmetrics executor needs the sibling repos, but these are missing next to the repo:"
  printf '  %s\n' "${missing[@]}"
  echo "Clone the zen workspace in the expected layout (see README 'Workspace prerequisite'), then retry."
  echo "(You can still build the standalone zenfleet-worker with: cargo build --release -p zenfleet-worker)"
  exit 3
fi

mkdir -p "$OUT"
cd "$REPO"

echo "== [1/3] zenmetrics executor (codecs + CPU metrics) =="
cargo build --release --target "$TARGET" \
  -p zenmetrics-cli --no-default-features --features sweep,png,jpeg,webp,avif,jxl,cpu-metrics

echo "== [2/3] zenfleet-worker (claim/ledger loop) =="
cargo build --release --target "$TARGET" -p zenfleet-worker

echo "== [3/3] assemble payload =="
cp "target/$TARGET/release/zenmetrics"      "$OUT/"
cp "target/$TARGET/release/zenfleet-worker" "$OUT/"
cp "$HERE/run.sh" "$HERE/install.sh" "$HERE/uninstall.sh" \
   "$HERE/com.imazen.zenworker.plist" "$HERE/worker.env.example" "$OUT/"
chmod +x "$OUT"/*.sh "$OUT/zenmetrics" "$OUT/zenfleet-worker"

# s5cmd (matching arch) from the pinned upstream release
S5VER="2.2.2"
if [ ! -x "$OUT/s5cmd" ]; then
  echo "== fetching s5cmd $S5VER ($S5) =="
  curl -fsSL "https://github.com/peak/s5cmd/releases/download/v${S5VER}/s5cmd_${S5VER}_${S5}.tar.gz" \
    | tar -xz -C "$OUT" s5cmd && chmod +x "$OUT/s5cmd"
fi

echo ""
echo "payload ready: $OUT"
ls -la "$OUT"
echo ""
echo "next:  cp $OUT/worker.env.example $OUT/worker.env   # fill in scoped creds (ubuntu-node/mint_cred.sh)"
echo "       cd $OUT && sudo bash install.sh              # register the launchd LaunchDaemon"
