#!/bin/zsh -l
# End-to-end verification of the mac Rust+C dev environment. Runs in a login shell (~/.zprofile loaded).
echo "=== cargo bin tools (~/.cargo/bin) ==="
ls ~/.cargo/bin 2>/dev/null | tr '\n' ' '; echo
echo
echo "=== key tool versions ==="
for t in "cargo nextest" "cargo watch" "cargo audit" "cargo deny" "cargo semver-checks" "cargo expand" "cargo llvm-lines" "cargo asm" "cargo hack" "cargo machete" "cargo flamegraph" "cargo zigbuild" "just" "cross" "sccache" "wasm-pack" "wasm-tools" "bacon"; do
  printf "  %-22s " "$t:"; eval "$t --version" 2>&1 | head -1
done
echo
echo "=== C toolchain end-to-end (compile + run) ==="
D=$(mktemp -d)
cat > "$D/h.c" <<'CEOF'
#include <stdio.h>
int main(){ printf("  C compile+run OK\n"); return 0; }
CEOF
clang "$D/h.c" -O2 -o "$D/hc" && "$D/hc" || echo "  C FAILED"
echo
echo "=== Rust native + cross-compile smoke ==="
printf 'fn main(){println!("  Rust native OK");}\n' > "$D/h.rs"
rustc -O "$D/h.rs" -o "$D/hr" && "$D/hr" || echo "  Rust native FAILED"
rustc --target x86_64-apple-darwin "$D/h.rs" -o "$D/hx" 2>/dev/null && echo "  cross x86_64-apple-darwin: $(file "$D/hx" | sed 's/.*: //')" || echo "  cross x86_64 FAILED"
rustc --target wasm32-unknown-unknown --crate-type=cdylib "$D/h.rs" -o "$D/h.wasm" 2>/dev/null && echo "  cross wasm32: $(file "$D/h.wasm" | sed 's/.*: //' | cut -c1-40)" || echo "  wasm32 (needs cdylib; ok)"
echo
echo "=== codec/image libs discoverable via pkg-config (for FFI vs zen crates) ==="
for l in libpng libwebp libjxl libavif libheif lcms2 libjpeg libtiff-4 aom dav1d libde265 zlib libzstd; do
  v=$(pkg-config --modversion "$l" 2>/dev/null); printf "  %s=%s" "$l" "${v:-NO}";
done; echo
echo
echo "=== gcc + llvm extras ==="
printf "  gcc:        "; ls /opt/homebrew/bin/gcc-* 2>/dev/null | head -1
printf "  clang-fmt:  "; /opt/homebrew/opt/llvm/bin/clang-format --version 2>&1 | head -1
printf "  clang-tidy: "; /opt/homebrew/opt/llvm/bin/clang-tidy --version 2>&1 | head -1
printf "  lldb:       "; lldb --version 2>&1 | head -1
printf "  lld:        "; /opt/homebrew/opt/llvm/bin/ld64.lld --version 2>&1 | head -1
rm -rf "$D"
echo "VERIFY DONE"
