#!/bin/bash
# zen-mac-devsetup.sh -- comprehensive Rust + C development environment for macOS (Apple Silicon).
# Self-contained: installs Xcode Command Line Tools headless if missing, then everything else.
# Non-interactive; needs passwordless sudo (see /etc/sudoers.d/90-lilith-nopasswd). Tolerant of
# individual formula failures (logs them, keeps going). Idempotent -- safe to re-run.
#   Run detached:  nohup bash zen-mac-devsetup.sh > ~/zen-devsetup.log 2>&1 &
set -uo pipefail
export NONINTERACTIVE=1 HOMEBREW_NO_ENV_HINTS=1 HOMEBREW_NO_AUTO_UPDATE=1
BREW=/opt/homebrew/bin/brew
mark(){ echo; echo "########## $(date -u +%H:%M:%S) $* ##########"; echo; }

mark "STAGE 1: Xcode Command Line Tools (headless -- the GUI installer can't run over SSH)"
if xcode-select -p >/dev/null 2>&1 && /usr/bin/clang --version >/dev/null 2>&1; then
  echo "CLT already present: $(xcode-select -p)"
else
  # softwareupdate honors this trigger file to expose the CLT package to a headless install.
  TRIG=/tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
  sudo -n touch "$TRIG"
  LABEL=$(softwareupdate -l 2>/dev/null | grep -E 'Label: Command Line Tools' | sed 's/.*Label: //' | sort -V | tail -1)
  echo "installing CLT: '${LABEL:-<none found>}'"
  [ -n "$LABEL" ] && sudo -n softwareupdate -i "$LABEL" --verbose
  sudo -n rm -f "$TRIG"
fi
/usr/bin/clang --version | head -1 || { echo "FATAL: no C compiler after CLT install"; exit 1; }

mark "STAGE 2: Homebrew"
if [ -x "$BREW" ]; then echo "brew already installed"; else
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
eval "$($BREW shellenv)"
grep -q 'brew shellenv' ~/.zprofile 2>/dev/null || echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
$BREW --version | head -1

mark "STAGE 3: C/C++ toolchain + build systems"
$BREW install --quiet llvm gcc cmake ninja meson make autoconf automake libtool pkg-config \
  ccache nasm yasm bison flex m4 gawk coreutils gnu-sed gnu-tar findutils grep zig || true

mark "STAGE 4: codec/image reference libraries (FFI + comparison vs zen crates)"
$BREW install --quiet jpeg-turbo libpng zlib webp libavif aom dav1d libheif jpeg-xl \
  openexr little-cms2 libtiff giflib openjpeg zstd brotli lz4 xz highway libde265 || true

mark "STAGE 5: general dev CLI + languages"
$BREW install --quiet gh git-lfs jq ripgrep fd bat eza fzf tmux htop btop wget tree \
  git-delta direnv tokei graphviz hyperfine node uv pipx protobuf capnp awscli wasmtime s5cmd || true

mark "STAGE 6: container runtime (for cross, cargo-zigbuild fallback, zenfleet exec image)"
$BREW install --quiet colima docker docker-buildx || true

mark "STAGE 7: Rust via rustup (stable + nightly, components, cross targets)"
if [ -x "$HOME/.cargo/bin/rustup" ]; then echo "rustup present"; else
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile default --default-toolchain stable
fi
. "$HOME/.cargo/env"
rustup toolchain install stable nightly --profile default || true
rustup component add rustfmt clippy rust-src rust-analyzer llvm-tools-preview --toolchain stable || true
rustup component add rustfmt clippy rust-src miri --toolchain nightly || true
rustup target add aarch64-apple-darwin x86_64-apple-darwin wasm32-unknown-unknown wasm32-wasip1 \
  aarch64-unknown-linux-gnu x86_64-unknown-linux-gnu i686-unknown-linux-gnu || true
grep -q 'cargo/env' ~/.zprofile 2>/dev/null || echo '. "$HOME/.cargo/env"' >> ~/.zprofile

mark "STAGE 8: cargo tools (cargo-binstall = prebuilt binaries, fast)"
$BREW install --quiet cargo-binstall || true
export PATH="$HOME/.cargo/bin:$PATH"
# NOTE: cargo-binstall aborts the whole list on the first failure unless --continue-on-failure;
# and `-y` already IS `--no-confirm` (passing both is a clap error). The `cargo flamegraph`
# subcommand ships in the crate named `flamegraph` (NOT `cargo-flamegraph`). tokei comes from brew.
cargo binstall -y --continue-on-failure \
  cargo-nextest cargo-watch cargo-edit cargo-outdated cargo-audit cargo-deny cargo-semver-checks \
  cargo-expand cargo-llvm-lines cargo-show-asm cargo-binutils cargo-insta cargo-hack cargo-machete \
  flamegraph sccache just bacon cargo-zigbuild cross wasm-pack wasm-tools cargo-wasi || true

mark "STAGE 8b: imazen internal cargo tools (built from GitHub source; not on crates.io)"
# cargo-read (`cargo read <crate>`), cargo-copter (rev-dep testing), cargo-superwork (release pipeline).
export PATH="$HOME/.cargo/bin:$PATH"
for t in cargo-read cargo-copter cargo-superwork; do
  echo "-- installing $t"
  cargo install --git "https://github.com/imazen/$t.git" 2>&1 || echo "FAILED: $t"
done

mark "STAGE 8c: AI / agent CLIs (Claude Code + herdr -> ~/.local/bin)"
# Claude Code (native installer, auto-updates); herdr = tmux-like multiplexer for AI coding agents.
curl -fsSL https://claude.ai/install.sh | bash || echo "FAILED: claude"
curl -fsSL https://herdr.dev/install.sh | sh   || echo "FAILED: herdr"

mark "STAGE 9: shell env for keg-only tools (llvm/bison/flex)"
if ! grep -q 'zen dev env' ~/.zprofile 2>/dev/null; then
cat >> ~/.zprofile <<'EOF'

# --- zen dev env ---
export PATH="$HOME/.local/bin:/opt/homebrew/opt/llvm/bin:/opt/homebrew/opt/bison/bin:/opt/homebrew/opt/flex/bin:$PATH"
export LIBRARY_PATH="/opt/homebrew/lib:${LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
# opt-in build cache: uncomment to route all cargo builds through sccache
# export RUSTC_WRAPPER=sccache
# --- end zen dev env ---
EOF
fi

mark "DONE -- versions"
. "$HOME/.cargo/env" 2>/dev/null || true
echo "clang:   $(clang --version 2>&1 | head -1)"
echo "llvm:    $(/opt/homebrew/opt/llvm/bin/clang --version 2>&1 | head -1)"
echo "gcc:     $(gcc-* --version 2>/dev/null | head -1 || $BREW list --versions gcc)"
echo "rustc:   $(rustc --version 2>&1)"
echo "cargo:   $(cargo --version 2>&1)"
echo "cmake:   $(cmake --version 2>&1 | head -1)"
echo "ninja:   $(ninja --version 2>&1)"
echo "nextest: $(cargo nextest --version 2>&1 | head -1)"
echo "s5cmd:   $(s5cmd version 2>&1 | head -1)"
echo "cargo-read:    $(cargo read --version 2>&1 | head -1)"
echo "cargo-copter:  $(cargo-copter --version 2>&1 | head -1)"
echo "claude:  $("$HOME/.local/bin/claude" --version 2>&1 | head -1)"
echo "herdr:   $("$HOME/.local/bin/herdr" --version 2>&1 | head -1)"
echo "ALL STAGES COMPLETE"
