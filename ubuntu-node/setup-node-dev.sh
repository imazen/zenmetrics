#!/usr/bin/env bash
# setup-node-dev.sh -- comprehensive Rust + C dev environment for an Ubuntu fleet node (x86_64).
# Non-interactive; needs passwordless sudo. Tolerant: filters apt to packages that exist in this
# release (apt is all-or-nothing otherwise) and waits on the apt lock. Idempotent-ish.
#   Run detached:  nohup bash setup-node-dev.sh > ~/zen-devsetup.log 2>&1 &
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
APT="sudo -n apt-get -o DPkg::Lock::Timeout=600 -y"
ME="$(id -un)"
mark(){ echo; echo "########## $(date -u +%H:%M:%S) $* ##########"; echo; }

mark "STAGE 1: apt update + C/C++ toolchain + build systems + codec dev libs + Linux profilers"
$APT update || true
CORE="build-essential clang llvm lld lldb clang-format clang-tidy clangd cmake ninja-build meson \
  autoconf automake libtool pkg-config ccache nasm yasm bison flex m4 gawk \
  git git-lfs curl wget jq ripgrep fd-find bat fzf tmux htop btop tree file unzip zip xz-utils \
  python3 python3-pip python3-venv pipx gh graphviz direnv protobuf-compiler capnproto \
  valgrind gdb heaptrack linux-tools-common linux-tools-generic hyperfine tokei"
CODEC="libjpeg-dev libturbojpeg0-dev libjpeg-turbo8-dev libpng-dev libwebp-dev libavif-dev \
  libaom-dev libdav1d-dev libheif-dev libjxl-dev libopenexr-dev liblcms2-dev libtiff-dev \
  libgif-dev libopenjp2-7-dev zlib1g-dev libzstd-dev libbrotli-dev liblz4-dev liblzma-dev \
  libde265-dev libx265-dev libhwy-dev libvmaf-dev"
AVAIL=$(for p in $CORE $CODEC; do apt-cache show "$p" >/dev/null 2>&1 && printf '%s ' "$p"; done)
echo "available -> installing: $AVAIL"
$APT install --no-install-recommends $AVAIL || true

mark "STAGE 2: docker (for cross-compiling + the fleet worker)"
$APT install docker.io || true
sudo -n usermod -aG docker "$ME" || true
sudo -n systemctl enable --now docker || true

mark "STAGE 3: extra tools not in apt (uv, s5cmd, wasmtime, zig -- best-effort)"
curl -LsSf https://astral.sh/uv/install.sh | sh || echo "FAILED: uv"
S5VER=2.3.0
( curl -fsSL "https://github.com/peak/s5cmd/releases/download/v${S5VER}/s5cmd_${S5VER}_Linux-64bit.tar.gz" \
    | sudo -n tar xz -C /usr/local/bin s5cmd ) 2>/dev/null && echo "s5cmd ok" || echo "FAILED: s5cmd"
curl -fsSL https://wasmtime.dev/install.sh | bash || echo "FAILED: wasmtime"

mark "STAGE 4: Rust via rustup (stable + nightly, components, cross targets)"
if [ -x "$HOME/.cargo/bin/rustup" ]; then echo "rustup present"; else
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile default --default-toolchain stable
fi
. "$HOME/.cargo/env"
rustup toolchain install stable nightly --profile default || true
rustup component add rustfmt clippy rust-src rust-analyzer llvm-tools-preview --toolchain stable || true
rustup component add rustfmt clippy rust-src miri --toolchain nightly || true
rustup target add x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu i686-unknown-linux-gnu \
  x86_64-unknown-linux-musl wasm32-unknown-unknown wasm32-wasip1 || true

mark "STAGE 5: cargo tools (cargo-binstall = prebuilt binaries)"
if [ ! -x "$HOME/.cargo/bin/cargo-binstall" ]; then
  curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash || cargo install cargo-binstall || true
fi
export PATH="$HOME/.cargo/bin:$PATH"
cargo binstall -y --continue-on-failure \
  cargo-nextest cargo-watch cargo-edit cargo-outdated cargo-audit cargo-deny cargo-semver-checks \
  cargo-expand cargo-llvm-lines cargo-show-asm cargo-binutils cargo-insta cargo-hack cargo-machete \
  flamegraph sccache just bacon cargo-zigbuild cross wasm-pack wasm-tools cargo-wasi || true

mark "STAGE 6: imazen internal cargo tools (from GitHub source)"
for t in cargo-read cargo-copter cargo-superwork; do
  echo "-- installing $t"; cargo install --git "https://github.com/imazen/$t.git" 2>&1 || echo "FAILED: $t"
done

mark "STAGE 7: AI / agent CLIs (Claude Code + herdr -> ~/.local/bin)"
curl -fsSL https://claude.ai/install.sh | bash || echo "FAILED: claude"
curl -fsSL https://herdr.dev/install.sh | sh   || echo "FAILED: herdr"

mark "STAGE 8: shell env (~/.bashrc)"
mkdir -p ~/.local/bin
command -v batcat >/dev/null && ln -sf "$(command -v batcat)" ~/.local/bin/bat
command -v fdfind >/dev/null && ln -sf "$(command -v fdfind)" ~/.local/bin/fd
if ! grep -q 'zen dev env' ~/.bashrc 2>/dev/null; then
cat >> ~/.bashrc <<'EOF'

# --- zen dev env ---
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
export PATH="$HOME/.local/bin:$PATH"
# opt-in build cache: uncomment to route all cargo builds through sccache
# export RUSTC_WRAPPER=sccache
# --- end zen dev env ---
EOF
fi

mark "DONE -- versions"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
echo "gcc:     $(gcc --version 2>&1 | head -1)"
echo "clang:   $(clang --version 2>&1 | head -1)"
echo "cmake:   $(cmake --version 2>&1 | head -1)"
echo "ninja:   $(ninja --version 2>&1)"
echo "valgrind:$(valgrind --version 2>&1)"
echo "rustc:   $(rustc --version 2>&1)"
echo "cargo:   $(cargo --version 2>&1)"
echo "nextest: $(cargo nextest --version 2>&1 | head -1)"
echo "docker:  $(docker --version 2>&1)"
echo "cargo-read:   $(cargo read --version 2>&1 | head -1)"
echo "cargo-copter: $(cargo-copter --version 2>&1 | head -1)"
echo "claude:  $($HOME/.local/bin/claude --version 2>&1 | head -1)"
echo "herdr:   $($HOME/.local/bin/herdr --version 2>&1 | head -1)"
echo "ALL STAGES COMPLETE"
