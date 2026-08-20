#!/usr/bin/env bash
# Parity setup for ARM (or x86) zen dev boxes — both Hetzner CAX and Oracle
# A1.Flex. Run remotely:
#
#   ssh arm-zen 'sudo -u ubuntu bash -s' < ~/work/zen/scripts/setup-arm-box.sh
#
# Layered design:
#   1. APT layer    — system libs / dev headers / fonts (root, apt-managed)
#   2. mise layer   — user-space toolchain + CLI fleet (ubuntu, mise-managed)
#                     See zenmetrics/ubuntu-node/dev-env/mise.aarch64.toml for the tool list.
#   3. extras       — Node LTS + Claude Code CLI (npm), cloud CLIs
#                     (hcloud, s5cmd, rclone) which are outside the dev
#                     toolchain scope.
#
# Idempotent. Safe to re-run. Skips any step whose output already exists.
set -euo pipefail
log() { printf "\n=== %s ===\n" "$*"; }

if [ "$(id -u)" -eq 0 ]; then
  echo "ERROR: run as the ubuntu user, not root." >&2
  echo "       Use: ssh root@<host> 'sudo -u ubuntu bash -s' < setup-arm-box.sh" >&2
  exit 1
fi

# ──────────────────────────────────────────────────────────────────────────
# Step 1: APT system libraries + dev headers (sudo, root scope)
# ──────────────────────────────────────────────────────────────────────────
log "APT delta (system libs + dev headers; CUDA intentionally excluded)"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  ninja-build meson mold tig \
  valgrind heaptrack \
  ghostscript protobuf-compiler \
  libjpeg-dev libpng-dev libtiff-dev libwebp-dev libsqlite3-dev libbz2-dev \
  libdeflate-dev liblzma-dev libffi-dev libzstd-dev libtbb-dev \
  libnuma-dev libudev-dev zlib1g-dev libcap-dev libssl-dev libpq-dev \
  pipx \
  || true

# ──────────────────────────────────────────────────────────────────────────
# Step 2: mise + the zen tool fleet (user scope)
# ──────────────────────────────────────────────────────────────────────────
# Note: mise owns rust, node, python, jj, just, samply, ripgrep/fd/bat/fzf,
# hyperfine, the cargo-* subcommand fleet, mdbook, wasm-tools, gh, oxipng.
# The .mise.toml lives in this script's repo and gets staged at
# ~/.config/mise/config.toml.
log "mise + tool fleet"
if ! command -v mise >/dev/null 2>&1; then
  curl -fsSL https://mise.run | sh
fi
export PATH="$HOME/.local/bin:$PATH"
mise --version

mkdir -p "$HOME/.config/mise"
# Caller must SCP the .mise.toml to one of:
#   ~/zen-mise.toml                 (preferred staging path; this script moves it)
#   ~/.config/mise/config.toml      (final canonical path)
if [ -f "$HOME/zen-mise.toml" ] && [ ! -f "$HOME/.config/mise/config.toml" ]; then
  mv "$HOME/zen-mise.toml" "$HOME/.config/mise/config.toml"
fi
if [ ! -f "$HOME/.config/mise/config.toml" ]; then
  echo "WARN: no .mise.toml found; skipping mise install." >&2
  echo "      scp zenmetrics/ubuntu-node/dev-env/mise.aarch64.toml <host>:~/zen-mise.toml first." >&2
else
  mise trust "$HOME/.config/mise/config.toml" || true
  # GITHUB_TOKEN avoids the rate-limit failure mode that bites bulk binstalls.
  # Caller can pass it via: ssh -o SendEnv=GITHUB_TOKEN ...
  # We also write it to ~/.config/binstall/credentials.toml so binstall picks
  # it up even when the env doesn't propagate through `mise exec` subprocesses.
  if [ -n "${GH_TOKEN:-${GITHUB_TOKEN:-}}" ]; then
    mkdir -p "$HOME/.config/binstall"
    cat > "$HOME/.config/binstall/credentials.toml" <<EOF
[github]
token = "${GH_TOKEN:-$GITHUB_TOKEN}"
EOF
    chmod 600 "$HOME/.config/binstall/credentials.toml"
  fi
  # --jobs 1 serializes through binstall's GitHub API (with the token above we
  # could go higher, but serialization keeps the log readable on failure).
  mise install --yes --jobs 1 || true
fi

# Wire mise into login shells
BASHRC="$HOME/.bashrc"
if ! grep -q 'mise activate' "$BASHRC" 2>/dev/null; then
  printf '\n# mise: load activated tools\neval "$(~/.local/bin/mise activate bash)"\n' >> "$BASHRC"
fi

# ──────────────────────────────────────────────────────────────────────────
# Step 3: Tools outside mise's scope
# ──────────────────────────────────────────────────────────────────────────
log "Cargo install for tools without prebuilt ARM binaries (dssim, butteraugli-cli, git-delta)"
. "$HOME/.cargo/env" 2>/dev/null || eval "$(mise activate bash)"
for crate in dssim butteraugli-cli git-delta; do
  command -v "${crate%-cli}" >/dev/null 2>&1 || cargo install --locked "$crate" || true
done

log "Node LTS + Claude Code CLI (via nvm, isolated from mise's node)"
# We keep node-via-nvm here (not mise) because Claude CLI ships globally
# and we want a stable LTS line independent of any project node version.
if [ ! -d "$HOME/.nvm" ]; then
  curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
fi
export NVM_DIR="$HOME/.nvm"
set +u; [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"; set -u
nvm install --lts >/dev/null 2>&1 || true
set +u; nvm use --lts >/dev/null 2>&1 || true; set -u
command -v claude >/dev/null 2>&1 || npm install -g @anthropic-ai/claude-code

log "Cloud CLIs (hcloud, s5cmd, rclone) — root install to /usr/local/bin"
if ! command -v hcloud >/dev/null 2>&1; then
  TMP=$(mktemp -d)
  curl -fsSL "https://github.com/hetznercloud/cli/releases/latest/download/hcloud-linux-arm64.tar.gz" -o "$TMP/hcloud.tgz"
  tar -xzf "$TMP/hcloud.tgz" -C "$TMP" hcloud
  sudo install -m755 "$TMP/hcloud" /usr/local/bin/hcloud
  rm -rf "$TMP"
fi
if ! command -v s5cmd >/dev/null 2>&1; then
  TMP=$(mktemp -d)
  S5_VER=$(curl -fsSL https://api.github.com/repos/peak/s5cmd/releases/latest | grep tag_name | cut -d\" -f4 | tr -d v)
  curl -fsSL "https://github.com/peak/s5cmd/releases/download/v${S5_VER}/s5cmd_${S5_VER}_Linux-arm64.tar.gz" -o "$TMP/s5cmd.tgz"
  tar -xzf "$TMP/s5cmd.tgz" -C "$TMP" s5cmd
  sudo install -m755 "$TMP/s5cmd" /usr/local/bin/s5cmd
  rm -rf "$TMP"
fi
command -v rclone >/dev/null 2>&1 || curl -fsSL https://rclone.org/install.sh | sudo bash

# OCI CLI via pipx (Ubuntu 24.04 PEP 668 blocks bare `pip3 install`). Falls back
# to a hand-rolled venv if pipx-from-apt is too old. Not gating — hcloud is the
# primary cloud tool on this box.
if ! command -v oci >/dev/null 2>&1; then
  pipx ensurepath 2>/dev/null || true
  pipx install oci-cli >/dev/null 2>&1 || \
    { python3 -m venv "$HOME/.venvs/oci" 2>/dev/null && \
      "$HOME/.venvs/oci/bin/pip" install --quiet oci-cli && \
      mkdir -p "$HOME/.local/bin" && \
      ln -sf "$HOME/.venvs/oci/bin/oci" "$HOME/.local/bin/oci"; } || \
    echo "WARN: oci install failed (non-fatal)" >&2
fi

# fd symlink (Ubuntu names it fdfind; mise also installs it, but the apt fdfind may be present)
if command -v fdfind >/dev/null 2>&1 && [ ! -e /usr/local/bin/fd ]; then
  sudo ln -sf "$(command -v fdfind)" /usr/local/bin/fd 2>/dev/null || true
fi

# ──────────────────────────────────────────────────────────────────────────
# Final report
# ──────────────────────────────────────────────────────────────────────────
log "Final versions"
eval "$(~/.local/bin/mise activate bash)" 2>/dev/null || true
. "$HOME/.cargo/env" 2>/dev/null || true
set +u; [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" >/dev/null 2>&1; set -u
for t in rustc cargo node python3 jj just samply rg fd bat fzf hyperfine \
         cargo-nextest cargo-asm cargo-deny gh hcloud s5cmd rclone claude \
         dssim delta mdbook oxipng oci; do
  v=$(command -v "$t" >/dev/null 2>&1 && "$t" --version 2>/dev/null | head -1 || echo "MISSING")
  printf "  %-15s %s\n" "$t" "$v"
done

log "DONE"
