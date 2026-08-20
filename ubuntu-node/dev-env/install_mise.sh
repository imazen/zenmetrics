#!/usr/bin/env bash
# Install mise (https://mise.jdx.dev/) and apply the zen workspace tool fleet.
#
# Run on a fresh ARM (or x86) dev box:
#   scp zenmetrics/ubuntu-node/dev-env/mise.aarch64.toml <host>:~/zen-mise.toml
#   ssh <host> 'bash -s' < ~/work/zen/scripts/install_mise.sh
#
# Or do it remotely:
#   rsync -a zenmetrics/ubuntu-node/dev-env/mise.aarch64.toml arm-zen:~/.config/mise/config.toml
#   ssh arm-zen 'bash -s' < ~/work/zen/scripts/install_mise.sh
#
# Idempotent. Safe to re-run.
set -euo pipefail
log() { printf "\n=== %s ===\n" "$*"; }

# We always invoke this for the `ubuntu` user (the parity user on Hetzner +
# Oracle). If started as root, refuse — the caller should ssh as ubuntu
# directly, or use:
#   ssh root@<host> 'sudo -u ubuntu bash -s' < install_mise.sh
if [ "$(id -u)" -eq 0 ]; then
  echo "ERROR: install_mise.sh must run as a regular user, not root." >&2
  echo "       Either ssh as ubuntu, or use:" >&2
  echo "         ssh root@<host> 'sudo -u ubuntu bash -s' < install_mise.sh" >&2
  exit 1
fi

### MAIN
log "Bootstrap mise"
if ! command -v mise >/dev/null 2>&1; then
  curl -fsSL https://mise.run | sh
fi
export PATH="$HOME/.local/bin:$PATH"
mise --version

log "Wire mise into shell init (idempotent)"
BASHRC="$HOME/.bashrc"
if ! grep -q 'mise activate bash' "$BASHRC" 2>/dev/null; then
  printf '\n# mise: load activated tools in every interactive shell\neval "$(~/.local/bin/mise activate bash)"\n' >> "$BASHRC"
fi
PROFILE="$HOME/.profile"
if [ -f "$PROFILE" ] && ! grep -q 'mise activate' "$PROFILE" 2>/dev/null; then
  printf '\n# mise: also load in non-interactive login shells\neval "$(~/.local/bin/mise activate bash --shims)"\n' >> "$PROFILE"
fi

log "Stage the zen .mise.toml as the user's global config"
mkdir -p "$HOME/.config/mise"
# Caller (the SCP/rsync step) is responsible for landing the file at one of:
#   ~/zen-mise.toml      (preferred staging path)
#   ~/.config/mise/config.toml  (final canonical path)
# If only the staging path exists, move it. If neither exists, fail loud.
if [ -f "$HOME/zen-mise.toml" ] && [ ! -f "$HOME/.config/mise/config.toml" ]; then
  mv "$HOME/zen-mise.toml" "$HOME/.config/mise/config.toml"
fi
if [ ! -f "$HOME/.config/mise/config.toml" ]; then
  echo "ERROR: no .mise.toml found at \$HOME/.config/mise/config.toml" >&2
  echo "       Caller must scp it before running this script." >&2
  exit 1
fi
echo "Using config: $HOME/.config/mise/config.toml"
mise trust "$HOME/.config/mise/config.toml" || true

log "Install the tool fleet"
# Use --yes to avoid any prompts; --jobs 4 keeps the box responsive
mise install --yes --jobs 4 || true

log "Final versions"
mise list 2>&1 | head -60
echo "---direct probes---"
for t in jj just samply rg fd bat fzf hyperfine gh oxipng mdbook node python rustc; do
  printf "  %-15s %s\n" "$t" "$(mise exec -- $t --version 2>/dev/null | head -1 || echo MISSING)"
done

log "DONE"
